"""
fraud_detection.py — Fixed version
Fixes applied:
  1. is_cutoff added to extra_features (was in features list but never created → KeyError)
  2. All feature columns guaranteed to exist before building X
  3. count_hub_scans kept but noted as unused
  4. simulate_labels branch left as stub with clear comment
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for Flask
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

plt.style.use("seaborn-v0_8")

# ── Helpers ──────────────────────────────────────────────────────────────────

def count_hub_scans(x):
    """Count hub scans from a delimited string. Currently unused."""
    if pd.isna(x):
        return 0
    s = str(x)
    for sep in ['|', ';', '>', '->', ',']:
        if sep in s:
            return len([p for p in s.split(sep) if p.strip()])
    return len([p for p in s.split() if p.strip()])


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    inp              = args.input
    out              = args.output
    contamination    = args.contamination
    simulate_labels  = args.simulate_labels
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    print(f"Loading data from: {inp}")
    df = pd.read_csv(inp, low_memory=False)
    print("Columns detected:", list(df.columns))
    print("Rows:", len(df))

    # ── Column mapping (Delhivery dataset) ───────────────────────────────────
    pickup_col   = "od_start_time"
    delivery_col = "od_end_time"
    tracking_col = "trip_uuid"
    source_col   = "source_center"
    dest_col     = "destination_center"

    # Delivery time
    if pickup_col in df.columns and delivery_col in df.columns:
        df[pickup_col]   = pd.to_datetime(df[pickup_col],   errors="coerce")
        df[delivery_col] = pd.to_datetime(df[delivery_col], errors="coerce")
        df["delivery_time_hours"] = (
            df[delivery_col] - df[pickup_col]
        ).dt.total_seconds() / 3600.0
    else:
        print("Warning: pickup/delivery columns missing. delivery_time_hours → NaN")
        df["delivery_time_hours"] = np.nan

    # Duplicate shipment flag
    if tracking_col in df.columns:
        df["duplicate_flag"] = df.duplicated(subset=[tracking_col], keep=False).astype(int)
    else:
        df["duplicate_flag"] = 0

    # Same-city flag
    if source_col in df.columns and dest_col in df.columns:
        df["same_city"] = (
            df[source_col].astype(str).str.lower()
            == df[dest_col].astype(str).str.lower()
        ).astype(int)
    else:
        df["same_city"] = 0

    # ── Extra numeric features ────────────────────────────────────────────────
    # BUG FIX: is_cutoff was in `features` but never created; added here
    extra_features = [
        "actual_distance_to_destination",
        "actual_time",
        "osrm_time",
        "osrm_distance",
        "factor",
        "segment_actual_time",
        "segment_osrm_time",
        "segment_osrm_distance",
        "segment_factor",
        "is_cutoff",          # ← FIXED: was missing, caused KeyError
    ]
    for col in extra_features:
        if col not in df.columns:
            df[col] = np.nan
        else:
            # Coerce boolean-like is_cutoff to numeric
            if df[col].dtype == object:
                df[col] = df[col].map({"TRUE": 1, "FALSE": 0, True: 1, False: 0}).fillna(0)

    # ── Feature list ─────────────────────────────────────────────────────────
    features = [
        "delivery_time_hours",
        "actual_time",
        "osrm_time",
        "actual_distance_to_destination",
        "osrm_distance",
        "factor",
        "segment_factor",
        "is_cutoff",
        "duplicate_flag",
        "same_city",
    ]

    # BUG FIX: guarantee every feature column exists before slicing
    for col in features:
        if col not in df.columns:
            df[col] = 0

    X = df[features].fillna(0).astype(float)

    # ── Standardise ──────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── IsolationForest ───────────────────────────────────────────────────────
    print(f"Training IsolationForest (contamination={contamination})...")
    iso      = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(X_scaled)
    iso_pred = iso.predict(X_scaled)
    iso_scores = iso.decision_function(X_scaled)

    df["isolation_anomaly"] = (iso_pred == -1).astype(int)
    df["isolation_score"]   = iso_scores
    df["Anomaly"]           = df["isolation_anomaly"].map({0: "Normal", 1: "Fraud"})

    # ── Save results ──────────────────────────────────────────────────────────
    result_cols = [tracking_col, "isolation_anomaly", "isolation_score", "Anomaly"] + features
    if tracking_col not in df.columns:
        df.insert(0, "tracking_id_auto", [f"row_{i}" for i in range(len(df))])
        result_cols.insert(0, "tracking_id_auto")

    # Include source/dest names for the frontend route display
    for extra_col in ["source_name", "destination_name"]:
        if extra_col in df.columns and extra_col not in result_cols:
            result_cols.append(extra_col)

    out_df = df[result_cols]
    out_df.to_csv(out, index=False)
    print(f"Saved results to: {out}")

    # Top anomalies preview
    top = df.sort_values("isolation_score").head(10)
    display_cols = [tracking_col, "isolation_score", "isolation_anomaly", "Anomaly"] + features
    print("\nTop 10 anomaly candidates:")
    print(top[display_cols].to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        out_dir = os.path.dirname(out) or "."

        plt.figure(figsize=(8, 4))
        sns.histplot(df["delivery_time_hours"].dropna(), bins=60, kde=True)
        plt.title("Delivery time (hours) distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "delivery_time_dist.png"))
        plt.close()
        print("Saved delivery_time_dist.png")

        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            x="delivery_time_hours",
            y="actual_distance_to_destination",
            hue="isolation_anomaly",
            data=df,
            palette={0: "steelblue", 1: "crimson"},
            alpha=0.6,
        )
        plt.title("Anomaly (red) vs Normal (blue)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "anomaly_scatter.png"))
        plt.close()
        print("Saved anomaly_scatter.png")
    except Exception as e:
        print("Plotting skipped:", e)

    # ── simulate_labels stub ──────────────────────────────────────────────────
    if simulate_labels:
        print("\n[simulate_labels] Not implemented — argument accepted but no classifier trained.")

    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",          required=True)
    p.add_argument("--output",         default="fraud_results.csv")
    p.add_argument("--contamination",  type=float, default=0.05)
    p.add_argument("--simulate_labels",action="store_true")
    main(p.parse_args())
