# FraudGuard — Fraud Shipment Identification

A Scalable Framework for Anomaly-Based Fraud Detection in Shipment Systems

Published in IJEDR, March 2026, Volume 14, Issue 1

**Authors:** Lahari Talakokkula, Maaria Munawer, Ashritha Sanem
**Guide:** Mrs. Asma Begum (Assistant Professor)
**Department:** Artificial Intelligence and Data Science
**Institution:** Stanley College of Engineering and Technology for Women, Hyderabad

---

## What it does

FraudGuard detects fraudulent shipments in logistics data using the Isolation Forest machine learning algorithm. It scores every shipment, classifies it as Critical / High / Medium / Low risk, and displays results on an interactive dashboard.

---

## Files

| File | Description |
|---|---|
| `index.html` | Complete frontend — upload CSV, run detection, view results (works standalone in browser) |
| `app.py` | Flask REST API — receives CSV, runs detection, returns JSON |
| `fraud_detection.py` | Core ML pipeline — feature engineering + IsolationForest |
| `backend.py` | Bridge between Flask and the ML pipeline |
| `requirements.txt` | Python dependencies |

---

## How to run

### Option 1 — Frontend only (no server needed)
Just open `index.html` in any browser and upload your CSV. Everything runs in the browser.

### Option 2 — Full stack (Flask backend)

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Start the server:
```
python app.py
```

3. Open `index.html` in your browser.

---

## Dataset

Tested on the Delhivery logistics dataset (10,000 shipment records).
Columns used: `trip_uuid`, `od_start_time`, `od_end_time`, `actual_time`, `osrm_time`,
`actual_distance_to_destination`, `osrm_distance`, `factor`, `segment_factor`, `is_cutoff`

---

## Results

- Total shipments processed: 10,000
- Anomalies detected: 500 (5.0%)
- Normal shipments: 9,500 (95.0%)
- AUC-ROC (internal consistency): 1.000
- Cohen's d (separation strength): 4.28 (strong)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| CSV Parsing | PapaParse |
| ML (browser) | Custom JS IsolationForest port |
| Backend | Python, Flask |
| ML (server) | Scikit-learn IsolationForest |
| Data handling | Pandas, NumPy |
| Deployment | Vercel (frontend), Railway (backend) |

---

## License
For academic use. Published in IJEDR 2026.
