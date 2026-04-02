import base64
import os

import pandas as pd
from backend import detect_fraud
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER  = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return jsonify({"message": "FraudGuard Shipment Detection API running"})


@app.route("/upload", methods=["POST"])
def upload_file():


    if request.is_json:
        payload       = request.get_json(force=True)
        filename      = secure_filename(payload.get("filename", "upload.csv"))
        file_b64      = payload.get("file", "")
        contamination = float(payload.get("contamination", 0.05))

        if not filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files are supported"}), 400

        try:
            file_bytes = base64.b64decode(file_b64)
        except Exception:
            return jsonify({"error": "Invalid base64 file data"}), 400

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(file_bytes)

    else:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file     = request.files["file"]
        filename = secure_filename(file.filename)

        if not filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files are supported"}), 400

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        contamination = float(request.form.get("contamination", 0.05))

    contamination = max(0.01, min(0.5, contamination))

    base       = filename.replace(".csv", "_results.csv")
    output_csv = os.path.join(RESULTS_FOLDER, base)

    try:
        detect_fraud(filepath, output_csv=output_csv, contamination=contamination)
    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

    df = pd.read_csv(output_csv)
    return df.to_json(orient="records")


@app.route("/results/<path:filename>")
def results_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)


@app.route("/results", methods=["GET"])
def get_results():
    files = sorted(
        [f for f in os.listdir(RESULTS_FOLDER) if f.endswith("_results.csv")],
        key=lambda f: os.path.getmtime(os.path.join(RESULTS_FOLDER, f)),
        reverse=True,
    )
    if not files:
        return jsonify({"error": "No results found. Upload a CSV first."}), 404

    df = pd.read_csv(os.path.join(RESULTS_FOLDER, files[0]))
    return df.to_json(orient="records")


if __name__ == "__main__":
    app.run(debug=True)
