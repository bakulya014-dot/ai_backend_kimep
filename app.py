"""
Flask API for AI Research Module.

Endpoints:
- GET /api/health
- GET /api/model-info
- POST /api/predict
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "student_score_model.joblib"
META_PATH = BASE_DIR / "model" / "model_info.json"

app = Flask(__name__)
CORS(app)

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
model_info = (
    json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
)


def build_explanation(study: float, sleep: float, tests: float, score: float) -> str:
    parts = []
    if study >= 6:
        parts.append("study hours are strong")
    elif study < 3:
        parts.append("study hours are low")
    else:
        parts.append("study hours are moderate")

    if 7 <= sleep <= 9:
        parts.append("sleep pattern looks healthy")
    else:
        parts.append("sleep may be affecting focus")

    if tests >= 6:
        parts.append("practice test frequency is high")
    elif tests <= 2:
        parts.append("practice tests are limited")
    else:
        parts.append("practice tests are moderate")

    return (
        f"Predicted score is {score:.2f}. This prediction suggests that "
        + ", ".join(parts)
        + "."
    )


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.get("/api/model-info")
def get_model_info():
    if not model_info:
        return jsonify({"error": "Model metadata not found. Run train_model.py first."}), 500
    return jsonify(model_info)


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        study = float(data.get("StudyHours"))
        sleep = float(data.get("SleepHours"))
        tests = float(data.get("PracticeTests"))
    except (TypeError, ValueError):
        return jsonify({"error": "Please provide numeric StudyHours, SleepHours, PracticeTests."}), 400

    if model is not None:
        x = np.array([[study, sleep, tests]], dtype=float)
        pred = float(model.predict(x)[0])
        model_used = model_info.get("best_model", "Saved Model")
    else:
        # Fallback linear formula to keep demo functional before training.
        pred = 28.0 + 5.1 * study + 2.1 * sleep + 3.2 * tests
        model_used = "Fallback Demo Formula"
    pred = max(0.0, min(100.0, pred))
    explanation = build_explanation(study, sleep, tests, pred)
    return jsonify(
        {
            "predicted_score": round(pred, 2),
            "explanation": explanation,
            "model_used": model_used,
        }
    )


if __name__ == "__main__":
    # For local development only.
    app.run(host="0.0.0.0", port=5000, debug=True)
