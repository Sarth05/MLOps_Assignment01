"""FastAPI inference API with request logging and simple health endpoint."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from heart_ml.features import FEATURES

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart_ml.api")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/heart_disease_pipeline.joblib"))

app = FastAPI(
    title="Heart Disease Risk Classification API",
    version="1.0.0",
    description="Educational demonstration only; not for medical diagnosis.",
)
_model = None


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=1, le=120)
    sex: float = Field(ge=0, le=1)
    cp: float = Field(ge=1, le=4)
    trestbps: float = Field(ge=40, le=300)
    chol: float = Field(ge=50, le=800)
    fbs: float = Field(ge=0, le=1)
    restecg: float = Field(ge=0, le=2)
    thalach: float = Field(ge=40, le=250)
    exang: float = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-5, le=15)
    slope: float = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=4)
    thal: float = Field(ge=3, le=7)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(json.dumps({
        "event": "http_request",
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": elapsed_ms,
    }))
    response.headers["X-Process-Time-ms"] = str(elapsed_ms)
    return response


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run `python -m heart_ml.train` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_path_exists": MODEL_PATH.exists()}


@app.post("/predict")
def predict(payload: PatientFeatures) -> dict[str, Any]:
    try:
        model = load_model()
        row = pd.DataFrame([payload.model_dump()], columns=FEATURES)
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        return {
            "prediction": prediction,
            "label": "presence_of_disease" if prediction == 1 else "absence_of_disease",
            "confidence": float(max(probabilities)),
            "probabilities": {
                "absence_of_disease": float(probabilities[0]),
                "presence_of_disease": float(probabilities[1]),
            },
            "disclaimer": "Educational model only; not clinically validated. Do not use for medical decisions.",
        }
    except FileNotFoundError as exc:
        logger.exception("Model file unavailable")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed; see service logs.") from exc
