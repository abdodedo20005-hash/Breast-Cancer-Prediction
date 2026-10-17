import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.predict import get_artifacts, model_info, predict
from src.schemas import PatientFeatures, PredictionResponse

logging.basicConfig(level=logging.INFO)

DISCLAIMER = "Educational project. Not a medical device and not a substitute for professional diagnosis."


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_artifacts()  # fail fast at startup if the model is missing
    yield


app = FastAPI(
    title="Breast Cancer Prediction API",
    version="1.0.0",
    description=DISCLAIMER,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def info():
    return model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(payload: PatientFeatures):  # type: ignore[valid-type]
    try:
        return predict(payload.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
