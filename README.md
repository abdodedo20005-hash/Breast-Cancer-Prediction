# Breast Cancer Prediction

Machine-learning classifier that predicts whether a breast mass is **malignant or benign** from 30 cell-nucleus measurements (Wisconsin Diagnostic Breast Cancer dataset), served as a **FastAPI** service, a **Gradio** demo, and a **Docker** image.

> ⚠️ **Disclaimer:** educational project only. It is not a medical device and must not be used for real diagnosis.

**Live demo:** _add your Hugging Face Space link here_

## Results

| Metric | Value |
|---|---|
| Accuracy | _TODO_ |
| Recall (malignant) | _TODO_ |
| Precision (malignant) | _TODO_ |
| ROC-AUC | _TODO_ |

In a medical screening setting, **recall on the malignant class matters more than accuracy**: a missed malignant case costs more than a false alarm. The decision threshold is configurable (`THRESHOLD` env var) so it can be tuned toward higher recall.

_Add your confusion matrix / ROC curve image here._

## Project structure

```
data/         dataset
notebook/     EDA and training experiments
requirements/ dependency files (api / ui / dev)
models/       trained pipeline (model.joblib)
src/          inference code + FastAPI app
tests/        API tests (pytest)
app.py        Gradio demo
Dockerfile
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/requirements-ui.txt

python app.py                              # Gradio demo  -> http://localhost:7860
uvicorn src.main:app --reload              # REST API     -> http://localhost:8000/docs
```

## Docker

```bash
docker build -t breast-cancer-api .
docker run -p 8000:8000 breast-cancer-api
```

## API

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample.json
```

| Endpoint | Description |
|---|---|
| `GET /health` | liveness check |
| `GET /model-info` | model type, features, threshold |
| `POST /predict` | 30 features in, label + probabilities out |

Response:

```json
{"label": "Malignant", "malignant_probability": 0.98, "benign_probability": 0.02, "threshold": 0.5}
```

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `MALIGNANT_CLASS` | `1` | class label that means malignant in your training data |
| `THRESHOLD` | `0.5` | decision threshold on P(malignant) |
| `MODEL_PATH` | auto (`models/*.joblib`) | explicit path to the model file |

## Tests & CI

```bash
pip install -r requirements/requirements-dev.txt && pytest
```

GitHub Actions runs lint, tests, and a Docker build + smoke test on every push.
