import pytest
from fastapi.testclient import TestClient

from src.features import EXAMPLE_BENIGN, EXAMPLE_MALIGNANT
from src.predict import find_model_path

pytestmark = pytest.mark.skipif(
    find_model_path() is None, reason="no trained model found in models/"
)


@pytest.fixture(scope="module")
def client():
    from src.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    assert client.get("/model-info").json()["n_features"] == 30


def test_predict_returns_valid_probabilities(client):
    r = client.post("/predict", json=EXAMPLE_MALIGNANT)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"Malignant", "Benign"}
    assert 0 <= body["malignant_probability"] <= 1
    assert abs(body["malignant_probability"] + body["benign_probability"] - 1) < 1e-3


def test_malignant_example_scores_higher_than_benign(client):
    m = client.post("/predict", json=EXAMPLE_MALIGNANT).json()["malignant_probability"]
    b = client.post("/predict", json=EXAMPLE_BENIGN).json()["malignant_probability"]
    assert m > b


def test_missing_field_rejected(client):
    payload = dict(EXAMPLE_BENIGN)
    payload.pop("radius_mean")
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_value_rejected(client):
    payload = dict(EXAMPLE_BENIGN, area_mean=-1)
    assert client.post("/predict", json=payload).status_code == 422
