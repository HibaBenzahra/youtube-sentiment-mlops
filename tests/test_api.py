import time
from fastapi.testclient import TestClient

from src.api.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_batch_basic():
    with TestClient(app) as client:
        response = client.post(
            "/predict_batch",
            json={"comments": ["this is amazing", "this is terrible", "I watched it yesterday"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 3
        assert data["stats"]["total"] == 3
        assert data["predictions"][0]["sentiment"] == "positive"
        assert data["predictions"][1]["sentiment"] == "negative"


def test_stats_are_consistent():
    with TestClient(app) as client:
        response = client.post("/predict_batch", json={"comments": ["great", "bad", "okay", "love it"]})
        stats = response.json()["stats"]
        assert stats["positive"] + stats["neutral"] + stats["negative"] == stats["total"]
        total_pct = stats["positive_pct"] + stats["neutral_pct"] + stats["negative_pct"]
        assert 99.5 <= total_pct <= 100.5


def test_different_batch_sizes():
    with TestClient(app) as client:
        for size in [1, 10, 50, 200, 1000]:
            response = client.post("/predict_batch", json={"comments": ["nice video"] * size})
            assert response.status_code == 200
            assert response.json()["stats"]["total"] == size


def test_empty_batch_is_rejected():
    with TestClient(app) as client:
        response = client.post("/predict_batch", json={"comments": []})
        assert response.status_code == 422


def test_batch_too_large_is_rejected():
    with TestClient(app) as client:
        response = client.post("/predict_batch", json={"comments": ["x"] * 1001})
        assert response.status_code == 422


def test_invalid_payloads_are_rejected():
    with TestClient(app) as client:
        assert client.post("/predict_batch", json={}).status_code == 422
        assert client.post("/predict_batch", json={"comments": "not a list"}).status_code == 422
        assert client.post("/predict_batch", json={"comments": [1, 2, 3]}).status_code == 422
        assert client.post("/predict_batch", content="not json").status_code == 422


def test_empty_text_after_cleaning_is_neutral():
    with TestClient(app) as client:
        response = client.post("/predict_batch", json={"comments": ["😍😍😍", "مرحبا"]})
        assert response.status_code == 200
        for prediction in response.json()["predictions"]:
            assert prediction["sentiment"] == "neutral"
            assert prediction["confidence"] == 0


def test_response_time_for_50_comments_is_reasonable():
    with TestClient(app) as client:
        start = time.perf_counter()
        response = client.post("/predict_batch", json={"comments": ["nice video"] * 50})
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert response.status_code == 200
        assert elapsed_ms < 500


def test_cors_headers_present():
    with TestClient(app) as client:
        response = client.options(
            "/predict_batch",
            headers={
                "Origin": "chrome-extension://abc",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "*"
