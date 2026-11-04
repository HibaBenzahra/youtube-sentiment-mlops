import time
import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.utils.text_cleaning import clean_text

MODEL_PATH = "models/model.joblib"
VECTORIZER_PATH = "models/vectorizer.joblib"
TEST_PATH = "data/processed/test.csv"


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def vectorizer():
    return joblib.load(VECTORIZER_PATH)


@pytest.fixture(scope="module")
def test_data():
    return pd.read_csv(TEST_PATH).dropna()


def predict(model, vectorizer, texts):
    cleaned = [clean_text(t) for t in texts]
    return model.predict(vectorizer.transform(cleaned))


def test_accuracy_above_80_percent(model, vectorizer, test_data):
    pred = model.predict(vectorizer.transform(test_data["text"]))
    assert accuracy_score(test_data["label"], pred) >= 0.80


def test_f1_per_class_above_075(model, vectorizer, test_data):
    pred = model.predict(vectorizer.transform(test_data["text"]))
    scores = f1_score(test_data["label"], pred, average=None)
    assert all(score > 0.75 for score in scores)


def test_inference_time_under_100ms_for_50_comments(model, vectorizer, test_data):
    batch = test_data["text"].tolist()[:50]
    start = time.perf_counter()
    model.predict(vectorizer.transform(batch))
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 100


@pytest.mark.parametrize(
    "text",
    [
        "ok",
        "a",
        "word " * 2000,
        "😍😍😍",
        "this is great 😍",
        "c'est vraiment nul mais I love it",
        "مرحبا",
        "!!!???",
        "   ",
    ],
)
def test_edge_cases_do_not_crash(model, vectorizer, text):
    result = predict(model, vectorizer, [text])
    assert len(result) == 1
    assert result[0] in (-1, 0, 1)


def test_clear_positive_and_negative(model, vectorizer):
    result = predict(model, vectorizer, ["this video is amazing, I loved it", "worst video ever, total waste of time"])
    assert result[0] == 1
    assert result[1] == -1
