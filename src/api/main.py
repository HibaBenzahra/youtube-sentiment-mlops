import time
import logging
from contextlib import asynccontextmanager
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PredictRequest, PredictResponse, HealthResponse, Prediction, Stats
from src.utils.text_cleaning import clean_text

MODEL_PATH = "models/model.joblib"
VECTORIZER_PATH = "models/vectorizer.joblib"
SENTIMENTS = {-1: "negative", 0: "neutral", 1: "positive"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sentiment-api")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"] = joblib.load(MODEL_PATH)
        state["vectorizer"] = joblib.load(VECTORIZER_PATH)
        logger.info("Modèle et vectoriseur chargés")
    except Exception as e:
        logger.error(f"Échec du chargement du modèle : {e}")
    yield
    state.clear()


app = FastAPI(title="YouTube Sentiment API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = "model" in state and "vectorizer" in state
    return HealthResponse(status="ok" if loaded else "error", model_loaded=loaded)


@app.post("/predict_batch", response_model=PredictResponse)
def predict_batch(request: PredictRequest):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    start = time.perf_counter()
    cleaned = [clean_text(c) for c in request.comments]
    probas = state["model"].predict_proba(state["vectorizer"].transform(cleaned))
    classes = state["model"].classes_

    predictions = []
    for original, text, proba in zip(request.comments, cleaned, probas):
        if text == "":
            label, confidence = 0, 0.0
        else:
            best = proba.argmax()
            label, confidence = int(classes[best]), float(proba[best])
        predictions.append(
            Prediction(
                text=original,
                sentiment=SENTIMENTS[label],
                label=label,
                confidence=round(confidence, 4),
            )
        )

    total = len(predictions)
    counts = {s: sum(1 for p in predictions if p.sentiment == s) for s in SENTIMENTS.values()}
    stats = Stats(
        total=total,
        positive=counts["positive"],
        neutral=counts["neutral"],
        negative=counts["negative"],
        positive_pct=round(counts["positive"] / total * 100, 1),
        neutral_pct=round(counts["neutral"] / total * 100, 1),
        negative_pct=round(counts["negative"] / total * 100, 1),
        average_confidence=round(sum(p.confidence for p in predictions) / total, 4),
    )

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(f"predict_batch : {total} commentaires en {elapsed_ms} ms")
    return PredictResponse(predictions=predictions, stats=stats, processing_time_ms=elapsed_ms)
