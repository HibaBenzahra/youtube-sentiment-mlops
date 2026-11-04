from typing import List
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    comments: List[str] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    text: str
    sentiment: str
    label: int
    confidence: float


class Stats(BaseModel):
    total: int
    positive: int
    neutral: int
    negative: int
    positive_pct: float
    neutral_pct: float
    negative_pct: float
    average_confidence: float


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    stats: Stats
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
