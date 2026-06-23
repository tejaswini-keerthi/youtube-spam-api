"""
app.py — YouTube Spam Classifier microservice (FastAPI).

Endpoints
---------
GET  /               — Service info + endpoint directory
GET  /health         — Health check (confirms model is loaded)
POST /predict        — Classify one comment
POST /predict_batch  — Classify up to 50 comments at once

Start locally:
    uvicorn app:app --reload
"""

from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="YouTube Spam Classifier",
    description=(
        "Classifies YouTube comments as **spam** or **ham** using a "
        "TF-IDF + Logistic Regression model trained on the UCI YouTube "
        "Spam Collection dataset."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model globals ─────────────────────────────────────────────────────────────
MODEL_DIR = Path("model")
tfidf = None
clf   = None


@app.on_event("startup")
def load_model():
    global tfidf, clf
    tfidf_path = MODEL_DIR / "tfidf.pkl"
    clf_path   = MODEL_DIR / "classifier.pkl"

    if not tfidf_path.exists() or not clf_path.exists():
        raise RuntimeError(
            "Model files not found. Run `python train.py` first."
        )

    tfidf = joblib.load(tfidf_path)
    clf   = joblib.load(clf_path)
    print("Model loaded successfully.")


# ── Schemas ──────────────────────────────────────────────────────────────────
class CommentRequest(BaseModel):
    comment: str = Field(
        ...,
        min_length=1,
        description="The YouTube comment text to classify.",
        examples=["Check out my FREE giveaway! Click here now!!!"],
    )


class PredictionResponse(BaseModel):
    comment: str
    prediction: str          # "spam" or "ham"
    confidence: float        # probability of the predicted class
    spam_probability: float
    ham_probability: float


class BatchRequest(BaseModel):
    comments: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of YouTube comment texts to classify (max 50).",
    )


class BatchResponse(BaseModel):
    results: List[PredictionResponse]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _classify(comment: str) -> PredictionResponse:
    vec   = tfidf.transform([comment])
    pred  = int(clf.predict(vec)[0])
    proba = clf.predict_proba(vec)[0]        # [P(ham), P(spam)]
    return PredictionResponse(
        comment=comment,
        prediction="spam" if pred == 1 else "ham",
        confidence=float(max(proba)),
        spam_probability=float(proba[1]),
        ham_probability=float(proba[0]),
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "service": "YouTube Spam Classifier API",
        "version": "1.0.0",
        "model": "TF-IDF (10k features, bigrams) + Logistic Regression",
        "dataset": "UCI YouTube Spam Collection (5 videos, ~1,950 comments)",
        "endpoints": {
            "POST /predict":       "Classify a single YouTube comment",
            "POST /predict_batch": "Classify up to 50 comments at once",
            "GET  /health":        "Health check",
            "GET  /docs":          "Interactive Swagger UI",
        },
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "status": "ok",
        "model_loaded": tfidf is not None and clf is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Classification"])
def predict(request: CommentRequest):
    """
    Classify a **single** YouTube comment as spam or ham.

    Returns the predicted label, confidence score, and individual class
    probabilities.
    """
    return _classify(request.comment.strip())


@app.post("/predict_batch", response_model=BatchResponse, tags=["Classification"])
def predict_batch(request: BatchRequest):
    """
    Classify up to **50** YouTube comments in one call.

    Returns a list of predictions in the same order as the input comments.
    """
    if len(request.comments) > 50:
        raise HTTPException(
            status_code=422,
            detail="Maximum 50 comments per batch request.",
        )
    results = [_classify(c.strip()) for c in request.comments if c.strip()]
    return BatchResponse(results=results)
