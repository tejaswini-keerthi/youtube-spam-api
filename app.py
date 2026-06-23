from pathlib import Path
from typing import List
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="YouTube Spam Classifier", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_DIR = Path("model")
tfidf = None
clf   = None

@app.on_event("startup")
def load_model():
    global tfidf, clf
    tfidf = joblib.load(MODEL_DIR / "tfidf.pkl")
    clf   = joblib.load(MODEL_DIR / "classifier.pkl")
    print("Model loaded.")

class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1,
        examples=["Check out my FREE giveaway! Click here now!!!"])

class PredictionResponse(BaseModel):
    comment: str
    prediction: str
    confidence: float
    spam_probability: float
    ham_probability: float

class BatchRequest(BaseModel):
    comments: List[str]

class BatchResponse(BaseModel):
    results: List[PredictionResponse]

def _classify(comment: str) -> PredictionResponse:
    vec   = tfidf.transform([comment])
    pred  = int(clf.predict(vec)[0])
    proba = clf.predict_proba(vec)[0]
    return PredictionResponse(
        comment=comment,
        prediction="spam" if pred == 1 else "ham",
        confidence=float(max(proba)),
        spam_probability=float(proba[1]),
        ham_probability=float(proba[0]),
    )

@app.get("/")
def root():
    return {"service": "YouTube Spam Classifier API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": tfidf is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: CommentRequest):
    return _classify(req.comment.strip())

@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if len(req.comments) > 50:
        raise HTTPException(422, "Maximum 50 comments per batch.")
    return BatchResponse(results=[_classify(c.strip()) for c in req.comments if c.strip()])
