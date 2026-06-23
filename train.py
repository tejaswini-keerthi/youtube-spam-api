"""
train.py — Downloads the YouTube Spam Collection and trains a TF-IDF +
Logistic Regression classifier. Run once before starting the API server.

Usage:
    python train.py
"""

import io
import zipfile
import requests
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

# UCI YouTube Spam Collection (5 CSV files inside zip)
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip"
)


def download_data() -> pd.DataFrame:
    print("Downloading YouTube Spam Collection from UCI...")
    r = requests.get(DATASET_URL, timeout=120)
    r.raise_for_status()

    dfs = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
        print(f"  Found {len(csv_files)} CSV file(s): {csv_files}")
        for fname in csv_files:
            with z.open(fname) as f:
                try:
                    tmp = pd.read_csv(f, encoding="latin-1")
                    # Normalise column names (some files use different casing)
                    tmp.columns = [c.upper() for c in tmp.columns]
                    if "CONTENT" in tmp.columns and "CLASS" in tmp.columns:
                        dfs.append(tmp[["CONTENT", "CLASS"]])
                except Exception as e:
                    print(f"  Warning — could not read {fname}: {e}")

    if not dfs:
        raise ValueError("No valid CSV files found in the downloaded zip.")

    data = pd.concat(dfs, ignore_index=True)
    data.columns = ["text", "label"]
    data.dropna(subset=["text", "label"], inplace=True)
    data["label"] = data["label"].astype(int)

    print(
        f"  Loaded {len(data):,} comments  |  "
        f"spam: {data['label'].sum()} ({data['label'].mean()*100:.1f}%)  |  "
        f"ham: {(data['label']==0).sum()}"
    )
    return data


def train_and_save() -> float:
    data = download_data()

    # ── TF-IDF vectoriser ────────────────────────────────────────────────────
    tfidf = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),      # unigrams + bigrams
        sublinear_tf=True,       # log-normalise term frequency
        strip_accents="unicode",
        analyzer="word",
        min_df=2,
    )

    X = tfidf.fit_transform(data["text"])
    y = data["label"].values

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Logistic Regression ──────────────────────────────────────────────────
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # ── Evaluation ───────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # ── Persist ──────────────────────────────────────────────────────────────
    joblib.dump(tfidf, MODEL_DIR / "tfidf.pkl")
    joblib.dump(clf,   MODEL_DIR / "classifier.pkl")
    print(f"Model saved to {MODEL_DIR}/  (tfidf.pkl + classifier.pkl)")
    return acc


if __name__ == "__main__":
    train_and_save()
