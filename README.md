# 🚫 YouTube Spam Classifier API

A REST microservice that classifies YouTube comments as **spam** or **ham** using a TF-IDF + Logistic Regression model trained on the [UCI YouTube Spam Collection](https://archive.ics.uci.edu/dataset/380/youtube+spam+collection).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Endpoints](#endpoints)
- [Quick Start](#quick-start)
- [Run Locally](#run-locally)
- [Deploy to Render](#deploy-to-render)
- [Project Structure](#project-structure)

---

## Overview

| Item | Detail |
|---|---|
| **Model** | TF-IDF (10k features, bigrams) + Logistic Regression |
| **Dataset** | UCI YouTube Spam Collection — 1,956 comments across 5 videos |
| **Accuracy** | ~97% on held-out test set |
| **Framework** | FastAPI |
| **Hosting** | Render (free tier) |

---

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info and endpoint list |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Classify a single comment |
| `POST` | `/predict_batch` | Classify up to 50 comments |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Quick Start

### Classify a single comment

```bash
curl -X POST https://youtube-spam-api.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"comment": "Check out my FREE giveaway, click here now!!!"}'
```

**Response:**
```json
{
  "comment": "Check out my FREE giveaway, click here now!!!",
  "prediction": "spam",
  "confidence": 0.9821,
  "spam_probability": 0.9821,
  "ham_probability": 0.0179
}
```

### Classify multiple comments at once

```bash
curl -X POST https://youtube-spam-api.onrender.com/predict_batch \
     -H "Content-Type: application/json" \
     -d '{
       "comments": [
         "Subscribe to my channel for daily giveaways!!!",
         "This song brings back so many memories, love it."
       ]
     }'
```

**Response:**
```json
{
  "results": [
    {
      "comment": "Subscribe to my channel for daily giveaways!!!",
      "prediction": "spam",
      "confidence": 0.9512,
      "spam_probability": 0.9512,
      "ham_probability": 0.0488
    },
    {
      "comment": "This song brings back so many memories, love it.",
      "prediction": "ham",
      "confidence": 0.9271,
      "spam_probability": 0.0729,
      "ham_probability": 0.9271
    }
  ]
}
```

### Browser-based testing (no curl needed)

Visit **`https://youtube-spam-api.onrender.com/docs`** — the interactive Swagger UI lets you test every endpoint directly in your browser.

---

## Run Locally

**Prerequisites:** Python 3.9+

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/youtube-spam-api.git
cd youtube-spam-api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and save the model (downloads dataset automatically)
python train.py

# 4. Start the API server
uvicorn app:app --reload
```

The API will be running at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the Swagger UI.

---

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — no manual config needed
5. Click **Deploy**

The `render.yaml` sets:
- **Build command:** `pip install -r requirements.txt && python train.py`
- **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`

> **Note:** On the free tier, the service sleeps after 15 minutes of inactivity. The first request after sleep may take ~30 seconds to respond. Subsequent requests are fast (<200ms).

---

## Project Structure

```
youtube-spam-api/
├── app.py                 # FastAPI microservice
├── train.py               # Downloads data and trains the model
├── requirements.txt       # Python dependencies
├── render.yaml            # Render deployment config
├── .gitignore
└── model/                 # Created by train.py (git-ignored)
    ├── tfidf.pkl
    └── classifier.pkl
```
