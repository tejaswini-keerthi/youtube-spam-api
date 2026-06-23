# Module 5 Assignment — YouTube Spam Classifier Microservice

## Overview

This microservice exposes the **TF-IDF + Logistic Regression** spam
classifier built in Module 2. The model was trained on the UCI YouTube
Spam Collection — 1,956 comments drawn from five popular YouTube videos —
and achieves approximately **97% accuracy** on the held-out test set. The
service is built with **FastAPI** and deployed on **Render**.

---

## 1. Service Input and Output

### General Input

The primary endpoint accepts an HTTP **POST** request to `/predict` with
a JSON body containing a single field:

| Field     | Type   | Required | Description                           |
|-----------|--------|----------|---------------------------------------|
| `comment` | string | Yes      | The YouTube comment text to classify. |

The `/predict_batch` endpoint accepts the same kind of request but with a
`comments` field that is a **list of strings** (maximum 50 per call).

### General Output

The service returns a JSON object with the following fields:

| Field               | Type   | Description                                           |
|---------------------|--------|-------------------------------------------------------|
| `comment`           | string | The original comment text (echoed back).              |
| `prediction`        | string | `"spam"` or `"ham"`.                                  |
| `confidence`        | float  | Probability of the predicted class (0.0 – 1.0).       |
| `spam_probability`  | float  | Model's estimated probability that the comment is spam.|
| `ham_probability`   | float  | Model's estimated probability that the comment is ham. |

---

## 2. Specific Input / Output Examples

### Example 1 — Spam comment

**Request**

```bash
curl -X POST https://youtube-spam-api.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"comment": "CHECK OUT MY CHANNEL!!! FREE iPhone giveaway click here http://bit.ly/win-now"}'
```

**Response**

```json
{
  "comment": "CHECK OUT MY CHANNEL!!! FREE iPhone giveaway click here http://bit.ly/win-now",
  "prediction": "spam",
  "confidence": 0.9821,
  "spam_probability": 0.9821,
  "ham_probability": 0.0179
}
```

---

### Example 2 — Ham (legitimate) comment

**Request**

```bash
curl -X POST https://youtube-spam-api.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"comment": "This song takes me back to summer 2012. Absolute classic, I never get tired of it."}'
```

**Response**

```json
{
  "comment": "This song takes me back to summer 2012. Absolute classic, I never get tired of it.",
  "prediction": "ham",
  "confidence": 0.9634,
  "spam_probability": 0.0366,
  "ham_probability": 0.9634
}
```

---

### Example 3 — Batch prediction (two comments at once)

**Request**

```bash
curl -X POST https://youtube-spam-api.onrender.com/predict_batch \
     -H "Content-Type: application/json" \
     -d '{
       "comments": [
         "Subscribe to my channel for daily uploads and giveaways!!!",
         "Great video, really enjoyed the breakdown of the topic."
       ]
     }'
```

**Response**

```json
{
  "results": [
    {
      "comment": "Subscribe to my channel for daily uploads and giveaways!!!",
      "prediction": "spam",
      "confidence": 0.9512,
      "spam_probability": 0.9512,
      "ham_probability": 0.0488
    },
    {
      "comment": "Great video, really enjoyed the breakdown of the topic.",
      "prediction": "ham",
      "confidence": 0.9271,
      "spam_probability": 0.0729,
      "ham_probability": 0.9271
    }
  ]
}
```

---

### Example 4 — Health check (no body required)

**Request**

```bash
curl https://youtube-spam-api.onrender.com/health
```

**Response**

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## 3. Service URL

**Base URL:** `https://youtube-spam-api.onrender.com`

| Endpoint           | Method | Purpose                          |
|--------------------|--------|----------------------------------|
| `/`                | GET    | Service info and endpoint list   |
| `/health`          | GET    | Confirms the model is loaded     |
| `/predict`         | POST   | Classify a single comment        |
| `/predict_batch`   | POST   | Classify up to 50 comments       |
| `/docs`            | GET    | Interactive Swagger UI (browser) |

### Quick test (browser-friendly Swagger UI)

Navigate to `https://youtube-spam-api.onrender.com/docs` in any browser.
Click **POST /predict → Try it out**, paste any YouTube comment into the
`comment` field, and click **Execute**.

> **Note:** The service is hosted on Render's free tier. If it has been
> idle, the first request may take up to 30 seconds to wake the instance.
> Subsequent requests respond in under 200 ms.

---

## 4. Technical Notes

| Item | Detail |
|------|--------|
| Framework | FastAPI 0.111 |
| Model | TF-IDF (10 k features, unigrams + bigrams) + Logistic Regression |
| Training data | UCI YouTube Spam Collection — 1,956 comments |
| Test accuracy | ~97% |
| Hosting | Render (free web service) |
| Language | Python 3.11 |
