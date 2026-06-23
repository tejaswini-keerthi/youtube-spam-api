FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install "Cython<3.0.0" "numpy==1.26.4"

RUN pip install --no-build-isolation "scikit-learn==1.3.2"

RUN pip install \
    "pandas==2.1.4" \
    "joblib==1.3.2" \
    "requests==2.31.0" \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "pydantic==2.7.1"

COPY . .
RUN python train.py

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
