FROM python:3.11-slim

WORKDIR /app

# Upgrade pip so it can find modern wheels
RUN pip install --upgrade pip

# Install numpy and scikit-learn from pre-built binary wheels ONLY.
# --only-binary prevents pip from ever compiling from source,
# which is what causes the numpy==2.0.0rc1 build-dep error.
RUN pip install --only-binary=numpy,scikit-learn \
    numpy==1.26.4 \
    scikit-learn==1.4.2

# Install remaining dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Download dataset and train the model at build time
RUN python train.py

# Start the API server ($PORT is injected by Render at runtime)
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
