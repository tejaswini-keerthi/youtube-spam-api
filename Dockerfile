FROM continuumio/miniconda3:latest

WORKDIR /app

# Install scikit-learn, numpy, pandas via conda.
# conda ALWAYS uses pre-built binaries — it never compiles from source,
# so the numpy==2.0.0rc1 build-dependency error is impossible here.
RUN conda install -y -c conda-forge \
        python=3.11 \
        scikit-learn=1.3.2 \
        numpy=1.26.4 \
        pandas=2.1.4 \
        requests \
    && conda clean -afy

# Install web framework packages via pip
RUN pip install --no-cache-dir \
        fastapi==0.111.0 \
        "uvicorn[standard]==0.29.0" \
        pydantic==2.7.1 \
        joblib==1.3.2

# Copy source code
COPY . .

# Download dataset and train the model during the build
RUN python train.py

# Start the API ($PORT is provided by Render at runtime)
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
