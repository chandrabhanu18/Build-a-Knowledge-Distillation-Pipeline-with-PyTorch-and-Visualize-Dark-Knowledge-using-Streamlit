FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip
# Try to install CPU PyTorch wheels first (best-effort)
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision || true
RUN python -m pip install -r /tmp/requirements.txt

COPY . /workspace

RUN mkdir -p /workspace/output /workspace/data

EXPOSE 8501
