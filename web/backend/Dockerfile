FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DALIL_DATA_DIR=/data \
    DALIL_DENSE_COMPACT_DIMS=1024 \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# The dalil package, copied in by scripts/sync_dalil.sh. Installed without its
# optional deps: search needs numpy, faiss and requests, already installed above.
COPY vendor/ /opt/dalil/
RUN pip install --no-deps /opt/dalil

COPY daleel_api/ daleel_api/

CMD ["sh", "-c", "uvicorn daleel_api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
