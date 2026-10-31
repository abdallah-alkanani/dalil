# Build from the repository root, which holds the dalil package and the answering prompt:
#     docker build -f web/backend/Dockerfile .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DALIL_DATA_DIR=/data \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app
COPY web/backend/requirements.txt .
RUN pip install -r requirements.txt

# The dalil package, installed without its optional deps: search needs numpy, faiss
# and requests, already installed above. The answering prompt goes beside it.
COPY pyproject.toml README.md LICENSE /opt/dalil/
COPY dalil/ /opt/dalil/dalil/
RUN pip install --no-deps /opt/dalil
COPY prompts/answering_system_prompt.md /opt/dalil/prompts/

# DB-IP's country database (CC BY 4.0), for the country of each question.
COPY web/backend/scripts/download_country_db.py /tmp/
RUN python /tmp/download_country_db.py /opt/geo/country.mmdb && rm /tmp/download_country_db.py
ENV GEOIP_DB=/opt/geo/country.mmdb

COPY web/backend/dalil_api/ dalil_api/

CMD ["sh", "-c", "uvicorn dalil_api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
