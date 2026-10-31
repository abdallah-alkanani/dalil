FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DALIL_DATA_DIR=/data \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# The dalil package and prompts/answering_system_prompt.md, copied in by
# scripts/sync_dalil.sh. Installed without its optional deps: search needs numpy,
# faiss and requests, already installed above.
COPY vendor/ /opt/dalil/
RUN pip install --no-deps /opt/dalil

# DB-IP's country database (CC BY 4.0), for the country of each question.
COPY scripts/download_country_db.py /tmp/
RUN python /tmp/download_country_db.py /opt/geo/country.mmdb && rm /tmp/download_country_db.py
ENV GEOIP_DB=/opt/geo/country.mmdb

COPY dalil_api/ dalil_api/

CMD ["sh", "-c", "uvicorn dalil_api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
