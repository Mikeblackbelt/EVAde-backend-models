# EVAde model service (beta). Models are NOT baked in: mount a folder built by evade.training at /models.
#   docker build -t evade-api .
#   docker run -p 8000:8000 --env-file <file outside the repo with EVADE_API_KEY=...> \
#              -e EVADE_ALLOW_DRAFT=1 -v /path/to/evade_train_beta:/models:ro evade-api
# EVADE_API_KEY (32+ characters) is required: the service refuses to start without it. On Cloud Run it comes from
# Secret Manager (--set-secrets), never from the image or the repo (see evade/api/README.md).
# Cloudflare storage: add -e EVADE_STORAGE=cloudflare and the CF_* variables, also as runtime secrets.
# --no-access-log: uvicorn's access log has URL paths, which contain user IDs.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 EVADE_MODEL_DIR=/models
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# optional runtimes, e.g. --build-arg EXTRA_PIP="boto3" for Cloudflare R2, or "boto3 onnxruntime"
ARG EXTRA_PIP=""
RUN if [ -n "$EXTRA_PIP" ]; then pip install --no-cache-dir $EXTRA_PIP; fi
COPY evade ./evade
RUN useradd --create-home --uid 10001 evade
USER evade
EXPOSE 8000
# Forwarded headers are trusted only from FORWARDED_ALLOW_IPS (uvicorn reads it; default 127.0.0.1, e.g. a local
# Cloudflare Tunnel). On Cloud Run set FORWARDED_ALLOW_IPS='*': only Google's front end can reach the container and it
# overwrites X-Forwarded-Proto. Never expose port 8000 directly.
ENV FORWARDED_ALLOW_IPS=127.0.0.1
CMD ["uvicorn", "evade.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-access-log"]
