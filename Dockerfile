# EVAde model service (beta). Models are NOT baked in: mount a folder built by evade.training at /models.
#   docker build -t evade-api .
#   docker run -p 8000:8000 -e EVADE_API_TOKEN=... -e EVADE_CORS_ORIGINS=https://<frontend> \
#              -e EVADE_ALLOW_DRAFT=1 -v /path/to/evade_train_beta:/models:ro evade-api
# Cloudflare storage: add -e EVADE_STORAGE=cloudflare and the CF_* variables (see evade/api/README.md), passed as
# runtime secrets (--env-file outside the repo), never baked into the image.
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
CMD ["uvicorn", "evade.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
