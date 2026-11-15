# Tutorial: run the EVAde API on Google Cloud (Cloud Run)

This puts the beta FastAPI service (`evade/api`) on a public HTTPS URL that the frontend and the demo can call. The first time takes about 45 minutes. Every command runs in **Cloud Shell** (the terminal button at the top right of the Google Cloud console). Cloud Shell already has `gcloud`, `docker` and `git`, so you install nothing on your PC.

> The API returns flags for human review, never diagnoses. The demo uses public research data and pseudonymous user IDs only, so don't send real health data to a demo deployment.

## What you end up with

```
frontend (server side) ──HTTPS──▶ Cloud Run service "evade-api" (1 instance, 1 GiB, port 8000)
                                    ├── EVADE_API_TOKEN, from Secret Manager
                                    └── /models (read-only), from a private Cloud Storage bucket
```

| Piece | Why |
|---|---|
| Artifact Registry | stores the container image built from the repo's `Dockerfile` |
| Cloud Storage bucket | holds the trained model folders (`seizure/`, `ecg/`, ...). Models are never baked into the image. |
| Secret Manager | holds the API token (and the Cloudflare keys later), so no secret ends up in the image, the repo or your shell history |
| Cloud Run | runs the container behind HTTPS |

## 0. Before you start

1. **Billing and a budget alert.** You need a Google account with billing enabled (new accounts usually get free credit). Go to Billing → Budgets & alerts → Create budget, for example $10 with email alerts at 50/90/100%. A budget only *notifies*; it doesn't stop charges. To guarantee no charges, delete the project (see Teardown).
2. **The model folders.** The beta models live on the training server in its `evade_train_beta/` folder. Each of `seizure/` and `ecg/` holds `model.joblib`, `feature_names.json` and `metrics.json`; newer runs also write `manifest.json`. On the training server, pack them and record a checksum:
   ```bash
   tar czf evade_models.tgz -C <path to evade_train_beta> seizure ecg
   sha256sum evade_models.tgz        # send this hash to whoever deploys, separately from the file
   ```
   Then download `evade_models.tgz` to your PC.
3. **Library versions.** A `joblib` model only loads reliably with the same scikit-learn, numpy, scipy and joblib versions it was trained with. On the training machine, run:
   ```bash
   <training venv>/bin/python -c "import sys,sklearn,numpy,scipy,joblib; print(sys.version.split()[0], sklearn.__version__, numpy.__version__, scipy.__version__, joblib.__version__)"
   ```
   Pin those four versions in `requirements.txt` (`scikit-learn==X`, `numpy==Y`, ...) and **commit the change to the branch**, so every future build uses them. If the Python minor version isn't 3.12, change `FROM python:3.12-slim` in the `Dockerfile` to match.

## 1. Settings, saved once

Cloud Shell sessions time out and forget variables. Write the settings to a file that every new shell loads. Replace the values on the right-hand side. Placeholders in this guide never use `<angle brackets>` inside commands, because bash treats those as redirections.

```bash
cat > ~/evade.env <<'EOF'
export PROJECT=evade-demo-xyz123                 # globally unique project id: lowercase, digits, dashes
export REGION=us-east1                           # any Cloud Run region near you
export FRONTEND_ORIGIN=https://your-frontend.example   # exact origin, no trailing slash; leave empty if no frontend yet
export SA=evade-api@${PROJECT}.iam.gserviceaccount.com
export BUCKET=${PROJECT}-evade-models
EOF
echo 'source ~/evade.env' >> ~/.bashrc
source ~/evade.env
echo "$PROJECT $REGION $SA $BUCKET"              # check: four non-empty values
```

If any later command fails with an empty value, run `source ~/evade.env` and try again.

## 2. Project and services

```bash
gcloud projects create $PROJECT --name="EVAde demo"      # skip if you reuse an existing project
gcloud config set project $PROJECT
gcloud billing accounts list                             # copy the ACCOUNT_ID
gcloud billing projects link $PROJECT --billing-account=PASTE_ACCOUNT_ID

gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
    cloudbuild.googleapis.com secretmanager.googleapis.com storage.googleapis.com
```

## 3. A service account for the API (least privilege)

```bash
gcloud iam service-accounts create evade-api --display-name="EVAde API runtime"
```

Later steps give this account read access to the model bucket and the token, and nothing else.

## 4. The API token

```bash
# generates a random token and stores it; it is never printed or written to a file
python3 -c "import secrets; print(secrets.token_urlsafe(32), end='')" | \
    gcloud secrets create evade-api-token --replication-policy=automatic --data-file=-

gcloud secrets add-iam-policy-binding evade-api-token \
    --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor
```

**Who may hold the token.** The token protects every route, so it must not reach a browser. If the frontend is a browser app, put the token in a small server-side piece that adds the `Authorization` header: the frontend's own server route, or a Cloudflare Worker. Anything in browser JavaScript is visible to every visitor, and anyone holding the token can create users until the single in-memory instance runs out of memory. If there is no server side for the hackathon, treat the demo token as public, and rotate it right after the demo (step 9).

To read the token for a server-side config or a test, run `gcloud secrets versions access latest --secret=evade-api-token`. Share it through a password manager, never chat.

## 5. Upload the models to a private bucket

```bash
gcloud storage buckets create gs://$BUCKET --location=$REGION \
    --uniform-bucket-level-access --public-access-prevention
gcloud storage buckets update gs://$BUCKET --versioning      # keeps old model versions for rollback
```

Upload `evade_models.tgz` with Cloud Shell's ⋮ menu → **Upload**. It lands in your home folder. If the browser renamed it, for example to `evade_models (1).tgz`, rename it back. Then:

```bash
ls -l ~/*.tgz
sha256sum ~/evade_models.tgz      # must equal the hash from the training server; stop if it differs
tar tzf ~/evade_models.tgz        # expect seizure/... and ecg/... with no extra leading folder
rm -rf ~/models && mkdir ~/models && tar xzf ~/evade_models.tgz -C ~/models
gcloud storage cp --recursive ~/models/* gs://$BUCKET/
gcloud storage ls --recursive gs://$BUCKET/

gcloud storage buckets add-iam-policy-binding gs://$BUCKET \
    --member="serviceAccount:$SA" --role=roles/storage.objectViewer
```

**Why the checks matter.** A `joblib` file runs code when it loads, so the bucket must only ever hold models built by `evade.training` on a machine you control. Newer folders also carry a `manifest.json` with sha256 checksums that the API verifies before loading. Never give anyone else write access to this bucket.

## 6. Build the container image

```bash
cd ~
git clone --branch integrate-blindsight-data-nasa https://github.com/Mikeblackbelt/EVAde-backend-models.git
cd ~/EVAde-backend-models                 # the image tag below needs to run inside the repo
grep -E "^(scikit-learn|numpy|scipy|joblib)" requirements.txt    # expect pinned versions (step 0)

gcloud artifacts repositories create evade --repository-format=docker --location=$REGION
export IMAGE=$REGION-docker.pkg.dev/$PROJECT/evade/evade-api:$(git rev-parse --short HEAD)
echo "export IMAGE=$IMAGE" >> ~/evade.env
gcloud builds submit --tag $IMAGE .
```

The build takes a few minutes. **If it fails with a permission error:** projects created since mid-2024 build as the Compute Engine default service account, which no longer gets broad rights automatically. Check which account builds, then grant it the Cloud Build builder role:

```bash
gcloud builds get-default-service-account
PROJECT_NUMBER=$(gcloud projects describe $PROJECT --format='value(projectNumber)')
gcloud projects add-iam-policy-binding $PROJECT \
    --member=serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com \
    --role=roles/cloudbuild.builds.builder
```

Don't grant `roles/storage.admin` or Editor to fix build errors. Either would let the build account overwrite the models the API loads.

## 7. Deploy to Cloud Run

```bash
CORS_FLAG=""
[ -n "$FRONTEND_ORIGIN" ] && CORS_FLAG="--update-env-vars=^@^EVADE_CORS_ORIGINS=$FRONTEND_ORIGIN"

gcloud run deploy evade-api \
    --image=$IMAGE --region=$REGION \
    --service-account=$SA \
    --port=8000 --memory=1Gi --cpu=1 \
    --min=1 --max=1 \
    --execution-environment=gen2 \
    --set-secrets=EVADE_API_TOKEN=evade-api-token:latest \
    --set-env-vars=EVADE_ALLOW_DRAFT=1,EVADE_MODEL_DIR=/models \
    $CORS_FLAG \
    --add-volume=name=models,type=cloud-storage,bucket=$BUCKET,readonly=true \
    --add-volume-mount=volume=models,mount-path=/models \
    --allow-unauthenticated
```

| Flag | Reason |
|---|---|
| `--port=8000` | the image listens on 8000 (Cloud Run's default is 8080) |
| `--memory=1Gi --cpu=1` | the beta needs about 300-450 MB. ONNX models (the plan for the deep models) fit in 1 GiB. Use 2Gi if a PyTorch model is ever served directly. |
| `--min=1 --max=1` | service-level instance limits (older gcloud versions call them `--min-instances` / `--max-instances`). **Run exactly one instance.** The beta keeps users and baselines in memory. A second instance would have its own users, and scaling to zero would wipe them. Keep `--max=1` even after Cloudflare storage is added: the API's deletion gate and device checks are per process (see `evade/api/README.md`). Cloud Run can briefly exceed the maximum during traffic spikes, which is acceptable for a demo. |
| `--execution-environment=gen2` | needed for the Cloud Storage volume mount |
| `--set-secrets` | injects the token from Secret Manager as an environment variable |
| `^@^` before `EVADE_CORS_ORIGINS` | switches gcloud's list delimiter to `@`, so a comma-separated list of origins survives (`https://a.example,https://b.example`). With no origin set, the API adds no CORS headers. Add one later with `gcloud run services update evade-api --region=$REGION --update-env-vars=^@^EVADE_CORS_ORIGINS=...`. |
| `--allow-unauthenticated` | the API checks its own bearer token on every route except `/v1/health`, `/docs` and `/openapi.json`, which are public and show the API's shape and loaded models. If your organization blocks public services, drop this flag. Server-side callers then send a Google identity token in `X-Serverless-Authorization: Bearer ...` and keep the EVAde token in `Authorization: Bearer ...`. Browsers can't easily do that. |

**Restarts lose in-memory state.** Every deploy, settings change or restart starts a new container, which loses users and baselines until Cloudflare storage is on. Deploy before the demo, not during it.

## 8. Check that it works

```bash
export URL=$(gcloud run services describe evade-api --region=$REGION --format='value(status.url)')
echo "export URL=$URL" >> ~/evade.env
curl -s $URL/v1/health                   # no token needed; lists the loaded models (expect seizure and ecg)
TOKEN=$(gcloud secrets versions access latest --secret=evade-api-token)
curl -s -H "Authorization: Bearer $TOKEN" $URL/v1/models
curl -s -X POST $URL/v1/users -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" -d '{"consent": true, "cohort": "demo"}'
unset TOKEN
```

Interactive docs are at `$URL/docs`.

**Replay demo.** To replay a real EEG recording through the deployed API, run it from a clone of the repo with `pip install -r requirements.txt` and the EDF file on disk. The script reads the token from `EVADE_API_TOKEN`; its docstring lists all options.

```bash
EVADE_API_TOKEN="$(gcloud secrets versions access latest --secret=evade-api-token)" \
  python scripts/demo_replay.py --api $URL --edf path/to/recording.edf --seizure ONSET_S OFFSET_S
```

Logs are in the console under Cloud Run → evade-api → Logs, or from the shell:

```bash
gcloud run services logs read evade-api --region=$REGION --limit=50
```

## 9. Updating

**New code.** In `~/EVAde-backend-models`, `git pull`, then rebuild with a new tag (re-run the `export IMAGE=...` and `gcloud builds submit` lines from step 6), then:

```bash
gcloud run deploy evade-api --image=$IMAGE --region=$REGION
```

Other settings carry over.

**New models.**
1. Check the new folder first; its metrics and manifest should look right.
2. Archive the live one: `gcloud storage cp --recursive gs://$BUCKET/seizure gs://$BUCKET/archive/$(date +%F)/`.
3. Copy the new folder over it, then redeploy to load it: `gcloud run services update evade-api --region=$REGION`.
4. To roll back, copy the archive back and redeploy.

Cloud Run can restart the instance at any time, so never leave a half-uploaded model in place.

**Rotate the token** without printing it:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32), end='')" | \
    gcloud secrets versions add evade-api-token --data-file=-
gcloud run services update evade-api --region=$REGION        # picks up "latest"
gcloud secrets versions list evade-api-token                 # then disable the old one:
gcloud secrets versions disable OLD_VERSION_NUMBER --secret=evade-api-token
```

## 10. Later: Cloudflare storage

Do this once the Cloudflare backend is merged (branch `api-models-storage`):

1. **Add boto3.** Add `boto3` to `requirements.txt`, because R2 needs it. Commit, then rebuild (step 6).
2. **Apply the database schema.** Run this from a clone, with the three D1 variables set in that shell only:
   ```bash
   python -m evade.storage.migrate
   ```
   Or use `npx wrangler d1 execute DB_NAME --remote --file evade/storage/schema.sql`.
3. **Store each secret without typing it into the command line.** Each value is read silently, so it never lands in your shell history:
   ```bash
   for NAME in cf-api-token cf-r2-key-id cf-r2-secret; do
     read -rsp "value for $NAME: " V; echo
     printf %s "$V" | gcloud secrets create $NAME --replication-policy=automatic --data-file=-
     unset V
     gcloud secrets add-iam-policy-binding $NAME --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor
   done
   ```
4. **Deploy with the storage settings:**
   ```bash
   gcloud run services update evade-api --region=$REGION \
     --update-secrets=CF_API_TOKEN=cf-api-token:latest,CF_R2_ACCESS_KEY_ID=cf-r2-key-id:latest,CF_R2_SECRET_ACCESS_KEY=cf-r2-secret:latest \
     --update-env-vars=EVADE_STORAGE=cloudflare,CF_ACCOUNT_ID=PASTE_ID,CF_D1_DATABASE_ID=PASTE_ID,CF_R2_BUCKET=PASTE_BUCKET
   ```
   For an EU-jurisdiction R2 bucket, also set `CF_R2_ENDPOINT=https://ACCOUNT_ID.eu.r2.cloudflarestorage.com`.

With state in Cloudflare, `--min=0` becomes acceptable: a cold start of a few seconds reloads users from D1 on first use. **Keep `--max=1`.**

## Gotchas

| Symptom | Cause and fix |
|---|---|
| `/v1/health` lists no models | Folders must sit at the bucket root (`gs://$BUCKET/seizure/model.joblib`). The service account may lack `objectViewer`. `EVADE_ALLOW_DRAFT` may not be `1`. |
| scikit-learn or numpy version warnings or errors at load | The image's library versions differ from training. Pin all four (step 0), commit and rebuild. |
| `411 length_required` | The API requires a `Content-Length` header. Normal JSON clients (fetch, httpx, `curl -d`) send one; chunked or streaming uploads don't. Don't enable HTTP/2 end-to-end (`--use-http2`). |
| `413 payload_too_large` | Bodies are capped at 1 MB, about 30 s of 18-channel EEG. Send long recordings in several calls. |
| users disappear | The instance restarted or was redeployed (in-memory beta), or more than one instance ran. Check `--max=1`. |
| deploy rejects the volume, `--min` or `--max` flags | Update gcloud (`gcloud components update`; Cloud Shell is usually current). Check that `--execution-environment=gen2` is set. |
| browser CORS errors | `EVADE_CORS_ORIGINS` must match the frontend origin exactly: scheme, host and port, no trailing slash. |
| a variable is empty after a break | `source ~/evade.env` |

## Costs and teardown

The main cost is one always-on instance with 1 vCPU and 1 GiB; idle minimum instances are billed at a reduced rate. Storage, secrets and builds cost little at this size. Check the Cloud Run pricing page for current rates.

After the hackathon:

```bash
source ~/evade.env
gcloud run services delete evade-api --region=$REGION
gcloud storage rm --recursive gs://$BUCKET
gcloud storage rm --recursive gs://${PROJECT}_cloudbuild        # build source uploads
gcloud artifacts repositories delete evade --location=$REGION
gcloud secrets delete evade-api-token        # and cf-api-token, cf-r2-key-id, cf-r2-secret if created
gcloud compute instances delete evade-api-vm --zone=${REGION}-b   # only if you made the VM below
gcloud iam service-accounts delete $SA
```

Deleting the whole project (`gcloud projects delete $PROJECT`) is the sure way to stop every charge.

## Alternative: a small VM instead of Cloud Run

A plain server is useful, for example, to keep in-memory state across code updates or to run the replay demo next to the API. A Compute Engine `e2-small` (2 GB) running Docker works:

```bash
gcloud services enable compute.googleapis.com
gcloud compute instances create evade-api-vm --zone=${REGION}-b --machine-type=e2-small \
    --image-family=debian-12 --image-project=debian-cloud \
    --service-account=$SA --scopes=cloud-platform
gcloud compute ssh evade-api-vm --zone=${REGION}-b
```

On the VM, install Docker, copy the models from the bucket, and run the container as in `evade/api/README.md`. Put it behind HTTPS with a Cloudflare Tunnel, which needs no open inbound ports.

The VM runs as the same least-privilege account, which can read the models and the token but write nothing. Never expose port 8000 directly over plain HTTP, because the bearer token would travel unencrypted.
