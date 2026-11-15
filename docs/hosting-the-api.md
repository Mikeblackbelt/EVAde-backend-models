# Hosting the EVAde model API (plan for the team)

Michael will send the API as a **.zip** (tomorrow). This page says what's in it, where to host it, and how to connect it to the team's backend. It's a proof of concept, so the aim is one working, private-by-default instance for the demo.

> The API returns flags for human review, never diagnoses. Use demo or synthetic data only.

## 1. What you'll receive

| Item | How it arrives | Notes |
|---|---|---|
| `evade-api-<commit>.zip` | file transfer (Drive, Discord, etc.) | code + models, no secrets |
| SHA-256 of the zip | a separate message | check it before unzipping (step 3) |
| The API key | **password manager only**, never chat or email | or you generate your own (step 4); then send it to Michael the same way |

Inside the zip:

```
evade-api-<commit>/
  evade/                 the API package (FastAPI), preprocessing, risk engine, model loaders
  requirements.txt       pinned versions (they must match the ones the models were trained with)
  Dockerfile             python:3.12-slim, non-root, uvicorn on port 8000, no access log
  models/                trained model folders, each with manifest.json + weights + checksums
    ecg/  seizure/       beta models (joblib)
    ecg_deep/            deep 12-lead ECG model (ONNX, 2 MB, macro AUROC 0.927 on PTB-XL)
    spaceweather/        solar proton storm forecast (if included)
  .env.example           variable NAMES only
  COMMIT                 the exact code version
  HOSTING.md             this page
```

`sleep_deep` and `seizure_deep` have no folders yet. Their endpoints return clearly labelled placeholders (`"source": "mock"`) until the trained models are dropped in.

## 2. Pick where to host it (one of these)

| Option | When | Cost (rough) | Effort |
|---|---|---|---|
| **A. Google Cloud Run** (recommended if anyone has a Google Cloud account) | HTTPS, Secret Manager and logs built in | low; one always-on 1 GiB instance plus a little storage | about 45 min, following [google-cloud-setup.md](google-cloud-setup.md) |
| **B. Small VM (Vultr, in the team stack) behind a Cloudflare Tunnel** | if the team already pays for Vultr, or there's no Google Cloud account | about $10-12 a month for a 2 GB VM | about 45 min (section 6) |
| C. Cloudflare Containers | later, once API state lives in D1/R2 | Workers Paid plus usage | see [cloudflare-containers-setup.md](cloudflare-containers-setup.md); not for tomorrow |

Rules that apply to every option:
- **Exactly one instance**, with 1 GiB of RAM or more. The API keeps users and baselines in memory, so a second instance would split them and a restart clears them. **Deploy before the demo, not during it.**
- **HTTPS only.** Cloud Run and the Cloudflare Tunnel both provide it. Never expose port 8000 over plain HTTP.
- **Only the team's backend Worker calls the API.** No browser calls it, and there's no CORS. The Worker adds the `X-EVAde-Key` header from a Worker secret.
- **The key lives in a secret store,** never in code, the zip, git or chat:
  - option A: Google Secret Manager;
  - option B: a root-only env file on the VM;
  - the Worker: `wrangler secret`.
- **Spending:**
  - set a budget alert;
  - keep max instances at 1 (option A: `--max=1`; option B: one VM).

## 3. Check and unpack

```bash
sha256sum evade-api-<commit>.zip            # Windows: certutil -hashfile evade-api-<commit>.zip SHA256
# compare with the hash Michael sent; stop if it differs
unzip evade-api-<commit>.zip && cd evade-api-<commit>
```

Why this matters: the beta models are `joblib` files, which can run code when loaded. Only load model folders that came from Michael with a matching hash.

## 4. Run it locally first (10 min, any laptop with Docker)

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"     # a test key (at least 32 characters)
docker build -t evade-api .
docker run --rm -p 127.0.0.1:8000:8000 -e EVADE_API_KEY=<test key> -e EVADE_ALLOW_DRAFT=1 \
           -v "$PWD/models:/models:ro" evade-api
```

In another terminal:

```bash
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/v1/health                 # expect 401 (no key)
curl -s localhost:8000/v1/health -H "X-EVAde-Key: <test key>"                    # expect 200 + loaded models
curl -s localhost:8000/v1/models -H "X-EVAde-Key: <test key>"                    # ecg_deep "available"; sleep/seizure "mock"
curl -s -X POST localhost:8000/v1/users -H "X-EVAde-Key: <test key>" \
     -H "Content-Type: application/json" -d '{"consent": true, "cohort": "demo"}'
```

If the zip predates the key change, it uses `Authorization: Bearer <token>` with `EVADE_API_TOKEN` instead. Michael will say which version it is.

## 5. Option A: Google Cloud Run

Follow [google-cloud-setup.md](google-cloud-setup.md) with these zip-specific changes:
- **Image tag:** the zip has no git history, so in step 6 use `export IMAGE=$REGION-docker.pkg.dev/$PROJECT/evade/evade-api:$(cat COMMIT | cut -c1-7)` instead of `git rev-parse`. Build from the unzipped folder with `gcloud builds submit --tag $IMAGE .`.
- **Models:** in step 5, upload the zip's `models/*` folders to the bucket (`gcloud storage cp --recursive models/* gs://$BUCKET/`).
- **Key:** in step 4, store the key as secret `evade-api-key`, and in step 7 deploy with `--set-secrets=EVADE_API_KEY=evade-api-key:latest`. Don't set any CORS variable.
- **Instances and budget:** keep `--min=1 --max=1` and the budget alert.

Then share two things with the Worker owner: the service URL, through any channel, and the key, through the password manager.

## 6. Option B: a small VM behind a Cloudflare Tunnel

1. **Create the VM.** Ubuntu 24.04 LTS with 2 GB RAM. In the firewall, allow **SSH only**; no other inbound ports, because the tunnel connects outward.
2. **Install Docker.** Follow Docker's official instructions for Ubuntu.
3. **Copy the zip over and check its hash.** Run `scp evade-api-<commit>.zip <user>@<vm>:~` and repeat the `sha256sum` check on the VM.
4. **Put the key in a root-only file, outside the code folder:**
   ```bash
   sudo install -m 600 /dev/null /etc/evade-api.env
   sudo sh -c 'read -rs K; printf "EVADE_API_KEY=%s\nEVADE_ALLOW_DRAFT=1\n" "$K" > /etc/evade-api.env'   # paste the key; not echoed
   ```
5. **Build and run, bound to localhost only:**
   ```bash
   docker build -t evade-api .
   docker run -d --name evade-api --restart unless-stopped --memory 1g \
              -p 127.0.0.1:8000:8000 --env-file /etc/evade-api.env \
              -v "$PWD/models:/models:ro" evade-api
   ```
6. **Add HTTPS with a Cloudflare Tunnel.** In the Cloudflare dashboard, go to Zero Trust → Networks → Tunnels and create a tunnel. Install `cloudflared` on the VM with the command it shows, and add a public hostname (for example `models-api.<team domain>`) that points to `http://localhost:8000`. Turn on "Always Use HTTPS" for the domain.
   - Optional and stronger: protect that hostname with Cloudflare Access and a service token that only the Worker holds.
7. **Check it:** run the curl commands from section 4 against `https://models-api.<team domain>`.

Updating means replacing the folder with the new zip, then `docker build` and `docker rm -f evade-api` followed by the `docker run` line again. A restart clears in-memory users, so do it between demos.

## 7. Connect the Worker

Whoever owns `cloud/worker`:

```bash
npx wrangler secret put EVADE_API_KEY          # paste the key
npx wrangler secret put EVADE_API_URL          # https://<cloud run url> or https://models-api.<team domain>
```

In the Worker:
- call `fetch(env.EVADE_API_URL + "/v1/predict/...", { method: "POST", headers: { "X-EVAde-Key": env.EVADE_API_KEY, "Content-Type": "application/json" }, body })`;
- store results in `model_results` / `risk_scores` using the mapping on PR #3;
- skip anything with `"source": "mock"`;
- on account deletion, also call `DELETE /v1/users/{user_id}`;
- never log request or response bodies, because they can contain health data.

## 8. Demo-day checklist

- [ ] Hash checked; the zip's `COMMIT` matches the version Michael announced
- [ ] Without a key, any route returns 401; with the key, `/v1/health` returns 200
- [ ] `/v1/models` shows `ecg_deep` available and the beta models loaded; sleep and seizure deep models show as `mock`
- [ ] Worker → API call works end to end: create a user, run a prediction, the result is stored in D1
- [ ] Budget alert set; exactly one instance
- [ ] The key exists only in the secret stores and the password manager
- [ ] Deployed at least an hour before the demo; no redeploys during it

## 9. Who does what

| Task | Owner |
|---|---|
| Build the zip, send the hash and the key, answer questions | Michael |
| Host it (option A or B), share the URL | anyone with a Google Cloud or Vultr account |
| Worker secrets, calls and D1 storage | the `cloud/worker` owner |
| Frontend labels ("placeholder", "flag for review, not a diagnosis") | the web app owner |

## Appendix: how Michael builds the zip

From the merged backend branch, on a machine with the model files:

```bash
C=$(git rev-parse --short HEAD); D=evade-api-$C
git archive --format=tar --prefix=$D/ HEAD evade requirements.txt Dockerfile .dockerignore .env.example docs/hosting-the-api.md | tar -x
echo $(git rev-parse HEAD) > $D/COMMIT && cp docs/hosting-the-api.md $D/HOSTING.md
mkdir -p $D/models && cp -r <model dir>/{ecg,seizure,ecg_deep} $D/models/      # plus spaceweather/ if ready
(cd $D && find models -type f -exec sha256sum {} + > models/SHA256SUMS)
zip -qr $D.zip $D && sha256sum $D.zip     # send the zip and the hash separately; the key only via the password manager
```

Before zipping, search the folder for your own usernames, server hostnames or IPs, and any filled-in key: `grep -rnE "<your-usernames>|<your-server-ips>|API_KEY=." $D` must print nothing. No private details and no secrets go in the zip.
