# Tutorial: run the EVAde API on Cloudflare Containers

This runs the same FastAPI service as the Google Cloud tutorial ([google-cloud-setup.md](google-cloud-setup.md)), but on Cloudflare: a Worker in front, one container behind it, and D1/R2 for data next door. Facts were checked against the Cloudflare docs on 2026-11-14. Containers is generally available (since April 2026), but the `durable_object` scheduling mode used here is in **public beta**, so re-check the linked pages before relying on details.

> The API returns flags for human review, never diagnoses. Use demo data only: public research data and pseudonymous IDs.

## Hard prerequisites (don't deploy until all are true)

1. **The API uses the `X-EVAde-Key` header.** It must read `EVADE_API_KEY` from the environment, require `X-EVAde-Key` on **every** route including `/v1/health`, compare it in constant time, have no CORS middleware, and have `/docs` and `/openapi.json` turned off. As of 2026-11-14 that change is being made on the API branch. Earlier versions use `Authorization: Bearer` with `EVADE_API_TOKEN` and would reject every call from this Worker.
2. **`EVADE_STORAGE=cloudflare` works:** D1 is created with `schema.sql` applied, and there's an R2 data bucket and a **separate models bucket**. See `evade/api/README.md` (Storage section). The container's memory and disk are wiped whenever it sleeps or its host restarts. Cloudflare says it "does not guarantee that any container instance will run for a set period." Never run this deployment on memory storage.
3. **Trained model folders are in the models bucket.** Upload them from a training machine with the write key, using `scripts/fetch_models.py --push <folder>`. Each folder has a `manifest.json` with sha256 checksums, which the API verifies before loading.

## When to use this, and when not

| Use Cloudflare Containers when | Stay on Cloud Run when |
|---|---|
| the prerequisites above hold | the API still keeps state in memory |
| you want the API next to D1/R2, with no cross-cloud traffic | you need an always-on instance (`--min=1`) |
| a few seconds of cold start after idle is acceptable | you need to choose the region |

## How it fits together

```
backend caller ──HTTPS──▶ Worker "evade-gateway" ──▶ Durable Object "EvadeApi" (one fixed name)
  (team server, or        • HTTPS only, POST/GET/DELETE    starts ONE container, waits until ready,
   service binding)       • caller key check (fail-closed)  forwards to http://container:8000
                          • size/path checks, rate limit     └─▶ our image: uvicorn, models from R2,
                          • adds X-EVAde-Key                      state in D1/R2
```

- **No public container endpoint.** A container "can only be accessed through its Durable Object", and only through HTTP via the Worker.
- **Exactly one instance.** Each container belongs to one Durable Object, so always calling `getByName()` with the same name gives one container. Keep it at one, because the API's deletion checks are per process.
- **The API key stays inside Cloudflare.** The backend caller sends its own key (`X-Gateway-Key`). The Worker verifies it, then adds the API's `X-EVAde-Key` from a secret. The two keys rotate separately. The backend team's client sends `X-Gateway-Key`, not `X-EVAde-Key`.

## Facts that shape the setup

| Topic | Current behaviour (docs) | What we do |
|---|---|---|
| Plan | Workers Paid ($5/month) is required; the Free plan has no Containers | upgrade first |
| Scheduling policy | `default` is deprecated. `durable_object` replaces it (public beta), and the policy **can't be changed** after the app is created | use `durable_object` from the start |
| Instance size | Presets are `lite` (256 MiB), `basic` (1 GiB) and `standard-1` (4 GiB, ½ vCPU). Under `durable_object`, `basic` is **rejected**, and the default is `lite`, which is too small: the API needs about 300–450 MB, and running out of memory restarts the instance | `instance: "standard-1"`. A custom `{vcpu, memoryMib, diskMb}` of about 1 GiB may work, but it's unverified, so test it |
| Instance count | `durable_object` has no `max_instances` setting; the count comes from the Durable Object names you use | one hard-coded name, plus a test that the name never changes |
| Image | linux/amd64. A Dockerfile build needs Docker running locally for `wrangler deploy`. Image size is capped at the instance's disk | `Dockerfile.cloudflare` (step 1) |
| Outbound network | off unless `start({ enableInternet: true })` | on: the container calls the D1 REST API and R2 over HTTPS |
| Bindings | not available inside the container | our storage layer uses the D1 REST API and the R2 S3 API |
| Forwarding | the docs' examples rewrite the URL to `http://container/...` and drop the `Host` header | same |
| Sleep | the container stops soon after the Durable Object goes idle, unless `setInactivityTimeout()` is set (max 6 h). The timeout is lost on Durable Object restarts | 1 h, set again in the constructor |
| Readiness | `start()` returns before the app listens; `running` doesn't mean ready | one shared readiness promise that polls an authed route |
| New image | under `durable_object`, a deploy does **not** restart the running container | an admin restart route |
| Cold start | "often 1–3 seconds", longer for big images. Ours adds Python imports and the model download | wait up to 45 s, then return `503` with `Retry-After` |
| Local secrets | `wrangler dev` reads secrets from `.dev.vars`, not from `wrangler secret put` | step 5 |
| Logs | turn on `observability`; logs are kept 7 days | never log bodies, keys or user IDs; uvicorn runs with `--no-access-log` |

## 0. Tools

- A Cloudflare account on Workers Paid.
- Node.js 20+ and Docker Desktop running (on Windows, use the WSL 2 backend).
- `npx wrangler --version` must be **4.136.0 or newer**, the documented minimum for this mode in local dev.

## 1. A container image that pulls models from R2

Containers have no volume mounts, so the image downloads the model folders (sha256-checked) at start. Use the repo's
`Dockerfile.cloudflare`:
- `python:3.12-slim`, with the pinned `requirements.txt` (including `onnxruntime` for the ONNX deep models) plus
  `boto3`; no torch, so the image stays small and cold starts stay short;
- at start it runs `scripts/fetch_models.py --dest /models` with the read-only models key, then uvicorn with
  `--no-access-log`. A folder that fails its checksum is skipped (listed as unavailable) and the API still starts;
- `.dockerignore` already includes `!scripts/fetch_models.py`;
- there is no `--platform` in the `FROM` line. `wrangler deploy` builds for linux/amd64; elsewhere use
  `docker build --platform linux/amd64 -f Dockerfile.cloudflare .`.

**Uploading models to the bucket** (training machine, models bucket WRITE key in the environment only):

```bash
python scripts/stamp_models_for_push.py --out push_ready --ecg <beta ecg> --seizure <beta seizure>        --spaceweather <spaceweather folder> --ecg-deep <deep ECG export> --id ecg_deep=EVA-13
for t in ecg seizure spaceweather ecg_deep; do python scripts/fetch_models.py --push push_ready/$t; done
```

The stamp script gives every folder a registry manifest with a real `EVA-n` ID (an `UNASSIGNED` folder is listed
but not served) and sha256 for every file. Folder names are the task names the API looks up: `ecg`, `seizure`,
`spaceweather`, `ecg_deep` (and `sleep_deep`, `seizure_deep` once trained). Then restart the container so it pulls them.

## 2. The Worker project

```bash
mkdir evade-gateway && cd evade-gateway
npm init -y
npm install --save-dev wrangler typescript @cloudflare/workers-types
npx wrangler login
echo ".dev.vars" >> .gitignore
```

`wrangler.jsonc`. Point `dockerfile` at your backend checkout, whatever its folder name is:

```jsonc
{
  "name": "evade-gateway",
  "main": "src/index.ts",
  "compatibility_date": "2025-11-17",
  "workers_dev": true,                 // false for option A in step 4 (service binding only)
  "preview_urls": false,
  "observability": { "enabled": true },
  "containers": [
    {
      "class_name": "EvadeApi",
      "scheduling_policy": "durable_object",
      "images": { "api": { "dockerfile": "../EVAde-backend-models/Dockerfile.cloudflare" } }
    }
  ],
  "durable_objects": { "bindings": [{ "name": "EVADE_API", "class_name": "EvadeApi" }] },
  "exports": { "EvadeApi": { "type": "durable-object", "storage": "sqlite" } },
  "ratelimits": [{ "name": "LIMITER", "namespace_id": "1001", "simple": { "limit": 120, "period": 60 } }]
}
```

- **Names must match:** the container `class_name`, the Durable Object binding `class_name` and the `exports` key must all equal the exported class.
- **SQLite-backed Durable Objects:** the `durable_object` policy requires them. New configs use `exports` instead of `migrations` / `new_sqlite_classes`, and the two can't be mixed.
- **Beta caveat:** this mode is in beta, so compare the field names, including the `ratelimits` binding, with the [Wrangler configuration reference](https://developers.cloudflare.com/workers/wrangler/configuration/).

## 3. The Worker and Durable Object code

`src/index.ts`:

```ts
import { DurableObject } from "cloudflare:workers";

interface Env {
  EVADE_API: DurableObjectNamespace<EvadeApi>;
  LIMITER: RateLimit;
  EVADE_API_KEY: string;            // secret: the key the API checks (X-EVAde-Key)
  GATEWAY_CALLER_KEY: string;       // secret: what the backend caller sends (X-Gateway-Key)
  ADMIN_KEY: string;                // secret: second key required for the restart route
  CF_ACCOUNT_ID: string; CF_API_TOKEN: string; CF_D1_DATABASE_ID: string;
  CF_R2_BUCKET: string; CF_R2_ACCESS_KEY_ID: string; CF_R2_SECRET_ACCESS_KEY: string;
  CF_R2_MODELS_BUCKET: string; CF_R2_MODELS_ACCESS_KEY_ID: string; CF_R2_MODELS_SECRET_ACCESS_KEY: string;
  CF_R2_ENDPOINT?: string;          // only for EU-jurisdiction buckets
}

const PORT = 8000;
const IDLE_MS = 60 * 60 * 1000;          // sleep after 1 h without requests (max 6 h)
const INSTANCE = "evade-api";            // ONE fixed name -> one container. Never derive it from the request.
const MAX_BODY = 1_000_000;              // the API's own cap is 1 MB
const READY_WAIT_MS = 45_000;
const FORWARD_HEADERS = ["content-type", "accept", "idempotency-key"];

// Fail-closed, constant-time key check: unset or short secrets and missing headers always deny.
async function keyOk(got: string | null, want: string | undefined): Promise<boolean> {
  if (!want || want.length < 32 || !got) return false;
  const enc = new TextEncoder();
  const [a, b] = await Promise.all([crypto.subtle.digest("SHA-256", enc.encode(got)),
                                    crypto.subtle.digest("SHA-256", enc.encode(want))]);
  return crypto.subtle.timingSafeEqual(a, b);
}

const err = (status: number, code: string, message: string, extra: Record<string, string> = {}) =>
  Response.json({ error: { code, message } }, { status, headers: extra });

export class EvadeApi extends DurableObject<Env> {
  private readyP?: Promise<void>;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    // the inactivity timeout is lost when the Durable Object restarts (for example after a deploy): set it again
    ctx.blockConcurrencyWhile(async () => {
      if (ctx.container?.running) await ctx.container.setInactivityTimeout(IDLE_MS);
    });
  }

  private containerEnv(): Record<string, string> {
    const e = this.env;
    const required = ["EVADE_API_KEY", "CF_ACCOUNT_ID", "CF_API_TOKEN", "CF_D1_DATABASE_ID", "CF_R2_BUCKET",
      "CF_R2_ACCESS_KEY_ID", "CF_R2_SECRET_ACCESS_KEY", "CF_R2_MODELS_BUCKET", "CF_R2_MODELS_ACCESS_KEY_ID",
      "CF_R2_MODELS_SECRET_ACCESS_KEY"] as const;
    for (const k of required) if (!e[k]) throw new Error(`missing secret ${k}`);   // names only, never values
    const env: Record<string, string> = { EVADE_STORAGE: "cloudflare", EVADE_ALLOW_DRAFT: "1" };
    for (const k of required) env[k] = e[k];
    if (e.CF_R2_ENDPOINT) env.CF_R2_ENDPOINT = e.CF_R2_ENDPOINT;
    return env;
  }

  private async startAndWait(): Promise<void> {
    const c = this.ctx.container!;
    if (!c.running) {
      c.start({ image: c.images.api, instance: "standard-1", enableInternet: true, env: this.containerEnv() });
      c.monitor().catch(() => {}).finally(() => { this.readyP = undefined; });   // forget readiness when it exits
    }
    await c.setInactivityTimeout(IDLE_MS);
    const deadline = Date.now() + READY_WAIT_MS;
    while (Date.now() < deadline) {
      try {   // an authed route: also proves the key wiring works
        const r = await c.getTcpPort(PORT).fetch("http://container/v1/health",
          { headers: { "X-EVAde-Key": this.env.EVADE_API_KEY }, signal: AbortSignal.timeout(2000) });
        if (r.ok) return;
      } catch { /* not listening yet */ }
      await new Promise((res) => setTimeout(res, 1000));
    }
    throw new Error("not ready");
  }

  async restart(): Promise<void> {        // picks up a newly deployed image on the next request
    this.readyP = undefined;
    if (this.ctx.container?.running) await this.ctx.container.destroy();
  }

  async fetch(request: Request): Promise<Response> {
    try {
      this.readyP ??= this.startAndWait();   // one start and one poll shared by all waiting requests
      await this.readyP;
    } catch (e) {
      this.readyP = undefined;
      console.error("container start failed", (e as Error).message.startsWith("missing secret") ? "config" : "timeout");
      return err(503, "starting", "API is starting, retry shortly", { "Retry-After": "10" });
    }
    const u = new URL(request.url);
    const fwd = new Request("http://container" + u.pathname + u.search, request);   // plain HTTP to uvicorn
    fwd.headers.delete("host");
    return this.ctx.container!.getTcpPort(PORT).fetch(fwd);
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.protocol !== "https:" && url.hostname !== "localhost") return err(403, "https_required", "use HTTPS");

    // backend callers only: fail-closed key check, then a rate limit per caller
    if (!(await keyOk(request.headers.get("X-Gateway-Key"), env.GATEWAY_CALLER_KEY)))
      return err(401, "unauthorized", "missing or wrong key");
    const { success } = await env.LIMITER.limit({ key: "gateway" });
    if (!success) return err(429, "rate_limited", "too many requests", { "Retry-After": "30" });

    const stub = env.EVADE_API.getByName(INSTANCE);

    if (url.pathname === "/__admin/restart") {      // needs BOTH keys; any other method is 404
      if (request.method !== "POST" || !(await keyOk(request.headers.get("X-Admin-Key"), env.ADMIN_KEY)))
        return new Response(null, { status: 404 });
      await stub.restart();
      return Response.json({ restarted: true });
    }

    if (!["GET", "POST", "DELETE"].includes(request.method) || !url.pathname.startsWith("/v1/"))
      return err(404, "not_found", "unknown route");

    // buffer the body (fixed Content-Length for the API, and a retry after a cold start can't lose a stream)
    let body: ArrayBuffer | null = null;
    if (request.method === "POST") {
      const cl = Number(request.headers.get("content-length"));
      if (!Number.isFinite(cl) || cl <= 0) return err(411, "length_required", "Content-Length header required");
      if (cl > MAX_BODY) return err(413, "payload_too_large", "body over 1 MB");
      body = await request.arrayBuffer();
      if (body.byteLength > MAX_BODY) return err(413, "payload_too_large", "body over 1 MB");
    }

    const headers = new Headers();                        // allow-list: nothing else reaches the API
    for (const h of FORWARD_HEADERS) { const v = request.headers.get(h); if (v) headers.set(h, v); }
    headers.set("X-EVAde-Key", env.EVADE_API_KEY);       // the API's key is added here and never leaves Cloudflare
    return stub.fetch(new Request(url.toString(), { method: request.method, headers, body }));
    // never log request bodies, headers or URLs here: they can carry health data and user IDs
  },
};
```

## 4. Who may call the Worker

- **Option A (best, if the team's backend is itself a Worker): a service binding.**
  - In the caller's config, add `"services": [{ "binding": "EVADE", "service": "evade-gateway" }]`. The caller then uses `env.EVADE.fetch(...)` and still sends `X-Gateway-Key`.
  - Set `"workers_dev": false` here with no route, so the gateway has **no public URL**.
  - Keep the key check anyway; it costs nothing.
  - Admin restarts then go through the caller, or through `npx wrangler` (step 7).
- **Option B: a caller key over HTTPS.** The backend server sends `X-Gateway-Key`. Keep that key server-side only, never in browser code.
- **Stronger, later:** put Cloudflare Access with a service token in front of the Worker.

## 5. Secrets

Each `secret put` prompts for the value, so it never lands in your shell history:

```bash
npx wrangler secret put EVADE_API_KEY
npx wrangler secret put GATEWAY_CALLER_KEY
npx wrangler secret put ADMIN_KEY
for n in CF_ACCOUNT_ID CF_API_TOKEN CF_D1_DATABASE_ID CF_R2_BUCKET CF_R2_ACCESS_KEY_ID CF_R2_SECRET_ACCESS_KEY \
         CF_R2_MODELS_BUCKET CF_R2_MODELS_ACCESS_KEY_ID CF_R2_MODELS_SECRET_ACCESS_KEY; do
  npx wrangler secret put $n
done
```

- **Generating the keys:** make the three keys at least 32 characters, with a password manager. Shorter keys are refused by `keyOk`.
- **Models key:** must be the **read-only** one. A server that can write to the models bucket could replace a model with malicious code.
- **Local development:** put **test** values (not the real keys) in `.dev.vars` (`NAME=value` per line). It's in `.gitignore`; never commit it.

## 6. Run it locally, then deploy

```bash
npx wrangler dev          # builds the image in Docker and runs the Worker and container locally; press [r] to rebuild
npx wrangler deploy       # uploads the Worker, then builds and pushes the image (Docker must be running)
```

After the **first** deploy, wait several minutes: "The Worker URL may respond while Cloudflare is still provisioning containers."

**Smoke test.** Set `GW=https://evade-gateway.<your-subdomain>.workers.dev` and a shell variable with your caller key, then check:

```bash
curl -s -o /dev/null -w "%{http_code}\n" $GW/v1/health                                   # expect 401 (no key)
curl -s $GW/v1/health -H "X-Gateway-Key: $GATEWAY_CALLER_KEY"                          # expect 200
curl -s -X POST $GW/v1/users -H "X-Gateway-Key: $GATEWAY_CALLER_KEY" \
     -H "Content-Type: application/json" -d '{"consent": true, "cohort": "demo"}'      # a POST end to end
```

The first request after idle can return `503 starting` with `Retry-After`; retry after that many seconds. Logs are in the dashboard under Workers & Pages → Containers, or the Worker's Logs tab.

## 7. Updating

- **New code or models:** run `npx wrangler deploy`. Under `durable_object` the running container keeps its old image until it restarts:
  ```bash
  curl -s -X POST $GW/__admin/restart -H "X-Gateway-Key: $GATEWAY_CALLER_KEY" -H "X-Admin-Key: $ADMIN_KEY"
  ```
  Do it between demos; a restart drops in-flight requests. With option A (no public URL), call it through the calling Worker instead.
- **New models only:** push them to the models bucket (`scripts/fetch_models.py --push`), then restart as above.
- **Rotate a key:** `npx wrangler secret put <NAME>`, then restart.

## Budget and limits

- **Billing alerts:** in the Cloudflare dashboard (Manage Account → Notifications), add billing and usage notifications for Workers and Containers with a monthly ceiling you're comfortable with. Alerts don't stop spending.
- **Request flood protection:** the rate-limit binding caps requests at 120 per minute. One fixed Durable Object name caps the container count at 1.

Rough costs, worked out from the pricing page (check it before relying on these; billing is per 10 ms while running, memory and disk on provisioned size, CPU on active use only; Workers Paid includes 25 GiB-hours of memory, 375 vCPU-minutes and 200 GB-hours of disk a month):

| Usage pattern | `standard-1` (4 GiB, 8 GB disk), per month |
|---|---|
| awake 24/7 | memory about $26 + disk about $1.4 + active CPU + the $5 plan + Worker and Durable Object requests |
| awake about 2 h a day | about 240 GiB-hours of memory and 480 GB-hours of disk: roughly $2 over the included amounts, plus the $5 plan |

Logs move to Cloudflare Observability pricing from 2026-12-01.

## Gotchas

| Symptom | Cause and fix |
|---|---|
| every call is 401 or 503 from the API | the API isn't on the `X-EVAde-Key` version yet (prerequisite 1), or `EVADE_API_KEY` differs between the Worker and the container |
| `503 starting` forever | a secret is missing (the log says `config`), or the container keeps exiting. Check the Containers dashboard logs |
| container restarts under load | out of memory (there's no swap). Check the instance size; `lite` is too small |
| `TypeError` on start | `instance: "basic"` isn't accepted in `durable_object` mode |
| old code still answering after a deploy | expected in this mode: run the restart in step 7 |
| D1 or R2 errors from the container | missing `enableInternet: true`, a missing variable (the API names missing variables, never values), or the token lacks permissions |
| a model listed as unavailable | checksum mismatch, or a draft without `EVADE_ALLOW_DRAFT=1`. Check `/v1/models` |
| startup crash mentioning the hostname | deployed containers have a 64-character hostname that doesn't resolve, which crashes some Python servers (`http.server`, `wsgiref`). The docs don't say whether uvicorn is affected; test locally and after the first deploy |
| `411` on POST | the caller didn't send `Content-Length`. Standard JSON clients do; chunked uploads aren't accepted |

## Teardown

```bash
npx wrangler delete                           # the Worker
npx wrangler containers list                  # if the container application is still listed:
npx wrangler containers delete CONTAINER_ID
npx wrangler containers images list
npx wrangler containers images delete IMAGE:TAG
```

The D1 database and R2 buckets stay until you delete them in the dashboard.

## Sources

- Overview, limits and pricing: https://developers.cloudflare.com/containers/ , /containers/platform/limits/ , /containers/platform/pricing/
- Scheduling policy (`durable_object`, instance types): https://developers.cloudflare.com/containers/configuration/scheduling-policy/
- Durable Object Container API (`start`, `getTcpPort`, `setInactivityTimeout`, `monitor`, `destroy`): https://developers.cloudflare.com/containers/api/durable-object-container/
- Env vars, secrets and `.dev.vars`: https://developers.cloudflare.com/containers/examples/env-vars-and-secrets/
- Scaling and routing: https://developers.cloudflare.com/containers/configuration/scaling-and-routing/
- Outbound traffic: https://developers.cloudflare.com/containers/configuration/outbound-traffic/
- FAQ (sleep, restarts, logs): https://developers.cloudflare.com/containers/faq/
- Wrangler containers commands: https://developers.cloudflare.com/workers/wrangler/commands/containers/
