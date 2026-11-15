# Raspberry Pi Bluetooth stream test: synthetic sensors to measure API responsiveness

**Status: plan only, no code.** It's a proof of concept, so the goal is useful numbers in about a day, not a benchmark study. It builds on [bluetooth-device-integration.md](bluetooth-device-integration.md) (device profiles, gateway options), the ingest contract in [api-schema.md](api-schema.md) (`/v1/devices/bind`, `/v1/ingest/batch`, `seq`), and the deployment guides in `docs/`. Bluetooth throughput figures and library details below are estimates to check in phase 1, not measured facts.

## 1. Goal and questions

A Raspberry Pi pretends to be one or more wearables and streams **synthetic** physiology over Bluetooth Low Energy (BLE). A gateway receives the stream and calls the EVAde API the way a real deployment would. We measure:

1. **Latency.** p50/p95/p99 time per call for `ingest/batch`, `predict/ecg`, `predict/eeg`, `predict/ecg-deep` and `predict/risk`.
2. **Event-to-flag time.** An anomaly is injected on the Pi (an arrhythmia segment, a seizure segment, a heart-rate spike). How long until the API returns a flag for it?
3. **Capacity.** How many simulated crew members, with their devices, can one API instance serve before p95 latency or the error rate crosses the targets?
4. **Robustness.** What happens on BLE disconnects, Wi-Fi drops, duplicate or missing `seq`, API restarts, and oversized or malformed batches?
5. **Stability.** Do memory, latency and error rates stay flat over a 2-hour soak?

## 2. Setup

```
Raspberry Pi (BLE peripheral)         Gateway (BLE central)                    EVAde API
synthetic generators  ──BLE notify──▶  laptop or 2nd Pi (Python + bleak)  ──HTTPS──▶ staging instance
 • HR strap   (0x180D standard)         • buffer, window, seq numbers             (Cloud Run, 1 instance)
 • ECG patch  (custom service, 1 lead)  • X-EVAde-Key (staging key only)
 • PPG wrist  (custom service)          • timing log (no signal values)
 • EEG band   (custom service, 4 ch)
        └──────── Wi-Fi HTTP path (stress only: 12-lead ECG, 18-channel EEG) ────────▶
```

- **Hardware:**
  - a Raspberry Pi 4 or 5 (both have BLE 5.0 and Wi-Fi) running 64-bit Raspberry Pi OS with BlueZ;
  - a laptop with Bluetooth as the gateway (or a second Pi);
  - optionally, a real heart-rate chest strap to compare the synthetic stream with a real device.
- **Gateway mode A, Python with `bleak`:** the main test path, scriptable and with precise timing.
- **Gateway mode B, Web Bluetooth page:** a smoke test only, for the frontend demo path (Chromium over HTTPS). It shows the browser path works, but isn't used for timing.
- **Security:**
  - The test targets a **staging** API with its own test key.
  - In production a device gateway would **not** hold the model API key: it would go through the team's backend Worker with the user's session, and the Worker adds `X-EVAde-Key`. So the test measures the API itself; Worker overhead is a separate, small measurement (phase 5).
- **Data labelling:** all data is synthetic and labelled. Devices are bound with `kind: "simulated"` and windows carry `source: "simulated"`, so the API marks every result provisional. Test users are deleted afterwards with `DELETE /v1/users/{id}`.

## 3. Simulated devices and the bandwidth check

BLE is the bottleneck for raw waveforms, so each simulated device copies what real devices send:

| Simulated device | GATT | Data | Approximate payload | Over BLE? |
|---|---|---|---|---|
| Heart-rate strap | standard Heart Rate Service 0x180D, Measurement 0x2A37 (with RR intervals) | HR + RR, about 1 notification per second | < 0.1 kbps | yes |
| ECG patch | custom 128-bit service | 1 lead, 250 Hz, int16 | about 4 kbps | yes |
| PPG wrist | custom | 1-2 channels, 64 Hz | about 1-2 kbps | yes |
| EEG headband | custom | 4 channels, 256 Hz, int16 | about 16 kbps | yes (check in phase 1) |
| 18-channel clinical EEG | custom | 18 channels, 256 Hz | about 74 kbps | stretch only; default over Wi-Fi |
| 12-lead ECG | n/a | 12 leads, 500 Hz | about 96 kbps | **no**: Wi-Fi only |

Notes:
- **Packet format.** Each notification carries a small header: device sample counter, generation time in ms, and a scenario tag. The gateway uses it to measure BLE loss and latency without logging any signal values. Payloads use the negotiated MTU (BlueZ negotiates ATT MTUs above 23; data length extension depends on the controllers at both ends, so confirm it with `btmon`), but measure the real rate in phase 1 rather than trusting nominal numbers.
- **Which models the BLE devices feed.** The deep ECG model needs **12 leads**, so it is tested over the Wi-Fi path with replayed 12-lead records. The BLE single-lead patch feeds the beta `/v1/predict/ecg` (HR, HRV, rhythm backstop) and the baseline comparison. That's also true of real single-lead patches.
- **API rules the gateway must respect** (from the API):
  - windows are 2-60 s;
  - an ingest batch holds at most 20 windows or 500 readings;
  - request bodies are at most 1 MB with a `Content-Length` header (30 s of 18-channel EEG is about 138k JSON numbers, so it fits only with compact values of about 6-7 characters each; full-precision floats fit only about 10 s);
  - `seq` increases by one per batch per device.

## 4. Synthetic data

Two sources:

1. **Generators, for continuous streams.** `neurokit2` (`ecg_simulate`, `ppg_simulate`) for heart rate and waveforms with chosen HR and HRV, plus 1/f noise, baseline wander and motion artifacts. EEG background is replayed from public recordings rather than generated, because generated EEG isn't realistic enough for the models.
2. **Replay, for known events.** Short segments from the public datasets already used for training: PTB-XL 12-lead ECG (CC BY 4.0), CHB-MIT/Siena seizure EEG, Sleep-EDFx. The events come with ground truth:
   - a seizure starting at a known second;
   - an MI-labelled 12-lead record;
   - a tachycardia stretch;
   - a period where HRV falls below the user's baseline.

**Scenario scripts** are YAML timelines that define what each device sends and when:
- **Baseline:** 2-5 min of calm data so the per-user baselines lock (demo mode needs 30 windows; the seizure model needs at least 30 four-second EEG windows).
- **Normal:** steady state.
- **Events:**
  - a seizure segment at t = 10 min;
  - HR 150 for 2 min at t = 15 min;
  - HRV drop at t = 20 min.
- **Faults:** BLE link killed for 30 s; Wi-Fi down for 2 min; a replayed duplicate `seq`; a skipped `seq`; a malformed window; an oversized batch.

The same scenario file drives the BLE path, the Wi-Fi path and the HTTP-only load generator, so results are comparable.

## 5. Measurements

| What | How |
|---|---|
| Per-call latency | the gateway logs send time, response time, status, endpoint, `seq` and sample count (no values) |
| BLE loss and delay | sample counter gaps; generation time on the Pi vs receive time on the gateway (both clocks synced with chrony/NTP; record the offset) |
| Event-to-flag | the scenario's event start time on the Pi vs the first API response carrying a flag for that event (for example `seizure.flag` = `review` or `suspect_seizure`) |
| Server side | Cloud Run request latency and memory charts; the API's own timing header if added (no bodies or values in logs, per the security rules) |
| Correctness | the flag fires for injected events and not during normal stretches (counts only, PoC level) |
| Robustness | duplicates ignored, gaps counted (`next_seq_expected`), 411/413/422 handled, recovery time after restarts |

**Targets (PoC, staging on Cloud Run in the same region as the gateway):**

| Metric | Target |
|---|---|
| `ingest/batch` p95 | < 300 ms |
| `predict/ecg`, `predict/eeg` p95 | < 500 ms |
| `predict/ecg-deep` p95 (2 MB ONNX model) | < 500 ms |
| `predict/risk` p95 | < 200 ms |
| event-to-flag, seizure | < 10 s (4 s window plus the step, upload and processing) |
| event-to-flag, HR or HRV | < 5 s after the first complete window that contains it |
| error rate at the target load | < 1 % (excluding injected faults) |
| BLE sample loss (Pi within 2 m) | < 1 % |
| soak, 2 h | flat memory (in-memory state bounded), no p95 drift above 20 % |

## 6. Test tiers

| Tier | Load | Path | Purpose |
|---|---|---|---|
| T0 | 1 user, HR strap | BLE | wiring check, baseline latency |
| T1 | 1 user: HR, 1-lead ECG, PPG and 4-ch EEG | BLE | realistic single astronaut; event-to-flag |
| T2 | crew of 4-6 users (virtual device IDs multiplexed from 1-2 Pis) | BLE + Wi-Fi | realistic crew load |
| T3 | ramp 1 → 50 virtual devices | HTTP-only load generator (Locust or k6) replaying the same scenarios | find the knee of a single API instance (`--max=1`) |
| T4 | crew load for 2 h | BLE + Wi-Fi | soak: memory, drift, reconnects |
| T5 | fault scenarios | all | robustness: disconnects, offline buffering and catch-up, duplicate or missing `seq`, API restart (in-memory state loss is expected until Cloudflare storage is on) |

For deployments behind the Cloudflare Worker gateway, rerun T1 and T3 through the Worker to measure its added latency and the 120 requests/min rate limit.

## 7. Components to build

| Component | Location | Notes |
|---|---|---|
| Pi peripheral | `tools/pi_sim/peripheral.py` | Python BLE GATT server on BlueZ (for example the `bless` library, or BlueZ's D-Bus GATT example). Advertises the services in section 3; notifications are driven by the scenario timeline |
| Generators and replay | `tools/pi_sim/sources.py` | neurokit2 generators plus dataset replay; fixed seeds so runs repeat |
| Scenario files | `tools/scenarios/*.yaml` | baseline, normal, events, faults |
| Gateway | `tools/gateway/gateway.py` | `bleak` client: subscribe, buffer into 2-10 s windows, bind devices, number batches, upload with retries and an offline queue, timing log |
| Wi-Fi sender | `tools/gateway/http_sender.py` | sends the same scenario over HTTP for the 12-lead ECG and 18-channel EEG paths |
| Load generator | `tools/loadtest/locustfile.py` | many virtual devices from scenario files; no BLE |
| Report | `tools/loadtest/report.py` → `docs/stream-test-results.md` | latency percentiles per endpoint and tier, event-to-flag table, error counts, a few plots |

Install on the Pi: `bluez`, Python 3.11+, `bless`, `numpy`, `neurokit2`. On the gateway: `bleak`, `httpx`, `numpy`. Secrets (the staging key) come from environment variables only, never from files in the repo.

## 8. Plan and timing (about one working day)

| Step | Time | Done when |
|---|---|---|
| 1. Pi setup and BLE throughput calibration | 1-2 h | the Pi advertises a test service; the gateway measures achievable notification throughput and loss at 1 m and 5 m |
| 2. Generators, replay and scenarios | 2 h | a scenario file produces repeatable streams with ground-truth event times |
| 3. Peripheral services | 2 h | HR (standard), ECG, PPG and EEG services stream from the scenario |
| 4. Gateway and timing log | 2 h | T0 and T1 run end to end against staging; latency log written |
| 5. Tiers T2-T5, plus the Worker comparison | 2-3 h | numbers for each tier |
| 6. Report | 1 h | `docs/stream-test-results.md` with targets met or missed and the bottleneck found |

## 9. Risks

| Risk | Mitigation |
|---|---|
| BLE throughput lower than expected for 4-channel EEG | send EEG over Wi-Fi, or lower its rate; keep the BLE path for HR, ECG and PPG |
| BLE drops or interference at the venue | test at 1-2 m; the gateway's reconnect and offline queue are part of what's measured |
| Clock offset spoils the one-way latency | chrony on both; report round-trip times as the primary number |
| Single API instance saturates early | that's a finding: report the knee, then decide on optimizations (batching, smaller windows) before scaling. Scaling out needs shared storage first (one process per database) |
| Synthetic data trips flags unexpectedly | that's useful: log the counts, and keep scenario seeds fixed so runs can be repeated |
| Real data mixed in by mistake | `kind: "simulated"`, a separate staging deployment and key, test users deleted after each run |

## 10. Open items

- **Gateway machine:** a second Pi or a laptop, depending on what's available at the hackathon.
- **Worker test:** whether the team Worker is ready to test through, or only the direct staging API.
- **Real-device comparison:** whether a real HR strap is available.
