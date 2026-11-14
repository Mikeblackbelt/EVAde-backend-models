# Bluetooth Device Integration (EVAde)

**Status: plan only, no code.** How wearable and handheld sensors reach the EVAde API over Bluetooth Low Energy (BLE), tied to the per-user baseline design in [baseline-comparison.md](baseline-comparison.md) and the endpoints in [implementation-pipeline.md](implementation-pipeline.md). Standard BLE service identifiers below are from the Bluetooth SIG GATT specifications as I know them; device sampling rates and proprietary protocols were **not verified in this review** and must be checked against each vendor's documentation before building.

## 1. Where Bluetooth fits

The cloud backend cannot talk to a device directly. A **client near the person** connects over BLE and forwards data over HTTPS or WebSocket to the FastAPI service with the `user_id`.

```
sensor (BLE)  ->  client (browser, phone, or laptop gateway)  ->  API (/ingest, /compare)  ->  baseline, models, risk
```

| Client option | How | Good for | Limits |
|---|---|---|---|
| **Web Bluetooth in the frontend** | the Cloudflare-hosted page connects to the device on a button press and streams to the API | hackathon: no install, one code base with the UI | works in Chromium browsers over HTTPS only, needs a user gesture each session, not in Safari or iOS browsers, limited background operation |
| **Python gateway app** (for example using the `bleak` library) on a laptop or Raspberry Pi | scans, connects, buffers, uploads | long sessions, reconnect logic, offline buffering, proprietary protocols | needs installation; one more thing to demo |
| **Native phone app** | full BLE stack | production | out of hackathon scope |
| **Fallbacks** | the prototype's simulator, file upload (CSV or EDF), manual entry | demo when hardware or Bluetooth fails | not real-time sensing |

Recommendation for the 2-day hackathon: Web Bluetooth for standard-profile devices, the simulator as the guaranteed fallback, and a Python gateway only if a proprietary-protocol EEG headset is needed. Always rehearse with the fallback, because Bluetooth is the most likely live-demo failure.

## 2. Device and profile matrix

| Signal | Preferred path | BLE interface | Notes |
|---|---|---|---|
| Heart rate and RR intervals | any chest strap or watch exposing the standard Heart Rate profile | Heart Rate Service (0x180D), Heart Rate Measurement characteristic (0x2A37), which can carry RR intervals when the device sets the flag | easiest first device; chest straps are more accurate than wrist PPG; wrist-derived intervals are pulse-rate variability, not ECG HRV (see [vital-sign-monitoring](../vital-sign-monitoring/README.md)) |
| Raw single-lead ECG | a chest strap that streams raw ECG | vendor-specific streaming service (for example Polar's proprietary data service), not a standard profile | confirm rate (the Polar H10 is commonly stated as 130 Hz, not verified here), so resample to the pipeline's 250 Hz or compute HRV at the native rate |
| EEG | a consumer headband with a documented or reverse-engineered stream | proprietary; the prototype's 4 channels (AF7, AF8, TP9, TP10) match a Muse-style headband, commonly 256 Hz (not verified here) | use a vendor SDK or an open library (BrainFlow, LSL-based tools); consumer EEG has few channels and dry electrodes, so only simple features (band power ratios, drowsiness) are credible |
| SpO2 | fingertip or wrist device | Pulse Oximeter Service (0x1822) where supported | wrist and consumer SpO2 carry known skin-tone bias (see the vitals section); keep in shadow mode for the flag until validated |
| Blood pressure | cuff | Blood Pressure Service (0x1810) | spot readings, not continuous |
| Body temperature | thermometer | Health Thermometer Service (0x1809) | spot readings |
| Blood glucose | meter | Glucose Service (0x1808) | spot readings; note any use-case limits |
| Activity and sleep | watch or ring | usually proprietary cloud APIs rather than raw BLE | take summaries through the vendor API or manual entry; do not pool across brands without a bridging study |
| Battery | all | Battery Service (0x180F) | send battery level with every stream; low battery marks data quality |

Rule: prefer **standard GATT profiles** because they work across vendors and reduce reverse-engineering. Use proprietary streams only for ECG and EEG where no standard exists.

## 3. Pairing, identity, and binding

| Item | Rule |
|---|---|
| Device binding | a `device_id` (hardware address or a vendor ID, hashed) is bound to exactly one `user_id` at setup by an authenticated action; the API rejects a stream whose device is not bound to the calling user |
| One person, many devices | allowed; each device has a modality and a priority (primary or fallback) |
| Rebinding | requires the user to confirm; the old binding is archived |
| Metadata stored | device model, firmware, modality, first-seen date, battery |
| Why it matters | the baseline is per person and per device, so a device change must start a bridging period, not silently reuse the old baseline |
| Link security | use encrypted, paired connections where the device supports it; do not log hardware addresses in plain text; do not store raw waveforms server-side beyond the demo |

## 4. Ingest protocol (client to API)

| Endpoint | Method | Purpose |
|---|---|---|
| `/devices/bind` | POST | bind a device to the `user_id` |
| `/devices/{device_id}` | GET | status, last seen, battery, signal quality |
| `/ingest/batch` | POST | upload a batch of samples or readings (offline buffer flush, spot readings) |
| `/ingest/stream` | WebSocket | live samples with the `user_id` and token; server acknowledges sequence numbers |

Every message carries:

| Field | Purpose |
|---|---|
| `user_id`, `device_id`, `modality` | identity and routing |
| `seq` | monotonically increasing per device; gaps are detected and reported |
| `t0_utc`, `fs` | start time and sample rate; the client timestamps from its own clock and reports the offset estimated at connection |
| `samples` or `readings` | data with units |
| `quality` | lead-off, contact quality, motion, or the device's own flags |
| `battery` | percent when known |
| `client_info` | client type and version |

Server rules: reject data older than a set age or from the future beyond a small tolerance; deduplicate by (`device_id`, `seq`); limit message size and rate; return a typed error, never a silent drop; pass accepted windows to preprocessing and then to `/compare`.

## 5. Handling real-world BLE problems

| Problem | Handling |
|---|---|
| Disconnects and reconnects | client auto-reconnect with backoff; buffer locally (a few minutes) and flush with the original timestamps; the server marks the gap, so the baseline does not interpolate silently |
| Packet loss and out-of-order | `seq` numbers; gap flags; reject windows with more than a set fraction missing (design: more than 10% of samples) |
| Clock drift | record client-to-server offset at start and every few minutes; correct timestamps; report residual drift in the quality field |
| Variable sampling rate | resample to the pipeline's grid inside preprocessing; keep the original `fs` |
| Motion and contact artifacts | device quality flags plus the signal quality index from [implementation-pipeline.md](implementation-pipeline.md); low-quality windows are returned with `low_quality` and excluded from baselines |
| Throughput | EEG at 256 Hz with 4 channels is small; send in 1 to 2 second chunks; do not stream raw waveforms to the cloud if features suffice (compute features at the client and upload them where privacy or bandwidth requires) |
| Device swap or firmware update | new baseline phase or a bridging period (see baseline-comparison.md); log firmware |
| Battery and sleep modes | surface low battery; mark data after a battery warning as lower confidence |
| Multiple users near one client | a client session is bound to one `user_id`; a device bound to another user is refused |

## 6. Spaceflight considerations (low confidence, not researched here)

Bluetooth radios in a spacecraft raise electromagnetic compatibility, radio-frequency certification, and approved-hardware questions that depend on the vehicle and the agency. Radiation can also cause upsets in consumer electronics (see [space-radiation-detection](../space-radiation-detection/README.md)). This plan treats BLE as a **terrestrial and analog-habitat** interface and lists spaceflight use as an open question to be answered with the relevant safety and operations authorities, not assumed. A wired or approved-link fallback should be planned for any real mission.

## 7. Mapping to the platform features

| Platform feature | Bluetooth input | Used by |
|---|---|---|
| Heart rate, HRV | heart rate profile or raw ECG | `/predict/ecg`, cardiac domain, stress features |
| Brain activity | EEG headband | `/predict/eeg`, neuro domain (drowsiness) |
| Oxygen, blood pressure, temperature, glucose | standard profiles | vitals domain, with measurement type and device in the record |
| Sleep, activity | vendor summaries or manual entry | behavioral and sleep components |
| Mood, journal | not Bluetooth; journaling and voice pipeline | behavioral domain (LLM-derived, labelled as such) |
| Radiation, bone | not Bluetooth in this plan; telemetry and scans | radiation and bone domains |

## 8. Hackathon plan

| Time box | Task |
|---|---|
| 1-2 h | choose devices that are on hand; check which expose standard profiles; confirm the browser and HTTPS requirement |
| 2-4 h | prototype a Web Bluetooth page that connects to a heart rate device, reads heart rate and RR intervals, and posts batches with `user_id`, `device_id`, `seq`, `t0_utc` to a mock `/ingest/batch` |
| 4-8 h | add reconnect and buffering, quality flags, and the bind step; feed the stream through the baseline comparison |
| 8-12 h | optional: EEG or ECG via the proprietary route (gateway app or vendor SDK) if a device is available; otherwise replay the prototype simulator through the same ingest endpoint |
| 12 h | freeze the interface; write the fallback switch (live, replay, simulator) and rehearse it |

## 9. Tests

| Test | Pass condition |
|---|---|
| Binding | a stream from an unbound device, or a device bound to another user, is rejected |
| Gaps | an induced 5 s disconnect produces a gap flag and no interpolated data in the baseline |
| Duplicates | a repeated batch is ignored |
| Clock | a client with a deliberately wrong clock is corrected or rejected |
| Quality | a lead-off or motion segment returns `low_quality` and is not used for the baseline |
| Throughput | one hour of a 4-channel 256 Hz stream plus heart rate runs without queue growth |
| Fallback | switching to the simulator needs one action and keeps the same API shape |

## 10. Open items

- Which devices the team actually has, and which expose standard profiles.
- Whether the demo runs Web Bluetooth only or also a gateway app.
- Vendor terms for EEG and ECG streaming SDKs (not checked).
- Spaceflight Bluetooth approval (not researched).
- Whether features are computed at the client or on the server (privacy versus consistency).
