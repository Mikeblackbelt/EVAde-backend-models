# EVAde space weather (radiation inputs)

Turns space-weather data into inputs for the radiation risk domain (`evade/risk/engine.py`, `radiation_domain` and the MDTR radiation limit). Everything here gives flags and estimates for human review. **It is not dosimetry and not a warning service.**

| Part | What it does | Where |
|---|---|---|
| Providers | one small interface, a NOAA SWPC reference adapter (public, no key), a file replay adapter, a cache | `providers.py`, `schema.py` |
| (a) Nowcast | NOAA S-, R- and G-scale levels, flare class, SPE-in-progress flag | `model.py` `nowcast` |
| (b) Forecast | P(>=10 MeV proton flux reaches 10 pfu within 24 h), given it is quiet now: a draft model (EVA-8) trained on GOES 1998-2019 | `features.py`, `train.py`, `model.py` `forecast` |
| (c) Dose rate | GCR baseline by location plus an SPE term from proton flux and a shielding preset; a measured dosimeter rate replaces it | `model.py` `dose_estimate` |
| API | `POST /v1/predict/space-weather` | `evade/api/app.py` |

## What your API needs to return

Write an adapter (below) that returns these fields, converted to these units. One record per timestamp, and any record may carry any subset of fields. Missing values are left out (not 0, not -99999).

| Field | Unit | Needed for | Required? |
|---|---|---|---|
| `t_utc` | ISO-8601 UTC timestamp | everything | **yes** |
| `proton_10mev_pfu` | integral proton flux >=10 MeV, pfu = protons cm-2 s-1 sr-1 | S-scale, SPE flag, forecast, dose | **yes** (no protons, no result) |
| `proton_50mev_pfu` | integral >=50 MeV, pfu | forecast features | strongly recommended |
| `proton_100mev_pfu` | integral >=100 MeV, pfu | dose behind spacecraft or shelter shielding, forecast | strongly recommended |
| `xray_long_wm2` | 0.1-0.8 nm (XRS-B) irradiance, W m-2, **true irradiance** (GOES-R units; if your source is GOES-8..15 operational data, divide by 0.7) | R-scale, flare class, forecast features | strongly recommended |
| `xray_short_wm2` | 0.05-0.4 nm (XRS-A), W m-2 | display only | optional |
| `kp` | planetary K index, 0-9 | G-scale | optional |
| `sw_speed_kms`, `sw_density_cm3`, `bz_gsm_nt` | solar wind speed (km/s), density (cm-3), IMF Bz GSM (nT) | display only | optional |
| `dose_rate_usv_h` | measured onboard dose-equivalent rate, microSv/h | replaces the modelled dose rate | optional |

Cadence and history: 1-min or 5-min samples for protons and X-rays work best. The SPE flag uses NOAA's rule (three consecutive 5-min points at or above 10 pfu), and coarser data fall back to the latest value. The forecast needs the **last 48 hours**, so `fetch(hours=72)` should return at least 48 h. Kp and solar wind may be hourly or 3-hourly.

### Adding your adapter

```python
from evade.spaceweather.providers import SpaceWeatherProvider, ProviderUnavailable, register
from evade.spaceweather.schema import Observations, Record

class MyApiProvider(SpaceWeatherProvider):
    name = "my_api"
    def fetch(self, hours: int = 72) -> Observations:
        rows = call_my_api(hours, timeout=8)          # raise ProviderUnavailable on any failure
        recs = [Record.make(r["time"], proton_10mev_pfu=r["p10"], proton_100mev_pfu=r["p100"],
                            xray_long_wm2=r["xrs_b"]) for r in rows]
        return Observations(self.name, recs)

register("my_api", MyApiProvider())                    # wrapped in the cache (5 min success, 1 min failure)
```

Then add `"my_api"` to the `provider` allow-list in `evade/api/schemas.py` (`SpaceWeatherRequest`). `Record.make` drops non-finite and out-of-range values. Clients choose providers by name only and never pass a URL or a path. Raw observations can also be POSTed directly (`observations`: up to 5000 records, none older than 7 days or in the future, spanning at most 96 h; otherwise 422), which needs no adapter at all. Inference also drops records older than 50 h before the latest one before building the hourly grid, so a provider that returns a long history cannot make the API build a huge frame.

## NOAA SWPC reference provider

`noaa_swpc` reads `services.swpc.noaa.gov` (public, no key; only that host is allowed, https only, 8 s timeout per feed, 8 MB cap):
`/json/goes/primary/integral-protons-3-day.json`, `/json/goes/primary/xrays-3-day.json`, `/products/noaa-planetary-k-index.json`, `/products/solar-wind/plasma-3-day.json`, `/products/solar-wind/mag-3-day.json`. Both SWPC payload styles (a list of objects, or a list of lists with a header row) are parsed. A missing proton feed means `provider_unavailable` (HTTP 503). Missing X-ray, Kp or solar wind feeds become warnings. Results are cached for 5 minutes and failures for 1 minute, with one fetch at a time, so NOAA is called at most about 5 times per 5 minutes whatever the request rate. NASA DONKI is not used.

`replay` (only when `EVADE_SPACEWX_REPLAY_FILE` is set) replays a normalized JSON file (`{"source": ..., "records": [{"t_utc": ..., <fields>}]}`), shifted so the last record is "now".

## (a) Nowcast

NOAA scales (standard thresholds): S1-S5 at >=10 MeV flux 10, 100, 1e3, 1e4, 1e5 pfu; R1-R5 at 0.1-0.8 nm peak 1e-5 (M1), 5e-5 (M5), 1e-4 (X1), 1e-3 (X10), 2e-3 (X20) W m-2 (R uses the max over the last hour); G1-G5 at Kp 5-9. Flare class is truncated, as NOAA defines it. Data older than 30 min raise a warning.

`spe_in_progress` feeds `spe_active` of `/v1/predict/radiation-risk`. **Override O5 is not raised from flux.** `risk-score.md` defines O5 as SPE dose >= 250 mSv or projected to reach it within 24 h, so the endpoint returns `o5.candidate` when the *estimated* event dose or its 24 h projection reaches 250 mSv, for a human to confirm (the same pattern as the EEG seizure candidate).

## (b) Forecast (EVA-7 architecture, EVA-8 model, draft)

- **Task:** at an hour when the hourly mean >=10 MeV flux is below 10 pfu, will it reach 10 pfu in any of the next 24 hours (SPE onset)? The hourly mean stands in for NOAA's three-point rule. During an event the endpoint returns `event_in_progress`, not a probability.
- **Data:** NOAA NCEI GOES SEM 5-min averages, 1998-2019 (`scripts/download_goes_history.sh`): integral protons (GOES-8..12 EPS `p3/p5/p7_flux_ic`; GOES-13/15 EPEAD corrected `ZPGT10/50/100`, mean of the E and W detectors that pass QC) and XRS (5-min for GOES-8..12, 1-min for GOES-13/15; per month the satellite with the most valid hours, gaps filled from the others; divided by 0.7 to undo the SWPC scaling, so values match GOES-R true irradiance).
- **Features (19, `features.py`, the same code for training and serving):** log flux at >=10/50/100 MeV, 24 h max and 48 h min of log >=10 MeV flux, its 1/3/6 h changes, spectral hardness (50/10, 100/10), log X-ray flux now and its max over 3/6/24/48 h, 24 h X-ray fluence, counts of M- and X-class hours in 48 h, hours since the last M-class hour (capped at 48). A test checks that features from a 22-year history and from the trailing 49 hours are identical.
- **Splits (no leakage across storms):** protocol A, time-forward: train 1998-2010, Platt recalibration 2011-2012, test 2013-2019, with a 72 h embargo (48 h lookback + 24 h horizon) at every boundary. Logistic vs gradient boosting was chosen by blocked CV inside 1998-2012 only. Protocol B: leave-one-block-out over 9 contiguous blocks with the same embargo (robustness). Baselines: climatology (the training base rate) and persistence (logistic on the current log flux only). CIs come from a 27-day block bootstrap.

RESULTS_PLACEHOLDER

## (c) Crew dose-rate estimate

`dose = GCR_baseline(location) + SPE_term`, where `SPE_term = T_location x T_shielding x 4 pi x max(0, F_channel - F_background) x S_tissue x 1.602e-10 Gy per MeV/g x w_R`. A measured `dose_rate_usv_h` replaces the estimate (`source: "measured"`). Every parameter is an assumption and is returned in `data.assumptions`:

| Parameter | Value | Evidence |
|---|---|---|
| GCR baseline, ISS (`leo_iss`) | 0.647 mSv/day | Berger 2017, DOSIS Columbus, end of May 2016 (PARTIAL) |
| GCR baseline, deep space | 1.8 mSv/day | derived from Zeitlin 2013 (0.66 Sv over a ~360-day round trip); also used in max-days-to-return.md (PARTIAL, derived) |
| GCR baseline, lunar surface | 1.37 mSv/day | 13.2 microGy/h absorbed in silicon (Zhang 2020, VERIFIED) x assumed quality factor 4.3 (UNVERIFIED) |
| GCR baseline, Mars surface | 0.67 mSv/day | as quoted in max-days-to-return.md from Hassler 2014; the abstract gives no number (UNVERIFIED) |
| Shielding presets | `eva_suit`: >=10 MeV, S = 15 MeV cm2/g; `spacecraft` (~10 g/cm2): >=100 MeV, S = 6.5; `storm_shelter` (~30 g/cm2): 0.3 x >=100 MeV, S = 5 | order-of-magnitude physics (UNVERIFIED; not transport-code output) |
| Location SPE transmission | deep space 1.0; lunar surface 0.5; Mars surface 0.25; LEO/ISS 0.01 | geometry and geomagnetic-shielding assumptions (UNVERIFIED) |
| Quiet background subtracted | 1.0 / 0.3 / 0.2 pfu at >=10 / 50 / 100 MeV | typical quiet GOES levels (UNVERIFIED) |
| Proton weighting factor w_R | 2 | ICRP 103 value, not in the lit review (UNVERIFIED here) |
| Limits | 600 mSv career, 250 mSv per SPE | NASA-STD-3001 Vol 1 Rev B (VERIFIED) |

Artemis I measured about fourfold dose differences between shielding locations (George 2024), so treat the estimate as an order of magnitude. Mapping to the radiation domain (`radiation_inputs`): `dose_equivalent_rate_msv_per_day` is the rate above; `rate_ratio_to_7d_median` is that rate divided by the quiet GCR baseline (there is no 7-day history here); `spe_dose_msv` is the SPE term integrated over the current run of samples at or above 10 pfu; and `spe_projected_dose_msv_24h` is that dose plus the current SPE rate held for 24 h (persistence, which is conservative near the peak). With `cumulative_dose_msv`, the response also previews `radiation_domain` and `T_rad = (600 - dose - 25) / rate`. **Nothing is stored:** forward `radiation_inputs` to `/v1/predict/radiation-risk` to update the user's radiation domain.

## Model files

`<EVADE_MODEL_DIR>/spaceweather/{model.joblib, manifest.json, metrics.json, feature_names.json}` (plus `provenance.json`: the source file of each month). `manifest.json` gives the task, model and architecture IDs, format, version, training data ranges, library versions and the sha256 of each file. The API loads the model only with `EVADE_ALLOW_DRAFT=1`, and only after the sha256 of `model.joblib` matches the manifest. joblib unpickles code, so load only artifacts built by `evade.spaceweather.train` on a machine you control. Model files are not committed to git.

## Run

```
bash scripts/download_goes_history.sh 1998 2019        # on the server; sequential, resumable, about 1 h, about 1.3 GB
python -I -c "import sys; sys.path.insert(0, '<code dir>'); from evade.spaceweather.train import main; sys.exit(main())" \
       --raw ~/evade_data/other/spaceweather/raw --out <model_dir> [--smoke]
```

On the shared server, run it through `guarded_job.sh` (4 threads, 8 GB and 1 h are plenty; the full run takes LIMITS_RUNTIME).

## Limits

- Trained on GOES-8..15 (to 2019). The live feed is GOES-16/18/19 (SGPS and XRS on GOES-R), which are different instruments. Proton fluxes are designed to be comparable but are not identical; X-rays are put on the same scale by undoing the 0.7 factor. The 5-min XRS of GOES-8..12 slightly lowers hourly peaks compared with 1-min data.
- Proton satellites change over the record (GOES-8, 10, 11, 13, 15), so the background levels shift slightly.
- SPE onsets are rare (tens of independent events in the test period), so the CIs are wide. The forecast uses only GOES proton and X-ray history: no CME, radio burst or active-region information, which operational forecasters use.
- The dose estimate is a parameterized demo, not a radiation transport calculation.
