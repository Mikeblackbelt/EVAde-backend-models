# EVAde space weather (radiation inputs)

Turns space-weather data into inputs for the radiation risk domain (`evade/risk/engine.py`, `radiation_domain` and the MDTR radiation limit). Everything here gives flags and estimates for human review. **It is not dosimetry and not a warning service.**

| Part | What it does | Where |
|---|---|---|
| Providers | one small interface, a NOAA SWPC reference adapter (public, no key), a NASA DONKI event adapter, a file replay adapter, a cache | `providers.py`, `schema.py` |
| (a) Nowcast | NOAA S-, R- and G-scale levels, flare class, SPE-in-progress flag, DONKI context (recent CMEs, flares, shocks, SEPs) | `model.py` `nowcast`, `donki_summary` |
| (b) Forecast | P(>=10 MeV proton flux reaches 10 pfu within 24 h), given it is quiet now: draft model EVA-9 (GOES protons and X-rays, per-satellite background correction), trained 1998-2016, tested 2017-2026 including GOES-R | `features.py`, `background.py`, `train.py`, `model.py` `forecast` |
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
| `proton_satellite` | GOES satellite number of the proton values (for example 18) | picks the per-satellite background correction; without it the proton background is not corrected (warning) | recommended if your data come from GOES |

If your API relays the SWPC operational feed, set `meta["background_prefix"] = "swpc:"` on the `Observations` (as `noaa_swpc` does), because SWPC's GOES-R integral flux has a different quiet floor from the NCEI archive values used in training. CMEs are optional and separate: an adapter can return them as DONKI-style events (`meta["events"]["cme"]`: `t_utc`, `speed_kms`, `half_angle_deg` or `width_deg`, `earth_directed`), or a client can POST them as `cmes` next to `observations`. The served model (EVA-9) does not use them; the CME variant (EVA-11) does.

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
`/json/goes/primary/integral-protons-3-day.json`, `/json/goes/primary/xrays-3-day.json`, `/products/noaa-planetary-k-index.json`, `/json/rtsw/rtsw_wind_1m.json` and `/json/rtsw/rtsw_mag_1m.json` (real-time solar wind, active spacecraft only; the older `/products/solar-wind/*` files returned 404 on 2026-11-15). Live check 2026-11-15: protons (GOES-18), X-rays and Kp parsed, 49 h of forecast coverage, `p_onset_24h` 0.008 on a quiet day. Both SWPC payload styles (a list of objects, or a list of lists with a header row) are parsed. A missing proton feed means `provider_unavailable` (HTTP 503). Missing X-ray, Kp or solar wind feeds become warnings. Results are cached for 5 minutes and failures for 1 minute, with one fetch at a time, so NOAA is called at most about 5 times per 5 minutes whatever the request rate. Proton records carry `proton_satellite` from SWPC's `satellite` field, which selects the background correction.

## NASA DONKI provider

`donki` fetches the last 4 days of CME, FLR, SEP, IPS and GST events. It tries the keyless CCMC endpoint (`kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/<type>`) first. If that does not answer, it uses `api.nasa.gov/DONKI/<type>` with `NASA_API_KEY` from the environment (never stored or logged), or the public `DEMO_KEY` if the variable is not set. It has the same etiquette as NOAA: 10 s timeout, 8 MB cap, one fetch at a time. Successes are cached for 1 hour and failures for 10 minutes, because `DEMO_KEY` allows about 10 requests per hour and windows of at most 60 days. CMEs keep the most accurate analysis (speed, half-angle, latitude and longitude). `earth_directed` is true if any ENLIL run gives an Earth arrival and null if no run exists. With `provider: noaa_swpc` the endpoint adds DONKI by default (`include_donki`). A DONKI failure is only a warning. Live check 2026-11-15: CCMC did not answer from the lab server or a local PC, so api.nasa.gov answered CME, SEP and IPS; its FLR and GST routes returned HTTP errors and became warnings.

`replay` (only when `EVADE_SPACEWX_REPLAY_FILE` is set) replays a normalized JSON file (`{"source": ..., "records": [{"t_utc": ..., <fields>}]}`), shifted so the last record is "now".

## (a) Nowcast

NOAA scales (standard thresholds): S1-S5 at >=10 MeV flux 10, 100, 1e3, 1e4, 1e5 pfu; R1-R5 at 0.1-0.8 nm peak 1e-5 (M1), 5e-5 (M5), 1e-4 (X1), 1e-3 (X10), 2e-3 (X20) W m-2 (R uses the max over the last hour); G1-G5 at Kp 5-9. Flare class is truncated, as NOAA defines it. Data older than 30 min raise a warning.

`spe_in_progress` feeds `spe_active` of `/v1/predict/radiation-risk`. **Override O5 is not raised from flux.** `risk-score.md` defines O5 as SPE dose >= 250 mSv or projected to reach it within 24 h, so the endpoint returns `o5.candidate` when the *estimated* event dose or its 24 h projection reaches 250 mSv, for a human to confirm (the same pattern as the EEG seizure candidate).

## (b) Forecast (draft; served model EVA-9)

- **Task:** at an hour when the hourly mean >=10 MeV flux is below 10 pfu, will it reach 10 pfu in any of the next 24 hours (SPE onset)? The hourly mean stands in for NOAA's three-point rule. During an event the endpoint returns `event_in_progress`, not a probability.
- **Data:**
  - **GOES-8..15:** NOAA NCEI 5-min averages, 1998 to 2020-03 (`scripts/download_goes_history.sh`). Integral protons come from the EPS `p3/p5/p7_flux_ic` channels or the EPEAD corrected `ZPGT10/50/100` (E/W mean). XRS is divided by 0.7 to undo the SWPC scaling.
  - **GOES-16 and GOES-18:** NCEI L2 (`scripts/download_goesr_history.sh`). SGPS 5-min averages start at NCEI in 2020-11. GOES-16 is used to 2022-12 and GOES-18 after, and each fills the other's gaps. XRS uses the 1-min science files, already true irradiance. SGPS publishes only differential channels, so the >=10/50/100 MeV integral is computed: the sum of j x dE above the threshold, plus the >500 MeV channel (`archive.py`).
  - **Gap:** no GOES-R proton data exist at NCEI for 2017 to 2020-10, and GOES-15 stops in 2020-03.
- **Features:** the same 19 GOES features as before (`features.py`), computed after the per-satellite background correction (`background.py`, below). The CME variant adds three features: the fastest CME speed in 72 h, whether any CME in 72 h was wider than 120 degrees, and the fastest such CME. For training they come from the SOHO/LASCO CDAW catalogue, assuming each CME is known 3 h after first sight. At serving they come from DONKI (width = 2 x half-angle).
- **Split (one, time-forward):** train 1998-2016, test 2017-01 to 2026-11, with a 72 h embargo at the boundary. The model is logistic regression with no recalibration step (EVA-8's Platt step had made things worse). Persistence is a logistic on the current log flux only.

### Results (2026-11-15, draft)

Test 2017-01 to 2026-11: 78,878 quiet hours, 1,009 positive, 39 independent onsets. The GOES-R era (2020-11 to 2026-11, solar cycle 25) holds 37 of the 39 onsets (51,424 hours, 955 positive).

| Model | Brier, all test | AUROC, all test | Brier, GOES-R era | AUROC, GOES-R era |
|---|---|---|---|---|
| **EVA-9** GOES + background correction (served) | 0.0118 | 0.890 | 0.0172 | 0.832 |
| EVA-11 GOES + CME features + background correction | 0.0117 | 0.892 | 0.0170 | 0.837 |
| GOES without background correction (comparison) | 0.0113 | 0.892 | 0.0165 | 0.836 |
| Persistence | 0.0117 | 0.763 | 0.0169 | 0.741 |
| Climatology (training rate 1.63%) | 0.0126 | 0.5 | 0.0182 | 0.5 |

- **Does DONKI/CME add skill?** Barely. In the GOES-R era, AUROC rises by 0.005 (95% block-bootstrap CI +0.0001 to +0.012), but the Brier difference CI (-0.0004 to +0.00004) includes zero. Over the whole test period both CIs include zero. The pre-stated rule (serve the CME model only if its Brier gain is significant over the whole test period) was not met, so EVA-9 is served and DONKI is shown as context.
- **Calibration in the GOES-R era (EVA-9):** the model is well calibrated below p = 0.01 (0.0045 vs observed 0.0046) and over-forecasts above that: 0.14 vs 0.07 in the 0.1-0.2 bin, 0.38 vs 0.20 in 0.3-0.5, 0.68 vs 0.45 above 0.5. Mean p is 0.027 against a rate of 0.019, so about 1.5x too high, better than EVA-8's 2x on cycle 24. Its Brier is about the same as persistence's (0.0172 vs 0.0169) while its ranking is much better (AUROC 0.83 vs 0.74).
- The GOES-15 era of the test (2017 to 2020-03) holds only 2 onsets. Its scores (AUROC 0.997) are not informative.

### Per-satellite background correction

- **Method:** `corrected = max(flux - b_sat + b_ref, 0.1 x b_ref)` on hourly protons. `b_sat` is the 25th percentile over quiet hours, per satellite and calendar year (nearest year as fallback). The reference is GOES-15 (all years). The served artifact carries its own table. `background_default.json` is the committed copy and the documented fallback, and an unknown satellite gets no correction.
- **Quiet floors at >=10 / 50 / 100 MeV (pfu):**

  | Satellite | >=10 MeV | >=50 MeV | >=100 MeV |
  |---|---|---|---|
  | GOES-15 | 0.14 | 0.056 | 0.029 |
  | GOES-16, NCEI-integrated | 0.70 | 0.50 | 0.45 |
  | GOES-18, NCEI-integrated | 0.36 | 0.23 | 0.21 |
  | GOES-18, SWPC live feed (7 days to 2026-11-15) | 0.23 | 0.21 | 0.20 |

  On 93 common hours, SWPC's GOES-18 integral is lower than our NCEI integration (median log10 ratio -0.29 / -0.14 / -0.11), which is why the live feed has its own `swpc:18` entry.
- **Overlap check, GOES-16 vs GOES-18, 2022-09 to 2022-12** (2,586 quiet hours, no event hours, overlap excluded from the floor estimates). Median |log10 ratio|:

  | Channel | Before correction | After correction |
  |---|---|---|
  | >=10 MeV | 0.036 | 0.61 |
  | >=50 MeV | 0.018 | 1.04 |
  | >=100 MeV | 0.028 | 1.69 |

  The two GOES-R satellites already agree raw. Subtracting floors estimated in different years leaves GOES-R instrument noise that is larger than GOES-15's much lower floor, so hour-by-hour quiet agreement gets worse. The correction targets the generation gap (GOES-8..15 vs GOES-R floors 5-15x apart at >=100 MeV), and those generations have no time overlap to validate on.
- **Effect on the forecast:** none measurable. The uncorrected comparison model scores slightly better on Brier (0.0165 vs 0.0172 in the GOES-R era), with the same AUROC. That may be partly coincidence: GOES-R's higher floors raise its probabilities in a more active cycle.

### Earlier model (EVA-8, kept as is)

EVA-8 (train 1998-2010, Platt 2011-2012, test 2013-2019, GOES-8..15 only) scored test Brier 0.00594 and AUROC 0.918 vs persistence 0.714 on 15 onsets, and over-forecast about 2x in cycle 24. Its artifact (commit a99b862) is unchanged; see MODEL_REGISTRY.md.

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

`<EVADE_MODEL_DIR>/spaceweather/{model.joblib, manifest.json, metrics.json, feature_names.json, background.json}`, plus `provenance.json` (the source file of each GOES-8..15 month). `model.joblib` holds the estimator and the background table. `manifest.json` gives the task, model and architecture IDs, format, version, training and test ranges, library versions and the sha256 of each file. The API loads the model only with `EVADE_ALLOW_DRAFT=1`, and only after the sha256 of `model.joblib` matches the manifest. joblib unpickles code, so load only artifacts built by `evade.spaceweather.train` on a machine you control. Model files are not committed to git.

## Run

```
bash scripts/download_goes_history.sh 1998 2020     # GOES-8..15 CSV, ~1.3 GB, ~1 h, sequential and resumable
bash scripts/download_goesr_history.sh [--dry-run]  # GOES-16 2020-11..2022 and GOES-18 2022-09 on: SGPS + XRS, ~1.5 GB
curl -o univ_all.txt https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/text_ver/univ_all.txt   # CDAW CMEs, 4 MB
python -I -c "import sys; sys.path.insert(0, '<code dir>'); from evade.spaceweather.train import main; sys.exit(main())"        --raw <goes csv root> --goesr <goes-r root> --cdaw univ_all.txt --out <model_dir> [--swpc-live] [--smoke]
```

`--swpc-live` estimates the live SWPC feed's quiet floor from its 7-day window (network). `scripts/download_donki_history.sh` exists for a full DONKI history once a NASA API key is available (see Limits). On the shared server, run training through `guarded_job.sh` with 4 threads and 8 GB; the full run took about 45 s and stayed well inside the guard's 8 GB limit. Training needs `h5py` to read the GOES-R netCDF files; serving does not.

## Limits (proof of concept)

- **DONKI history was not used for training.** On 2026-11-15 the keyless CCMC endpoint did not answer from the lab server or a local PC. api.nasa.gov with `DEMO_KEY` allows about 10 requests per hour and windows of at most 60 days, and its FLR route returned 404. So the CME features were trained on the SOHO/LASCO CDAW catalogue and are served from DONKI. CDAW speeds are plane-of-sky, DONKI's are deprojected (usually higher), so served CME features run high compared with training. The DONKI SEP list was not used as a label check either. With a NASA API key, the download script can fetch the full history.
- **GOES-R integral protons are integrated by us** from SGPS differential channels. They are not SWPC's operational product, and NCEI has no GOES-R proton L2 data before 2020-11.
- The background correction does not improve the GOES-16/18 overlap agreement or the forecast. It removes the generation offset in the quiet floors, but it cannot be validated on a common time window across generations.
- The test has one split. Its 39 onsets sit almost all in cycle 25. Probabilities are about 1.5x too high in the GOES-R era. Use the output as a ranked watch signal for human review.
- XRS: the 5-min GOES-8..12 data slightly lower hourly peaks compared with the 1-min GOES-13+ data. The forecast uses no radio burst or active-region information.
- The dose estimate is a parameterized demo, not a radiation transport calculation.
