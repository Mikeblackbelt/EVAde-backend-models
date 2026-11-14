# Planning: Automatic Risk Score and Days to Return

**Status: theoretical design. No code, no validation, not clinical guidance.** Every number below is a design parameter chosen so the system is concrete and testable. Some are anchored in the literature in this review; most are assumptions to be elicited from experts and calibrated on data later. The parameter register at the end says which is which.

## What this plans

Two automatically computed outputs per crew member, refreshed on a schedule, with human review:

| Output | Definition | Range |
|---|---|---|
| **Risk score R** | Estimated overall health risk that a clinician would recommend intervention or return planning within the next 14 days | open interval (0, 1), never exactly 0 or 1 |
| **Days to return (DTR)** | Recommended latest time to return to Earth (or to definitive care) given the current risk, its trend, and hard exposure limits | displayed as **"<5"**, a number from 5 to 100, or **">100"** |

**Interpretation taken:** "days to return" means days until the person should be back on Earth. The alternative reading, days until a person returns to duty after an event, is a different model (a recovery-time estimate) and is not planned here.

## Files

| File | Contents |
|---|---|
| [risk-score.md](risk-score.md) | Five domains, anchor tables, noisy-OR aggregation (version 2), flag, overrides, data-quality handling, bands, actions |
| [days-to-return.md](days-to-return.md) | The DTR formula, trend adjustment, hard-limit times (including the extreme-domain cap), logistics floor, intervals, display rules |
| [worked-examples.md](worked-examples.md) | Three scenarios computed by hand, plus single-domain sensitivity |
| [data-acquisition-plan.md](data-acquisition-plan.md) | Per-modality data sources (public research data, no TUSZ), a 2-day hackathon scope using only instant-download data, access steps, gaps, and an action checklist |
| [implementation-pipeline.md](implementation-pipeline.md) | Three-stage build plan: data ingestion and preprocessing, model training, FastAPI endpoints (`/predict/ecg`, `/predict/eeg`, `/predict/radiation-risk`, `/predict/risk`), CORS, Pydantic, latency budget, build order |
| [calibration-and-validation.md](calibration-and-validation.md) | How the numbers would be fitted and tested, targets, failure modes, governance |

## Design principles

1. **Personal baselines first.** Most inputs are deviations from the person's own baseline, not population norms. Individual variation in confinement is large (Mars-500 crew responses ranged from none to persistent insomnia), so a fixed population threshold is the wrong model.
2. **One feature, one domain.** No raw feature feeds two domains, so nothing is double counted. HRV feeds the behavioral stress component only. The cardiac domain uses heart rate, rhythm, SpO2, and respiration.
3. **Single failures do not hide, and do not alone hit critical.** The aggregation is a noisy-OR in log space with a 0.75 cap per domain. A single abnormal domain (s at or above 0.30) puts displayed R at 0.60 or more ("high"); reaching "critical" needs two severe domains or a hard-rule override.
4. **Missing data is not good news.** A domain with poor data is scored at least 0.30 and flagged low-confidence.
5. **Weak evidence gets a cap.** EEG affective markers are near zero effect in large samples, so they run in shadow mode. Wrist SpO2 carries known skin-tone bias, so its uncertainty is widened. Portable bone proxies are unvalidated, so bone is capped unless measured.
6. **Humans decide.** R and DTR are decision support. Nothing triggers action autonomously. Hard-rule overrides escalate to a clinician; they do not take clinical action.

## Parameter register

| Parameter | Value | Basis |
|---|---|---|
| Risk label horizon | 14 days | design assumption |
| Domain influence factors c_d | cardiac 1.0, behavioral 1.0, radiation 0.8, neuro 0.6, bone 0.6 | design assumption, to be elicited (former weights divided by 0.25) |
| Dead zone, per-domain cap | 0.10, 0.75 | design assumption |
| Noisy-OR map | R = 0.005 + 0.99 x (1 - exp(-L)) | design assumption (keeps R inside (0, 1)) |
| Abnormal-component flag | domain s at or above 0.30; floors displayed R at 0.60; not applied to DTR; excludes low-data floors and shadow-mode inputs | design assumption |
| Extreme-domain DTR cap | eligible domain s_d at or above 0.80: DTR at most 9 days, linear to 2 days at s_d = 1.0 | design assumption (requirement: any extremely high domain gives DTR under 10 days) |
| Per-domain false-alarm budget | about 0.2 high-or-above alerts per person-month | design assumption (overall target of 1 divided across 5 domains) |
| Risk band edges | 0.2, 0.4, 0.6, 0.8 | design assumption |
| PHQ-9 anchors | 5, 10, 15, 20 | literature (Kroenke 2001 severity bands) |
| GAD-7 anchors | 5, 10, 15 | literature (Spitzer 2006) |
| Career dose limit | 600 mSv | literature, from the radiation research notes (NASA-STD-3001 Vol. 1 Rev B); verify the current revision |
| Per-event solar particle limit | 250 mSv | literature, from the same notes; verify |
| Deep-space cruise dose rate | 1.8 mSv/day | literature (Hassler 2014, MSL cruise) |
| In-flight sleep, typical | about 6 h/night | literature (Barger 2014, press-reported figure) |
| Hip bone strength loss (finite element) | 2.0-2.6 %/month | literature, from the bone research notes; verify |
| Areal BMD loss prior | about 1 %/month | literature range 0.4-1.8 %/month depending on measure and site |
| DTR curve | 150 days x (1 - R)^1.5 | design assumption |
| Trend penalty | exp(-5 x max(0, change in R over 7 days)) | design assumption |
| Dose safety margin | 25 mSv | design assumption (about 4% of 600) |
| Return-vehicle lead time L | scenario-specific (examples in days-to-return.md) | assumption per mission architecture |

Literature values marked "verify" come from subagent research summaries and have not been re-checked in full text.
