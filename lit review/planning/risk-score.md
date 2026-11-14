# Risk Score R in (0, 1)

## 1. Pipeline

1. Raw streams produce **features** (per-domain, below), each with a quality value in [0, 1].
2. Each feature maps to a **component score** in [0, 1] through an anchor table (linear interpolation between anchors; below the first anchor = 0; above the last = the last value).
3. Components combine into a **domain score** s_d in [0, 1].
4. Domain scores combine into a **stress index X** in [0, 1], then into **R** by a logistic.
5. **Hard-rule overrides** may raise R to a floor of 0.80 and escalate to a human.
6. **Quality and uncertainty** are attached; a band and a recommended action follow.

## 2. Domains, components, anchors

Anchors are (feature value -> component score). "Own baseline" means the person's trailing 14-day median unless stated.

### Cardiac and vitals (weight 0.25)

| Component | Feature | Anchors |
|---|---|---|
| C1 Resting heart rate | absolute z-score vs own baseline (both tachycardia and bradycardia) | 1 -> 0; 2 -> 0.3; 3 -> 0.6; 4 -> 1.0 |
| C2 Rhythm | fraction of valid ECG time flagged irregular in the last 24 h | 0 -> 0; 1% -> 0.3; 5% -> 0.6; 20% -> 1.0 |
| C3 SpO2 | 5-min median, only windows with signal quality >= 0.8 (percent) | 95 -> 0; 92 -> 0.5; 88 -> 1.0 |
| C4 Respiration rate | breaths per minute | 12-20 -> 0; 24 -> 0.5; 30 -> 1.0; below 10 -> 0.5; below 8 -> 1.0 |

Domain score: s_card = 0.6 x max(C) + 0.4 x mean(C), over components with valid data.

SpO2 note: pulse oximetry error varies by skin tone in published studies, and a January 2025 FDA draft guidance asks for at least 150 participants across three skin-tone groups (from the vitals research notes; verify status). Add 2 percentage points to the SpO2 uncertainty before comparing with anchors, and do not let C3 alone trigger an override unless confirmed by a second sensor.

### Behavioral and sleep (weight 0.25)

| Component | Feature | Anchors |
|---|---|---|
| B1 Sleep | 7-day mean sleep, hours | 7 or more -> 0; 6 -> 0.35; 5 -> 0.7; 4 or less -> 1.0 |
| B2 Depressive symptoms | latest PHQ-9 (within 14 days) | 0 -> 0; 5 -> 0.15; 10 -> 0.5; 15 -> 0.75; 20 -> 1.0 |
| B3 Anxiety symptoms | latest GAD-7 (within 14 days) | 0 -> 0; 5 -> 0.15; 10 -> 0.5; 15 -> 0.9; 21 -> 1.0 |
| B4 Vigilance | brief PVT lapse count, ratio to own baseline | 1x -> 0; 2x -> 0.3; 4x -> 0.7; 8x -> 1.0 |
| B5 Physiological stress | drop in RMSSD vs own 7-day median (percent) | 0 -> 0; 20 -> 0.2; 40 -> 0.5; 60 -> 0.7 (capped at 0.7) |
| B6 Activity | 7-day mean activity vs own baseline (percent change) | -15 -> 0; -30 -> 0.3; -60 -> 0.7 |

Domain score: s_beh = 0.6 x max(B) + 0.4 x mean(B), over components with valid data.

Why the caps and anchors: sleep around 6 h is typical in flight, so it earns 0.35 rather than a high score; depression-related HRV effects are modest and not specific (meta-analytic effect around g = -0.46 for RMSSD in major depression, from the HRV research notes), so B5 is capped at 0.7; reduced activity and altered sleep emerged in Mars-500 (Basner 2013). PHQ-9 and GAD-7 anchors follow their published severity bands.

### Neuro (weight 0.15)

| Component | Feature | Anchors |
|---|---|---|
| N1 Seizure | detector events with confidence of at least 0.9 in 24 h | 0 -> 0; 1 -> 0.8; 2 or more -> 1.0 |
| N2 Drowsiness | EEG drowsiness index, z vs own baseline | 1 -> 0; 2 -> 0.3; 3 -> 0.6 (capped at 0.6) |
| N3 Affective markers | EEG-based mood/anxiety features | **shadow mode, weight 0** |

Domain score: s_neu = max(N1, N2).

N3 stays in shadow mode (computed and logged, not scored) because large studies show near-zero effects for frontal alpha asymmetry (d about -0.007 in n = 1,883 patients vs 2,161 controls, per the EEG research notes) and reported deep-learning accuracies are often inflated by subject leakage. N3 may enter scoring only after it adds measurable value in held-out, subject-wise tests on this project's own data.

### Radiation (weight 0.20)

| Component | Feature | Anchors |
|---|---|---|
| R1 Cumulative dose | career effective dose f = D / 600 mSv | f 0.05 -> 0; 0.25 -> 0.25; 0.50 -> 0.5; 0.75 -> 0.8; 1.0 or more -> 1.0 |
| R2 Solar particle event | event dose as a fraction of 250 mSv | 0 -> 0; 0.10 (25 mSv) -> 0.2; 0.50 (125 mSv) -> 0.6; 1.0 or more -> 1.0 |
| R3 Dose-rate anomaly | current rate / 7-day median | below 1.5x -> 0; 1.5x -> 0.2; 3x -> 0.6; 10x -> 1.0 |

Domain score: s_rad = max(R1, R2, R3).

The 600 mSv and 250 mSv limits come from the radiation research notes (NASA-STD-3001 Volume 1 Revision B) and need a check against the current revision. Radiation score reflects exposure, not symptoms. Data trust rule: if the detector reports more than 1 single-event upset per hour, mark the domain low-confidence.

### Bone (weight 0.15)

| Component | Feature | Anchors |
|---|---|---|
| K1 Areal BMD change from pre-flight (percent) | measured DXA or estimated | 0 -> 0; -3 -> 0.25; -6 -> 0.5; -10 -> 1.0 |
| K2 Finite element hip strength change (percent) | if CT-derived values exist | -10 -> 0.25; -20 -> 0.5; -30 -> 1.0 |

Domain score: s_bone = max(K1, K2) of the available measurements. When no measured value exists, K1 uses a model estimate (prior of about 1 %/month areal loss times months in flight, blended with any portable proxy) and **s_bone is capped at 0.6**, because portable proxies are not validated against spaceflight bone loss.

Rates for reference: DXA areal losses of 0.4-1.8 %/month and finite element hip strength losses of 2.0-2.6 %/month (bone research notes; verify). At the DXA prior, 100 days is about -3.3%, which scores about 0.28.

## 3. Aggregation

- Weights: w_card 0.25, w_beh 0.25, w_neu 0.15, w_rad 0.20, w_bone 0.15 (sum 1.00).
- Weighted mean: S = sum of w_d x s_d.
- Worst domain: M = max of s_d.
- Stress index: **X = 0.6 x S + 0.4 x M**.
- Risk: **R = 1 / (1 + exp(-8 x (X - 0.45)))**.

Properties: X = 0 gives R about 0.027; X = 1 gives R about 0.988; X = 0.45 gives R = 0.5. R is strictly inside (0, 1) for any finite X. Because X is in [0, 1], R stays between about 0.027 and 0.988.

## 4. Hard-rule overrides (floor R at 0.80 and escalate to a human)

| ID | Trigger | Escalation |
|---|---|---|
| O1 | any PHQ-9 item 9 response above 0 | clinician contact within 1 hour |
| O2 | SpO2 below 88% for at least 5 min, signal quality at least 0.8, confirmed by a second sensor | immediate clinician review |
| O3 | high-risk rhythm flag sustained at least 30 s with ECG quality at least 0.8 | immediate clinician review |
| O4 | seizure event with detector confidence at least 0.9 | immediate clinician review |
| O5 | solar particle event dose at least 250 mSv, or projected to reach it within 24 h | immediate crew and ground alert |

An override raises R to max(R, 0.80). It does not change the domain scores, so the display shows both the computed R and the override reason.

## 5. Data quality and confidence

- **Domain quality** Q_d = fraction of expected valid windows in the evaluation period (0 to 1).
- If Q_d < 0.5: set s_d = max(s_d, 0.30) and mark the domain "low confidence".
- **Overall quality** Q = sum of w_d x Q_d. If Q < 0.6, label R "provisional".
- **Uncertainty:** run 1,000 Monte Carlo draws that perturb feature values by their sensor uncertainty and perturb anchors by 10%; report the 10th-90th percentile interval of R.

## 6. Update cadence and stability

| Domain | Feature window | Re-evaluation |
|---|---|---|
| Cardiac and vitals | 5-min windows, 30-min medians | every 15 min |
| Radiation | continuous | every 15 min |
| Neuro | 24 h event count, 30-min drowsiness | every 15 min |
| Behavioral | 7-day rolling, 14-day baseline | daily (questionnaires weekly) |
| Bone | 90-day trend | monthly, or on a new scan |

- Displayed R is the median of the last four evaluations (1 hour).
- A band increases after 2 consecutive evaluations above its lower edge; it decreases only after 12 hours continuously below that edge minus 0.03 (hysteresis).

## 7. Bands and actions

| R | Band | Action |
|---|---|---|
| below 0.2 | low | routine monitoring |
| 0.2 to 0.4 | guarded | double the sampling cadence; include in the routine medical report |
| 0.4 to 0.6 | elevated | flight surgeon review within 24 h; countermeasure plan |
| 0.6 to 0.8 | high | flight surgeon review within 4 h; start return planning |
| 0.8 or above | critical | immediate clinician contact; execute return plan |

Targets, to be tuned: at most 1 alert per person-month at "high" or above in a healthy cohort, and at most 0.1 per person-month at "critical".
