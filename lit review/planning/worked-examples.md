# Worked Examples

All inputs are invented for illustration. Formulas are in [risk-score.md](risk-score.md) (aggregation version 2, noisy-OR) and [days-to-return.md](days-to-return.md). Calculations were done by hand and rounded.

Recap:
- p_d = c_d x 0.75 x max(0, (s_d - 0.10) / 0.90), with c: cardiac 1.0, behavioral 1.0, radiation 0.8, neuro 0.6, bone 0.6.
- L = sum of -ln(1 - p_d); R_agg = 0.005 + 0.99 x (1 - exp(-L)).
- Flag: any validated domain with s_d at or above 0.30 gives displayed R = max(R_agg, 0.60). DTR uses R_agg (floored at 0.80 only by a hard-rule override).
- DTR_R = 150 x (1 - R_dtr)^1.5.

## One component calculation (behavioral domain, scenario B)

Inputs: 7-day sleep 5.4 h; PHQ-9 = 9; GAD-7 = 8; PVT lapses 3x baseline; RMSSD down 35%; activity down 35%.

| Component | Interpolation | Score |
|---|---|---|
| B1 sleep, 5.4 h | between 6 h (0.35) and 5 h (0.70): 0.35 + 0.6 x 0.35 | 0.56 |
| B2 PHQ-9, 9 | between 5 (0.15) and 10 (0.5): 0.15 + 0.8 x 0.35 | 0.43 |
| B3 GAD-7, 8 | between 5 (0.15) and 10 (0.5): 0.15 + 0.6 x 0.35 | 0.36 |
| B4 PVT, 3x | between 2x (0.3) and 4x (0.7): 0.3 + 0.5 x 0.4 | 0.50 |
| B5 RMSSD, -35% | between 20 (0.2) and 40 (0.5): 0.2 + 0.75 x 0.3 | 0.43 |
| B6 activity, -35% | between -30 (0.3) and -60 (0.7): 0.3 + (5/30) x 0.4 | 0.37 |

max = 0.56, mean = 0.442. s_beh = 0.6 x 0.56 + 0.4 x 0.442 = 0.336 + 0.177 = **0.51**.

This component example shows the mechanics only and gives s_beh = 0.51. Scenario B below uses s_beh = 0.65, a more deteriorated state than this example's inputs.

## Scenario A: nominal crew member, mission day 60 of 180

| Domain | s | p |
|---|---|---|
| cardiac | 0.05 | 0 (dead zone) |
| behavioral | 0.12 | 0.017 |
| neuro | 0.03 | 0 |
| radiation | 0.15 | 0.8 x 0.0417 = 0.033 |
| bone | 0.10 | 0 |

- L = 0.0168 + 0.0339 = 0.0507; R_raw = 0.0494; **R = 0.005 + 0.99 x 0.0494 = 0.054** (band: low).
- No flag (largest s is 0.15).
- DTR_R = 150 x (0.946)^1.5 = 138.0 days.
- Trend flat (m = 1). T_rad: allowance remaining 300 mSv at 0.7 mSv/day = (300 - 25) / 0.7 = 393 days.
- DTR = min(138.0, 393) = 138.0, displayed **">100 days"**.

## Scenario B: behavioral deterioration with sleep loss, moderate radiation

| Domain | s | p | -ln(1 - p) |
|---|---|---|---|
| cardiac | 0.20 | 0.083 | 0.087 |
| behavioral | 0.65 | 0.458 | 0.613 |
| neuro | 0.25 | 0.6 x 0.125 = 0.075 | 0.078 |
| radiation | 0.40 | 0.8 x 0.25 = 0.200 | 0.223 |
| bone | 0.25 | 0.075 | 0.078 |

- L = 1.079; R_raw = 1 - exp(-1.079) = 0.660; **R = 0.005 + 0.99 x 0.660 = 0.659** (band: high).
- Flag is set (behavioral 0.65 and radiation 0.40 are at or above 0.30); the 0.60 floor is below 0.659, so it changes nothing here.
- DTR_R = 150 x (0.341)^1.5 = 29.9 days.
- Assume R rose 0.12 over 7 days: m = exp(-0.6) = 0.549, giving 16.4 days.
- T_rad: remaining 200 mSv at 1.8 mSv/day = 97 days (does not bind).
- Assume an R interval of 0.55 to 0.75 from the Monte Carlo draws: DTR_high = 150 x (0.45)^1.5 x 0.549 = 24.9; DTR_low = 150 x (0.25)^1.5 x 0.549 = 10.3.
- Output: **R = 0.66 (high); DTR about 16 days (10-25)**. Action: flight surgeon review within 4 h and start return planning. (Under the earlier logistic aggregation this scenario was "elevated" with R = 0.56; the noisy-OR is more sensitive to a single deteriorating domain by design.)

## Scenario C: acute multi-domain event

Seizure detected (confidence 0.95), solar particle event in progress, borderline hypoxemia.

| Domain | s | p | -ln(1 - p) |
|---|---|---|---|
| cardiac | 0.70 | 0.500 | 0.693 |
| behavioral | 0.30 | 0.167 | 0.182 |
| neuro | 1.00 | 0.6 x 0.75 = 0.450 | 0.598 |
| radiation | 0.85 | 0.8 x 0.625 = 0.500 | 0.693 |
| bone | 0.15 | 0.6 x 0.0417 = 0.025 | 0.025 |

- L = 2.192; R_raw = 0.888; **R = 0.005 + 0.99 x 0.888 = 0.884** (band: critical).
- Override O4 (seizure) is active; its 0.80 floor is below the computed value, so R stays 0.884 and the override reason is displayed.
- DTR_R = 150 x (0.116)^1.5 = 5.9 days. Assume R rose 0.40 in 7 days: m = exp(-2) = 0.135, giving 0.80 day. T_acute = 0.5 day.
- DTR_calc = min(0.80, 0.5) = 0.5 day. Assume low Earth orbit with L = 0.5 day: DTR_calc equals L, so the return window is met only with no margin; displayed DTR = 0.5 day, shown as **"<5 days"**.
- Escalation: immediate clinician contact and immediate crew and ground alert.

## Mechanics checks (equal influence, c = 1)

| Case | p | R_agg | Displayed R | DTR basis |
|---|---|---|---|---|
| one domain at s = 0.5, rest at or below 0.10 | 0.333 | 0.335 | 0.60 (flag) | R_agg 0.335, DTR_R 81.3 days |
| two domains at s = 0.8 | 0.583 each | 0.823 | 0.823 (critical) | DTR_R 11.2 days |
| five domains at s = 0.2 with the c_d above | 0.083, 0.083, 0.050, 0.067, 0.050 | 0.294 | 0.294 (guarded), no flag | DTR_R 88.9 days |

## Single-domain sensitivity

Baseline: all domains at s = 0.10, so every p = 0, R_agg = 0.005.

Setting one domain to 1.0, others unchanged:

| Domain set to 1.0 | p | R_agg | Displayed R | DTR_R from R_agg (days) |
|---|---|---|---|---|
| cardiac (c 1.0) | 0.75 | 0.747 | 0.747 (high) | 19.0 |
| behavioral (c 1.0) | 0.75 | 0.747 | 0.747 (high) | 19.0 |
| radiation (c 0.8) | 0.60 | 0.599 | 0.60 (high, flag) | 38.1 |
| neuro (c 0.6) | 0.45 | 0.451 | 0.60 (high, flag) | 61.1 |
| bone (c 0.6) | 0.45 | 0.451 | 0.60 (high, flag) | 61.1 |

Two domains at 1.0 (cardiac and behavioral): L = 2.773, R_agg = 0.933 (critical), DTR_R = 2.6 days.

Interpretation: a single failed domain is a "high" alert through the flag (return planning starts), and two failed domains are "critical". Because DTR uses R_agg, a single failed lower-influence domain (neuro, bone) gives a long DTR of about 61 days while the alert is still raised. Any domain can be overridden to critical only through the hard-rule list.
