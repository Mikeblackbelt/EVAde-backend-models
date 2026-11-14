# Worked Examples

All inputs are invented for illustration. Formulas are in [risk-score.md](risk-score.md) and [days-to-return.md](days-to-return.md). Calculations were done by hand and rounded.

Recap: S = sum of w x s; M = max s; X = 0.6 S + 0.4 M; R = 1 / (1 + exp(-8 (X - 0.45))); DTR_R = 150 (1 - R)^1.5.
Weights: cardiac 0.25, behavioral 0.25, neuro 0.15, radiation 0.20, bone 0.15.

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

| Domain | s |
|---|---|
| cardiac | 0.05 |
| behavioral | 0.12 |
| neuro | 0.03 |
| radiation | 0.15 |
| bone | 0.10 |

- S = 0.0125 + 0.030 + 0.0045 + 0.030 + 0.015 = 0.092
- M = 0.15
- X = 0.6 x 0.092 + 0.4 x 0.15 = 0.115
- R = 1 / (1 + exp(-8 x (0.115 - 0.45))) = 1 / (1 + exp(2.68)) = **0.064** (band: low)
- DTR_R = 150 x (0.936)^1.5 = 135.8 days
- Trend flat (m = 1). T_rad: allowance remaining 300 mSv at 0.7 mSv/day = (300 - 25) / 0.7 = 393 days.
- DTR = min(135.8, 393) = 135.8, displayed **">100 days"**.

## Scenario B: behavioral deterioration with sleep loss, moderate radiation

| Domain | s |
|---|---|
| cardiac | 0.20 |
| behavioral | 0.65 |
| neuro | 0.25 |
| radiation | 0.40 |
| bone | 0.25 |

- S = 0.050 + 0.1625 + 0.0375 + 0.080 + 0.0375 = 0.3675
- M = 0.65
- X = 0.6 x 0.3675 + 0.4 x 0.65 = 0.4805
- R = 1 / (1 + exp(-8 x 0.0305)) = 1 / (1 + exp(-0.244)) = **0.561** (band: elevated)
- DTR_R = 150 x (0.439)^1.5 = 43.7 days
- Assume R rose 0.12 over 7 days: m = exp(-0.6) = 0.549, giving 24.0 days.
- T_rad: remaining 200 mSv at 1.8 mSv/day = 97 days (does not bind).
- With R interval 0.45 to 0.67: DTR_high = 150 x (0.55)^1.5 x 0.549 = 33.6; DTR_low = 150 x (0.33)^1.5 x 0.549 = 15.6.
- Output: **R = 0.56 (elevated); DTR about 24 days (16-34)**. Action: flight surgeon review within 24 h and a countermeasure plan; return planning prepared in case R crosses 0.6.

## Scenario C: acute multi-domain event

Seizure detected (confidence 0.95), solar particle event in progress, borderline hypoxemia.

| Domain | s |
|---|---|
| cardiac | 0.70 |
| behavioral | 0.30 |
| neuro | 1.00 |
| radiation | 0.85 |
| bone | 0.15 |

- S = 0.175 + 0.075 + 0.150 + 0.170 + 0.0225 = 0.5925
- M = 1.00
- X = 0.6 x 0.5925 + 0.4 x 1.0 = 0.7555
- R = 1 / (1 + exp(-8 x 0.3055)) = 1 / (1 + exp(-2.444)) = **0.920** (band: critical)
- Override O4 (seizure) is active; floor 0.80 is below the computed value, so R stays 0.92 and the override reason is displayed.
- DTR_R = 150 x (0.080)^1.5 = 3.4 days. Assume R rose 0.40 in 7 days: m = exp(-2) = 0.135, giving 0.46 days. T_acute = 0.5 day.
- DTR_calc = min(0.46, 0.5) = 0.46 day. Assume low Earth orbit with L = 0.5 day: DTR_calc is below L, so status: **"return window cannot be met"; treat in place and request support**, displayed DTR = 0.5 day, shown as "<5 days".
- Escalation: immediate clinician contact and immediate crew and ground alert.

## Single-domain sensitivity

Baseline: all domains at s = 0.10, so S = M = X = 0.10 and R = 0.057.

Setting one domain to 1.0, others unchanged:

| Domain set to 1.0 | S | X | R | Band | DTR_R (days) |
|---|---|---|---|---|---|
| cardiac (w 0.25) | 0.325 | 0.595 | 0.761 | high | 17.5 |
| behavioral (w 0.25) | 0.325 | 0.595 | 0.761 | high | 17.5 |
| radiation (w 0.20) | 0.280 | 0.568 | 0.720 | high | 22.2 |
| neuro (w 0.15) | 0.235 | 0.541 | 0.674 | high | 27.9 |
| bone (w 0.15) | 0.235 | 0.541 | 0.674 | high | 27.9 |

Two domains at 1.0 (cardiac and behavioral): S = 0.55, X = 0.73, R = 0.904, which is critical and gives DTR_R = 4.5 days.

Interpretation: the design makes a single failed domain a "high" alert (return planning starts), and two failed domains "critical". Any domain can be overridden to critical only through the hard-rule list.
