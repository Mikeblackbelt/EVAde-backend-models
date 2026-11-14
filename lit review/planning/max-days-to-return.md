# Maximum Days to Return (MDTR)

MDTR is the **maximum** number of days the person should remain before returning to Earth or to definitive care: an upper bound (a deadline), not an expected time. It is the minimum of a risk-driven time and several hard-limit times, adjusted by the risk trend, floored by logistics, and displayed as "<5", a number from 5 to 100, or ">100".

## 1. Base curve from risk

**MDTR_R = 150 x (1 - R_mdtr)^1.5 days**, where R_mdtr is the aggregated risk before the abnormal-component flag floor (hard-rule overrides still floor it at 0.80; see [risk-score.md](risk-score.md) section 3). The table below is in terms of R_mdtr.

| R | MDTR_R (days) | Display |
|---|---|---|
| 0.05 | 139 | >100 |
| 0.10 | 128 | >100 |
| 0.20 | 107 | >100 |
| 0.237 | 100 | boundary |
| 0.30 | 88 | 88 |
| 0.40 | 70 | 70 |
| 0.50 | 53 | 53 |
| 0.60 | 38 | 38 |
| 0.70 | 25 | 25 |
| 0.80 | 13.4 | 13 |
| 0.896 | 5.0 | boundary |
| 0.90 | 4.7 | <5 |
| 0.95 | 1.7 | <5 |

Reasoning: the curve falls slowly at low risk and steeply near critical risk. It crosses 100 days at R = 0.237 (just inside the "guarded" band) and 5 days at R = 0.896. The exponent 1.5 and the 150-day scale are design assumptions, not fitted values.

## 2. Trend adjustment

Let dR7 = R now minus R seven days ago.

**m_trend = exp(-5 x max(0, dR7))**

| dR7 | m_trend |
|---|---|
| 0 or falling | 1.00 |
| +0.10 | 0.61 |
| +0.20 | 0.37 |
| +0.30 | 0.22 |
| +0.40 | 0.14 |

A falling R does not lengthen the MDTR beyond the base curve. A rising R shortens it.

## 3. Hard-limit times

| Limit | Formula | Notes |
|---|---|---|
| **T_rad** | (A_rem - 25) / d_rate | A_rem = remaining career allowance in mSv (600 minus dose already taken). d_rate = 7-day mean dose-equivalent rate in mSv/day. 25 mSv is the safety margin. |
| **T_bone** | (10 - L) / r | L = current estimated areal BMD loss in percent; r = trailing 90-day slope in percent per day. No limit if r is at most 0.01 percent/day (about 0.3 percent/month). |
| **T_acute** | 0.5 day | applies only while an override O2, O3, O4, or O5 is active |
| **T_extreme** | min over eligible domains of (9 - 35 x (s_d - 0.80)) days, for s_d at or above 0.80 | 9 days at s_d = 0.80, falling linearly to 2 days at s_d = 1.0. Guarantees MDTR below 10 days whenever any domain is extremely high. Eligible domains: Q_d at least 0.5 and not a shadow-mode input (EEG affective markers, unvalidated wrist SpO2). No limit if no eligible domain reaches 0.80. |

Example for T_rad: remaining allowance 200 mSv at the deep-space cruise rate of 1.8 mSv/day gives (200 - 25) / 1.8 = 97 days. At the Mars-surface rate of 0.67 mSv/day the same allowance gives 261 days. (Rates from Hassler 2014; the 600 mSv limit from the radiation research notes, to be re-verified.)

Example for T_extreme: neuro at s = 0.90 gives 9 - 35 x 0.10 = 5.5 days, even if the other domains are normal and R is moderate. A domain this high should be a measured event (for example a confirmed seizure burst or a very high dose), so the cap is intended to fire rarely; for bone, s_d near 1.0 would mean a measured fracture-level loss, since a trend proxy alone is capped below it.

Example for T_bone: current loss 4% and a trailing slope of 1 percent per month (0.033 percent/day) gives (10 - 4) / 0.033 = 180 days.

## 4. Combined MDTR

1. MDTR_calc = min( MDTR_R x m_trend, T_rad, T_bone, T_acute, T_extreme ).
2. **Logistics floor L:** L is the earliest feasible time to deliver the person to definitive care, set by mission architecture. Illustrative assumptions only: low Earth orbit 0.5 day; lunar distance 6 days; Mars transit 200 days.
3. If MDTR_calc is less than L, the system outputs the status **"return window cannot be met"** and recommends treating in place and requesting support. The displayed MDTR is L, not a number below it.
4. Otherwise MDTR_eff = MDTR_calc.

## 5. Display and stability

- If MDTR_eff is below 5: display **"<5 days"** (and show the exact value, to the nearest 0.5 day, in the detail view).
- If MDTR_eff is above 100: display **">100 days"**.
- Otherwise: nearest 0.5 day below 10 days, nearest 1 day above 10 days.
- **Rate limiter:** MDTR may drop immediately when risk rises, but may rise by at most 20% per day when risk falls.
- Show the driver: which of MDTR_R, T_rad, T_bone, T_acute, T_extreme set the value.

## 6. Uncertainty interval

MDTR_R is monotonic in R, so the interval comes from the R interval:

- MDTR_low = 150 x (1 - R_high)^1.5 x m_trend
- MDTR_high = 150 x (1 - R_low)^1.5 x m_trend

where R_low and R_high are the 10th and 90th percentiles from the Monte Carlo draws. Apply the hard limits to both ends, then display the central value with the range, for example "24 days (16-34)".

## 7. Edge cases

| Case | Behavior |
|---|---|
| Data quality Q below 0.6 | mark MDTR "provisional"; no value above 100 is shown as certain |
| Override active, R_mdtr floored to 0.80 | MDTR_R = 13.4 days, then the override rule and T_acute apply |
| Two severe domains (R above 0.9) | MDTR_R below 5 days |
| Any one eligible domain at s_d of 0.80 or more | T_extreme caps MDTR below 10 days (2 days at s_d = 1.0); if L exceeds the cap the status is "return window cannot be met" |
| R falls steadily | MDTR rises through the rate limiter |
| Missing logistics input L | refuse to display MDTR; show R only |
