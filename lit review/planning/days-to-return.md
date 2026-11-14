# Days to Return (DTR)

DTR is the recommended latest time to return to Earth or to definitive care. It is the minimum of a risk-driven time and several hard-limit times, adjusted by the risk trend, floored by logistics, and displayed as "<5", a number from 5 to 100, or ">100".

## 1. Base curve from risk

**DTR_R = 150 x (1 - R)^1.5 days**

| R | DTR_R (days) | Display |
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

A falling R does not lengthen the DTR beyond the base curve. A rising R shortens it.

## 3. Hard-limit times

| Limit | Formula | Notes |
|---|---|---|
| **T_rad** | (A_rem - 25) / d_rate | A_rem = remaining career allowance in mSv (600 minus dose already taken). d_rate = 7-day mean dose-equivalent rate in mSv/day. 25 mSv is the safety margin. |
| **T_bone** | (10 - L) / r | L = current estimated areal BMD loss in percent; r = trailing 90-day slope in percent per day. No limit if r is at most 0.01 percent/day (about 0.3 percent/month). |
| **T_acute** | 0.5 day | applies only while an override O2, O3, O4, or O5 is active |

Example for T_rad: remaining allowance 200 mSv at the deep-space cruise rate of 1.8 mSv/day gives (200 - 25) / 1.8 = 97 days. At the Mars-surface rate of 0.67 mSv/day the same allowance gives 261 days. (Rates from Hassler 2014; the 600 mSv limit from the radiation research notes, to be re-verified.)

Example for T_bone: current loss 4% and a trailing slope of 1 percent per month (0.033 percent/day) gives (10 - 4) / 0.033 = 180 days.

## 4. Combined DTR

1. DTR_calc = min( DTR_R x m_trend, T_rad, T_bone, T_acute ).
2. **Logistics floor L:** L is the earliest feasible time to deliver the person to definitive care, set by mission architecture. Illustrative assumptions only: low Earth orbit 0.5 day; lunar distance 6 days; Mars transit 200 days.
3. If DTR_calc is less than L, the system outputs the status **"return window cannot be met"** and recommends treating in place and requesting support. The displayed DTR is L, not a number below it.
4. Otherwise DTR_eff = DTR_calc.

## 5. Display and stability

- If DTR_eff is below 5: display **"<5 days"** (and show the exact value, to the nearest 0.5 day, in the detail view).
- If DTR_eff is above 100: display **">100 days"**.
- Otherwise: nearest 0.5 day below 10 days, nearest 1 day above 10 days.
- **Rate limiter:** DTR may drop immediately when risk rises, but may rise by at most 20% per day when risk falls.
- Show the driver: which of DTR_R, T_rad, T_bone, T_acute set the value.

## 6. Uncertainty interval

DTR_R is monotonic in R, so the interval comes from the R interval:

- DTR_low = 150 x (1 - R_high)^1.5 x m_trend
- DTR_high = 150 x (1 - R_low)^1.5 x m_trend

where R_low and R_high are the 10th and 90th percentiles from the Monte Carlo draws. Apply the hard limits to both ends, then display the central value with the range, for example "24 days (16-34)".

## 7. Edge cases

| Case | Behavior |
|---|---|
| Data quality Q below 0.6 | mark DTR "provisional"; no value above 100 is shown as certain |
| Override active, R floored to 0.80 | DTR_R = 13.4 days, then the override rule and T_acute apply |
| Two severe domains (R above 0.9) | DTR_R below 5 days |
| R falls steadily | DTR rises through the rate limiter |
| Missing logistics input L | refuse to display DTR; show R only |
