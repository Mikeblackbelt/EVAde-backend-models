# Calibration and Validation Plan

The numbers in this plan are placeholders until fitted. This file says how they would be fitted, what success looks like, and what could go wrong. Nothing here has been run.

## 1. Defining the label

R is trained to match an adjudicated label: **"a clinician panel would recommend intervention or return planning within the next 14 days."**

- Adjudication: two independent clinicians review each case window (summary of data, not raw streams); a third resolves disagreement.
- Record inter-rater agreement (Cohen's kappa) and report it. Target kappa of at least 0.6; if lower, the label is too noisy to calibrate against, and the label definition is revised first.

## 2. Phases

| Phase | What | Output |
|---|---|---|
| 0. Elicitation | structured expert rounds (Delphi, at least 8 experts across flight medicine, behavioral health, cardiology, neurology, radiation, bone; 2 rounds) to set influence factors c_d, anchors, dead zone and cap, flag threshold, band edges, and override triggers | initial parameter set; Kendall's W of at least 0.6 on weight ranking |
| 1. Simulation | synthetic cohort of 2,000 crew-missions with generated domain trajectories (healthy, slow decline, acute events, sensor failure) | check band distribution and alert rates; find unreachable states |
| 2. Terrestrial pilot | at least 30 participants for at least 8 weeks in an analog or lab-confinement setting, with reference sensors and weekly questionnaires | calibration data and labels |
| 3. Refit | fit c_d (non-negative, at most 1), dead zone, and flag threshold by maximizing held-out likelihood of the 14-day label, with L2 penalty and subject-wise cross-validation; set a per-domain false-alarm target of about 0.2 high-or-above alerts per person-month so the five domains together meet the overall 1 per person-month | compare against the fixed expert values |
| 4. External check | a second site or cohort that was not used in any fitting step | generalization result, including failures |

Adoption rule for refit parameters: use them only if held-out log-loss improves by at least 5% over the expert values under subject-wise cross-validation; otherwise keep the expert values.

## 3. Targets (design goals)

| Quantity | Target |
|---|---|
| Discrimination for the 14-day label | AUROC at least 0.80 |
| Calibration | slope between 0.8 and 1.2; calibration reported by band |
| Skill | Brier skill score of at least 0.20 against a prevalence-only baseline |
| False alerts, healthy cohort | at most 1 per person-month at "high" or above; at most 0.1 per person-month at "critical" |
| Missed events | sensitivity of at least 0.90 for override-class events at their trigger thresholds |
| Stability to parameter changes | influence factors varied by plus or minus 50% each and anchors by plus or minus 20%: Spearman correlation of R at least 0.9 and no more than 10% of case windows change band |
| Reliability | R changes by no more than 0.05 on re-computation with identical inputs and different random seeds |

## 4. DTR validation

There is no ground-truth data for "correct" return time, so validation is judgment-based first:

- 30 clinical vignettes, each with a computed DTR; a flight-medicine panel gives its own recommended return time independently. Target: agreement within a factor of 2 on at least 80% of vignettes, and no vignette where the panel recommends return sooner than the displayed lower bound.
- Hard-limit checks: unit tests of T_rad and T_bone on synthetic dose and bone-loss trajectories, verifying the margin (25 mSv) and the no-limit condition for slow bone loss.
- Back-test on any retrospective data obtainable; document the lack of data otherwise.

## 5. Sensitivity and robustness analyses

- **One-at-a-time:** vary each weight, anchor, and exponent; plot R and DTR change.
- **Missing data:** remove 30% of days at random and in blocks (sensor failure); verify the low-confidence flag and the 0.30 floor behave as intended.
- **Artifacts:** inject motion, noise, and dose-rate spikes; verify the quality gates suppress false escalation.
- **Subgroups:** report discrimination and calibration by sex, age group, and skin tone (for PPG-derived inputs).
- **Negative controls:** shuffled labels should give AUROC near 0.5.
- **Anchor ablation:** drop each domain in turn; confirm no domain is carrying the result alone unintentionally.

## 6. Failure modes and mitigations

| Failure | Consequence | Mitigation |
|---|---|---|
| Sensor failure across several domains | false low score | Q rules: low-confidence flag and the 0.30 floor; alert on low overall Q |
| Person games the questionnaire or removes the sensor | blind spot | multiple independent domains; compliance tracked as a feature |
| Skin-tone bias in SpO2 or PPG | systematic under- or overestimation | widened SpO2 uncertainty, confirmation by a second sensor for overrides, subgroup reporting |
| Subject leakage during tuning | inflated performance claims | subject-wise splits only |
| Radiation detector fault | false event or missed event | second detector cross-check; SEU-rate trust rule |
| Alert fatigue | clinicians ignore alerts | alert-rate targets; hysteresis; band-specific actions |
| Overfitting to the pilot cohort | poor transfer | external check; stability analysis |
| Mission-phase change (new L, new dose rate) | wrong DTR | L and dose rate are explicit inputs, logged with each output |

## 7. Governance

- **Human in the loop.** The system produces decision support; a clinician decides. No automatic action.
- **Logging and audit.** Every R and DTR is stored with its inputs, parameter version, and override reasons.
- **Change control.** Any parameter change is versioned and re-tested against the stability and simulation suites before use.
- **Privacy.** Treat all inputs as identifiable health data, especially in small crews. Collect the minimum features required.
- **Regulatory and ethical review.** The regulatory status of automated health-risk scoring and of mental-health monitoring was not researched in this plan. Obtain qualified advice and institutional review (IRB or equivalent) before any use with people.
- **Disclosure.** Crew members see their own scores and the reasons for them; the system never produces a score they cannot inspect.

## 8. Open questions

- Which clinical label, exactly, best defines "needs intervention" for each domain, and should R be one number or a vector?
- Can a single scale combine radiation exposure (long-term risk) with acute events (hours)? A two-score design (acute and chronic) may be better; the single-score version here is for simplicity.
- How should mission-phase context (launch, EVA, return window) change the weights?
- What is the right treatment of correlated domains (sleep loss drives both behavioral and cardiac signals)?
- What is the best way to present uncertainty to a flight surgeon without inducing false precision?
