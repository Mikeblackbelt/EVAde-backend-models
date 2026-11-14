# Bone Health Metrics

## Summary

Bone health is measured by:

- **Areal BMD by DXA**: the clinical standard. Not portable.
- **Volumetric BMD by QCT** and **HR-pQCT**: separate cortical and trabecular compartments. Used in research, including spaceflight studies.
- **FRAX**: a clinical risk tool that combines BMD (optional) with risk factors to give 10-year fracture probability.
- **Portable proxies** (heel quantitative ultrasound, gait or loading indices): not validated for spaceflight. Research needed.

In spaceflight, bone loss is fast at weight-bearing sites and is not fully recovered after return. The exact rate depends on the measure used, and the sources disagree (see below). For Blindsight, the usable output is a longitudinal trend and an uncertainty interval, not a fracture-risk diagnosis.

## Sources

- **Kanis, J. A., Johnell, O., Oden, A., Johansson, H., & McCloskey, E. (2008).** FRAX and the assessment of fracture probability in men and women from the UK. *Osteoporosis International*, 19(4), 385-397. DOI 10.1007/s00198-007-0543-5. Verified. Describes FRAX calibrated to UK data, with and without femoral neck BMD. Models now exist for many countries, per a 2018 review by the same group.
- **Lang, T., et al. (2004).** Cortical and trabecular bone mineral loss from the spine and hip in long-duration spaceflight. *Journal of Bone and Mineral Research*, 19(6), 1006-1012. Verified: citation. Study used QCT on 14 ISS crew members on 4-6 month missions. Sample is small, and the mission duration is shorter than a Mars mission.
  - **UNRESOLVED**: the sources disagree on the loss rates. The PubMed abstract gives hip trabecular vBMD losses of roughly 2.2-2.7%/month and areal figures for spine and hip. A press account gives cortical loss of 1.6-1.7%/month, while the PubMed summary gives cortical vBMD losses of about 0.4-0.5%/month. These probably reflect different measures. Read the full paper before quoting any single rate.
- **Black, D. M., & Rosen, C. J. (2016).** Postmenopausal osteoporosis. *New England Journal of Medicine*, 374(3), 254-262. DOI 10.1056/NEJMcp1513724. Verified. Clinical review covering treatment thresholds (T-score of -2.5 or lower, prior fracture, or elevated FRAX). Use for the clinical context of Blindsight's output, which should not be framed as a diagnostic threshold.
- **NASA bone loss reviews (ntrs.nasa.gov).** Search "bone loss spaceflight" for the current NASA Human Research Program summaries. These are the best source on countermeasures and recovery. Not checked by item.

## Implementation steps

1. **Define the target.** A bone-loss trend indicator per person, plus a flag when the trend exceeds a threshold agreed with clinicians. Write the threshold and its source in the config.
2. **Get calibration data.** Use DXA or QCT data from public or collaborator sources (see data sources). The on-device metric must be validated against DXA BMD on the same people.
3. **Screen portable proxies.** Evaluate heel QUS and gait-derived loading metrics for correlation with DXA. Report the correlation with confidence intervals, and do not assume a linear relation.
4. **Longitudinal model.** Per-person baseline, rate of change over time, and an uncertainty interval. Avoid single-point classification, which would ignore that bone changes slowly and measurements are noisy.
5. **Include mission context.** Duration, exercise and countermeasure use, and radiation dose (from the radiation module) all change bone loss. Record them as covariates.
6. **Scope statement.** The output is for research and monitoring, not diagnosis. Put this in the schema and the UI.
