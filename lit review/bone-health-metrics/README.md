# Bone Health Metrics

## Summary
Bone health is assessed with bone mineral density (BMD) by DXA, volumetric QCT, and increasingly finite element (FE) strength estimates from CT. Fracture risk tools such as FRAX combine BMD with clinical risk factors. In spaceflight, astronauts lose bone mass quickly (on the order of 1% per month at weight-bearing sites), and recovery after long missions is incomplete. For Blindsight the useful output is a trend metric and risk flag, not a diagnosis. DXA is not portable, so an accessible proxy (e.g., heel quantitative ultrasound or a load/gait-derived index) would be needed.

## Sources
- Kanis, J. A., et al. (2008). FRAX and the assessment of fracture probability in men and women from the UK. *Osteoporosis International*, 19(4), 385-397.
- Lang, T., et al. (2004). Cortical and trabecular bone mineral loss from the spine and hip in long-duration spaceflight. *Journal of Bone and Mineral Research*, 19(6), 1006-1012.
- NASA bone loss and countermeasures reviews (ntrs.nasa.gov). Search "bone loss spaceflight review".
- Black, D. M., & Rosen, C. J. (2016). Postmenopausal osteoporosis. *New England Journal of Medicine*, 374, 254-262.

## Implementation steps
1. Define the target: a bone-loss trend indicator over time, plus a flag when the trend exceeds a threshold agreed with clinicians.
2. Use DXA/QCT data from public or collaborator sources only for calibration; any on-device metric must be validated against them.
3. Evaluate candidate portable proxies (heel quantitative ultrasound, gait-derived loading metrics) for correlation with DXA BMD.
4. Build a longitudinal model: per-person baseline, rate of change, and an uncertainty interval; avoid single-point classification.
5. Pair with mission context (duration and radiation dose), since bone loss depends on both.
6. State clinical scope in the README: the output is for research and monitoring, not fracture diagnosis.
