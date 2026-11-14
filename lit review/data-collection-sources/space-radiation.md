# Space Radiation Data Sources

## Mars Science Laboratory RAD (Curiosity rover)
- **What:** energetic particle and dose measurements on the Martian surface, the source of the GCR and SEP dose figures in [space-radiation-detection](../space-radiation-detection/README.md). Hassler et al. (2014).
- **Access:** NASA Planetary Data System (PDS) is the standard archive for MSL data. Confirm the RAD dataset location on pds.nasa.gov before use.
- **Use:** an external benchmark for the dose-rate estimation and an example of real calibrated particle data.
- **Limit:** Mars surface, not in-cabin. Different shielding and particle environment.

## NASA Space Radiation Analysis and Human Research Program reports
- **What:** risk models and dosimetry summaries from NASA's Human Research Program, available through ntrs.nasa.gov.
- **Access:** public NASA technical reports.
- **Use:** background for the risk model and the requirements for dose estimation. Not raw data.

## CRaTER (Lunar Reconnaissance Orbiter)
- **What:** a lunar radiation dosimeter.
- **Access:** data are in the NASA PDS (PPI node, lro.crater PDS4 bundle; raw EDR data in the PDS3 archive). data.gov catalog entries list "restricted public" access with no license. Check PDS terms before any redistribution.

## Ground-based beam and source calibration
- **What:** lab sources (Sr-90, Cs-137) are the practical calibration tool for the prototype. Beam facilities for heavy-ion testing would be needed for SEU cross-section validation.
- **Access:** lab-owned sources need institutional radiation-safety approval. Beam facilities need proposals. Neither is open data.
- **Use:** the calibration plan in [space-radiation-detection](../space-radiation-detection/README.md), step 3.

## Notes
- Radiation data is often stored as counts or dose rates with mission-specific calibration. Record the calibration version with every dataset.
- Do not assume that a dosimeter reading from one location applies elsewhere in a spacecraft.
