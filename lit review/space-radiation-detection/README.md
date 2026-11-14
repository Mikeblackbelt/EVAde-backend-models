# Space Radiation Detection

## Summary
Galactic cosmic rays (GCR) and solar energetic particles (SEP) threaten crew health (cancer and CNS risk) and spacecraft electronics (single-event upsets, SEUs). Detection is done with dosimeters (TLD, OSLD, active silicon detectors), particle telescopes, and hardware-level SEU monitors. For Blindsight the relevant question is whether a body-worn or cabin sensor can flag dose or SEU events that could corrupt sensor data, so downstream health models can discount or flag affected samples.

## Sources
- Cucinotta, F. A., & Durante, M. (2006). Cancer risk from exposure to galactic cosmic rays: implications for space exploration by human beings. *The Lancet Oncology*, 7(5), 431-435.
- Dodd, P. E., & Massengill, L. W. (2003). Basic mechanisms and modeling of single-event upset in digital microelectronics. *IEEE Transactions on Nuclear Science*, 50(3), 583-602.
- Hassler, D. M., et al. (2014). Mars' surface radiation environment measured with the Mars Science Laboratory's Curiosity rover. *Science*, 343(6169). (RAD instrument; benchmark for active particle detection.)
- NASA Human Research Program, Space Radiation Element reports (ntrs.nasa.gov). Current risk models and dosimetry standards.

## Implementation steps
1. Define the target measurement: absorbed dose (Gy) versus event flags (SEU or particle hit). Blindsight likely needs event flags plus a coarse dose rate.
2. Pick a sensor class: start with a commercial silicon PIN diode or SRAM-based SEU monitor for event counts; treat TLD/OSLD as offline calibration references.
3. Build a calibration plan against a known lab source (e.g., Sr-90 or Cs-137) to establish a counts-to-dose mapping before any flight-like claims.
4. Specify the data interface: timestamped event log, dead time, and a health flag written alongside every physiological sample.
5. Write a data-quality rule: samples within N seconds of a detected event are marked suspect in the backend schema.
6. Document limits: a ground-level prototype cannot reproduce GCR heavy-ion spectra; validation must use published flight data or beam-test facilities.
