# Space Radiation Detection

## Summary

Galactic cosmic rays (GCR) and solar energetic particles (SEP) create two problems for Blindsight. First, crew exposure raises long-term cancer and central nervous system risk. Second, single-event upsets (SEUs) in electronics can corrupt sensor data or flip memory. Detection uses three tools: passive dosimeters (TLD, OSLD) that are read after flight, active particle detectors (silicon telescopes, PIN diodes), and SEU monitors built into memory or logic.

For Blindsight, the practical question is narrower: can a body-worn or cabin sensor flag events that could corrupt physiological data? The answer is probably yes for SEU and particle-hit flags. Dose-equivalent estimation for health risk is a separate problem that needs a physics model and is out of scope for the first prototype.

Key numbers from the Mars Science Laboratory RAD measurements (Hassler et al., 2014): GCR dose equivalent on the Martian surface averaged about 0.67 mSv/day, versus about 1.8 mSv/day during the cruise. These figures come from press and abstract excerpts; confirm against the paper before quoting.

## Sources

- **Cucinotta, F. A., & Durante, M. (2006).** Cancer risk from exposure to galactic cosmic rays: implications for space exploration by human beings. *The Lancet Oncology*, 7(5), 431-435. DOI not confirmed; look it up before citing. Verified: authors, journal, volume, pages. A review article; it sets out why GCR heavy ions carry cancer risk that low-LET models underestimate. Useful as the framing reference.
- **Dodd, P. E., & Massengill, L. W. (2003).** Basic mechanisms and modeling of single-event upset in digital microelectronics. *IEEE Transactions on Nuclear Science*, 50(3), 583-602. DOI 10.1109/TNS.2003.813129. Verified. A review of SEU mechanisms in DRAM, SRAM, and logic, plus mitigation and simulation methods. Use this for the SEU failure model.
- **Hassler, D. M., et al. (2014).** Mars' surface radiation environment measured with the Mars Science Laboratory's Curiosity rover. *Science*, 343(6169), 1244797. DOI 10.1126/science.1244797. Verified: citation (published online 9 Dec 2013; print 24 Jan 2014). Source of the surface GCR and SEP dose measurements listed above. The 0.67 and 1.8 mSv/day figures were confirmed in the SwRI press release (Dec 2013); the paper itself was not accessible.
- **NASA Human Research Program, Space Radiation Element.** Current risk models and dosimetry reports, available via ntrs.nasa.gov. Search "space radiation element risk" for the current versions. Not checked by item; this is a pointer, not a citation.

## Implementation steps

1. **Define the output.** Use two outputs: a binary or count SEU/particle-hit event flag, and a coarse dose rate. Do not attempt dose-equivalent risk estimation in the prototype.
2. **Pick the first sensor.** Start with a commercial silicon PIN diode with a charge-sensitive preamp for event counts. A memory-based SEU monitor is a cheaper second option, but it reports upsets only indirectly. Treat TLD/OSLD as offline calibration references.
3. **Calibrate on a known source.** Use a lab Sr-90 or Cs-137 beta/gamma source to establish counts-to-dose mapping and dead time. This validates the electronics only; it does not reproduce GCR heavy-ion response. Record the calibration uncertainty.
4. **Specify the interface.** Each event gets a timestamp, a channel ID, and an energy bin if the detector supports spectroscopy. Each physiological sample gets a radiation-health flag.
5. **Write a data-quality rule.** Samples within a fixed window (start at 5 s, tune later) of a detected event are marked suspect. Keep the raw sample; do not drop it silently.
6. **Validate against public data.** Use published flight or ground-test datasets to check the event-flag logic, if access is available (see [data-collection-sources](../data-collection-sources/space-radiation.md)).
7. **Document the limits.** A lab-calibrated prototype cannot reproduce the GCR spectrum or solar particle events. State this in the README and in any output schema.

## Evaluation and robustness

- **Metrics:** detection efficiency against source activity; dead-time correction accuracy; false-flag rate with the source removed (background run); fraction of physiological data flagged by the suppression rule.
- **Robustness checks:** temperature dependence of detector gain; coupling of power-supply noise into counts; agreement between two independent detectors; behavior under high count rates.
- **Reporting:** state calibration source, geometry, and uncertainty with every dose number.

## Limitations and open questions

- A lab beta or gamma source does not reproduce heavy-ion energy deposition. SEU cross-sections need beam testing, which is outside open data.
- Dose varies with shielding and location in a vehicle; one reading does not apply everywhere.
- Open: does a body-worn detector add anything beyond a cabin dosimeter for data-quality flagging? What flag window length is justified by SEU physics rather than convenience?
- Link to health outcomes: radiation also bears on CNS and behavioral risk; see [behavioral-health](../behavioral-health/README.md). No validated quantitative CNS risk model was found.
