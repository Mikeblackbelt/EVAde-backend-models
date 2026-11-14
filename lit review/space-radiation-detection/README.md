# Space Radiation Detection

Status labels: **VERIFIED** = bibliographic record and abstract (or primary document) checked; **PARTIAL** = record checked, but abstract numbers or some fields were not confirmed; **UNVERIFIED** = not confirmed. Full texts of paywalled papers were not read.

## 1. Summary

Galactic cosmic rays (GCR) and solar energetic particle (SEP) events create two problems for Blindsight. Crew exposure raises long-term cancer, cardiovascular, and possibly central nervous system (CNS) risk. Separately, single-event effects (SEEs) in electronics can corrupt sensor data or latch up a device. Measurements show that dose rate varies several-fold with shielding location, vehicle orientation, orbit phase, and solar activity, which is the main argument for per-person or per-location sensing instead of one cabin number.

The recommended Blindsight claim is narrow: **estimate radiation context (orbit phase, SPE onset) and flag possible SEU-induced sensor faults.** Do not claim to infer individual dose-equivalent risk or CNS and behavioral effects from wearable signals. Human CNS evidence is limited, and quantitative CNS risk models are not available.

## 2. Measured radiation environments (anchor numbers)

| Environment | Measurement | Source |
|---|---|---|
| Mars transit | Dose equivalent for the shortest round trip with current propulsion and comparable shielding: 0.66 +/- 0.12 Sv | Zeitlin 2013 (abstract) |
| Lunar surface | Total absorbed dose rate in silicon 13.2 +/- 1 microGy/h; neutral-particle dose rate 3.1 +/- 0.5 microGy/h | Zhang 2020 (abstract) |
| Orion, Artemis I | Dose rates differed about fourfold between shielding locations during inner-belt passes; an attitude change roughly halved dose rates; GCR dose-equivalent rates up to 60% lower than previous observations | George 2024 (abstract, paraphrased) |
| ISS, Columbus lab | Up to 286 microGy/day absorbed and 647 microSv/day dose equivalent at end of May 2016 | Berger 2017 (search summary of abstract; PARTIAL) |
| ISS, 2016-2022 | Variation with SAA crossings, solar cycle, altitude, and shielding | Zeitlin 2023, ISS-RAD parts 1-3 |

Numbers come from abstracts or abstract summaries. Confirm against full text before quoting. The Hassler 2014 abstract as retrieved does not give numeric dose rates, so none are quoted for it.

## 3. Sources: measurements and dosimetry

- **Zeitlin, C., et al. (2013).** Measurements of energetic particle radiation in transit to Mars on the Mars Science Laboratory. *Science*, 340(6136), 1080-1084. DOI 10.1126/science.1235989. VERIFIED. 253-day cruise; single mission, near solar maximum, cruise-stage shielding rather than a crewed vehicle.
- **Hassler, D. M., et al. (2014).** Mars' surface radiation environment measured with the Mars Science Laboratory's Curiosity rover. *Science*, 343(6169), 1244797. DOI 10.1126/science.1244797. VERIFIED (record and abstract). Online 9 Dec 2013, print 24 Jan 2014. Reports absorbed dose and dose equivalent over about 300 sols of high solar activity. Surface measurement under an atmosphere, not equal to transit or lunar conditions.
- **Zeitlin, C., et al. (2019).** Measurements of radiation quality factor on Mars with the MSL Radiation Assessment Detector. *Life Sci Space Res*, 22, 89-97. DOI 10.1016/j.lssr.2019.07.010. VERIFIED (title, journal, volume, pages; abstract not read).
- **George, S. P., et al. (2024).** Space radiation measurements during the Artemis I lunar mission. *Nature*, 634(8032), 48-52. DOI 10.1038/s41586-024-07927-7. VERIFIED (abstract). Uncrewed, one mission. Instruments included Timepix-based HERA, EAD, M-42, CAD, and the MARE phantom. Dose-rate figures quoted by web summaries (for example in microGy/min and mission totals) were **not confirmed in the primary text** and are not used here.
- **Zhang, S., et al. (2020).** First measurements of the radiation dose on the lunar surface. *Science Advances*, 6(39), eaaz1334. DOI 10.1126/sciadv.aaz1334. VERIFIED. Chang'e-4 lander, lightly shielded instrument, solar minimum.
- **Guo, J., et al. (2023).** The first ground level enhancement seen on three planetary surfaces: Earth, Moon, and Mars. *Geophys Res Lett*, 50(15), e2023GL103069. DOI 10.1029/2023GL103069. PARTIAL (record from secondary sources; abstract not retrieved). First Solar Cycle 25 GLE (28 Oct 2021) seen at several locations. Treat as evidence that SEP events are the acute-risk driver, without quoting dose ratios.
- **Zeitlin, C., et al. (2023).** Results from the Radiation Assessment Detector on the ISS, Parts 1-3. *Life Sci Space Res*, 39, 67-75 (DOI 10.1016/j.lssr.2023.01.003), 76-85 (10.1016/j.lssr.2023.03.005), 86-94 (10.1016/j.lssr.2023.06.002). VERIFIED (abstracts). Six years of dose rates at fixed ISS locations; charged-particle to neutron dose ratio relatively stable across the station.
- **Berger, T., et al. (2017).** DOSIS and DOSIS 3D: radiation measurements with the DOSTEL instruments onboard the Columbus Laboratory of the ISS in the years 2009-2016. *J Space Weather Space Clim*, 7, A8. DOI 10.1051/swsc/2017005. PARTIAL (title, DOI, journal, year confirmed; volume, article number, and authors partly unconfirmed). **Berger et al. (2016)** *J Space Weather Space Clim*, 6, A39, DOI 10.1051/swsc/2016034: VERIFIED citation only.
- **Straube, U., Berger, T., & Dieckmann, M. (2024).** The ESA Active Dosimeter (EAD) system onboard the ISS. *Z Med Phys*, 34(1), 111-139. DOI 10.1016/j.zemedi.2023.03.001. VERIFIED. The closest published analogue of a wearable crew active dosimeter: dose, quality factor, and SAA segmentation, with in-flight cross-calibration.
- **Rios, R. R., et al. (2015).** Timepix-based radiation environment monitor measurements aboard the ISS. *Nucl Instrum Methods Phys Res A*, 782, 143-148. DOI 10.1016/j.nima.2015.02.016. PARTIAL (first author and fields not confirmed in an API record). Pixel-detector tracking can run on commodity hosts.
- **ICCHIBAN heavy-ion intercomparison** (Benton et al., *Radioisotopes*, 2019; DOI 10.3769/radioisotopes.68.411): PARTIAL. Some active instruments lacked sensitivity to high-LET (Fe) particles, which is why heavy-ion beam calibration matters. No NSRL or GSI wearable-calibration papers were found.

## 4. Sources: single-event effects and COTS electronics

- **Dodd, P. E., & Massengill, L. W. (2003).** Basic mechanisms and modeling of single-event upset in digital microelectronics. *IEEE Trans Nucl Sci*, 50(3), 583-602. DOI 10.1109/TNS.2003.813129. VERIFIED. Mechanism reference; predates deeply scaled nodes and multi-bit-upset-dominated regimes.
- **ECSS-Q-ST-60-15C Rev.1 (20 March 2025).** Space product assurance: radiation hardness assurance, EEE components. PARTIAL (existence, date, and supersession of the 2012 edition confirmed through ecss.nl; text not read). Adds a radiation design lifetime concept and an Equipment Radiation Control Board.
- **Sinclair, D., & Dyer, J. (2013).** Radiation effects and COTS parts in SmallSats. *27th AIAA/USU Conference on Small Satellites*, SSC13-IV-3. PARTIAL (abstract-level). "Careful COTS" approach; if lifetime dose is under 30 krad commercial parts will probably work.
- **Simhony, Y., Segal, A., Amrani, O., & Etzion, E. (2026).** High-end space electronics: active shielding to mitigate catastrophic single-event effects. *Aerospace Science and Technology*, 169, 111344 (arXiv 2502.04504). PARTIAL. Detector-triggered power cycling with 95% detection accuracy and about one power cycle per seven hours per cm2 of sensitive electronics.
- **SEU rate tools.** CREME96, OMERE, and SPENVIS (PARTIAL; no verified head-to-head benchmark). Run at least two tools with measured Weibull cross-sections.
- **CubeSat SEU campaigns** (for example RadFxSat, RADECS 2017): PARTIAL. A blog-reported multi-bit upset distribution is unverified; plan on a few percent multi-bit upsets, so single-bit ECC alone is not sufficient.
- **Gaps:** no peer-reviewed smartphone or COTS-microcontroller SEU rates from ISS were found; a Raspberry Pi proton-test figure came from a blog and is not used.

## 5. Sources: health effects, risk limits, and CNS evidence

- **Cucinotta, F. A., & Durante, M. (2006).** Cancer risk from exposure to galactic cosmic rays: implications for space exploration by human beings. *Lancet Oncology*, 7(5), 431-435. DOI 10.1016/S1470-2045(06)70695-7. VERIFIED. Historical framing of GCR uncertainty; predates current NASA models.
- **NASA-STD-3001 Vol. 1, Rev B** (signed Jan 2022), as quoted in a NASA OCHMO radiation technical brief (read; VERIFIED for Rev B wording). Career effective dose shall be less than 600 mSv; 250 mSv per solar particle event; 20 mSv per mission year for nuclear technologies. Rev C was reported approved in 2023 but was not opened (UNVERIFIED).
- **Cucinotta, F. A., et al. (2020).** Benchmarking risk predictions and uncertainties in the NSCR model of GCR cancer risks with revised low-LET risk coefficients. *Life Sci Space Res*, 27, 64-73. DOI 10.1016/j.lssr.2020.07.008. PARTIAL. Wide uncertainty dominated by quality factors and non-targeted effects; avoid single-number personal risk estimates.
- **Cucinotta, F. A., et al. (2024).** Non-targeted effects and space radiation risks for astronauts on multiple ISS and lunar missions. *Life Sci Space Res*, 40, 166-175. DOI 10.1016/j.lssr.2023.08.003. VERIFIED (Crossref). **Cucinotta & Saganti (2024)**, *Life Sci Space Res*, 40, 72-80, DOI 10.1016/j.lssr.2023.09.001. VERIFIED. Models now include circulatory disease and sex and smoking dependence.
- **Cucinotta, F. A., et al. (2014).** Space radiation risks to the central nervous system. *Life Sci Space Res*, 2, 54-69. DOI 10.1016/j.lssr.2014.06.003. VERIFIED for journal, volume, pages, DOI, first author; co-authors unconfirmed. Quantitative CNS risk estimates were described as not yet developed.
- **Chancellor, J. C., et al. (2018).** Limitations in predicting the space radiation health risk for exploration astronauts. *npj Microgravity*, 4, 8. DOI 10.1038/s41526-018-0043-2. VERIFIED. Ground simulations and animal models do not match observed human crew effects.
- **Patel, Z. S., et al. (2020).** Red risks for a journey to the red planet. *npj Microgravity*, 6, 33. DOI 10.1038/s41526-020-00124-6. VERIFIED. Names radiation among four priority risks.
- **NCRP Report No. 183 (2019)** Radiation exposures in space and the potential for CNS effects: Phase II. PARTIAL (title and year only). **ICRP Publication 123 (2013)**, PARTIAL.
- **Rodent evidence (not human):** Parihar et al. (2015), *Science Advances*, 1(4), e1400256, DOI 10.1126/sciadv.1400256 (VERIFIED; performance decrements and reduced dendritic complexity after charged-particle exposure). Kiffer et al. (2022), *Behav Brain Res*, 419, 113677, DOI 10.1016/j.bbr.2021.113677 (VERIFIED; 75 cGy 33-beam GCR in male mice: anxiety-like behavior and novel-object recognition unaffected, preference for social novelty blunted). Rienecker et al. (2023), *Sci Rep*, 13, 1749, DOI 10.1038/s41598-023-28508-0 (VERIFIED; behavioral deficits depend on combined stressors such as isolation and unloading).

## 6. Implementation steps

1. Define the claim: radiation context and SEU-fault flagging, not individual risk or CNS-effect inference.
2. Build scenario baselines from the measurements above (ISS Columbus, lunar surface, Mars transit, Orion belt-pass variation), citing the abstract-level status.
3. Model dose-rate variation by orbit phase (SAA crossings), shielding location, and vehicle attitude, since Artemis I showed fourfold spatial and roughly 50% attitude effects.
4. Frame cumulative-dose tracking against 600 mSv career and 250 mSv per SPE (Rev B); check Rev C text.
5. If hardware is needed, evaluate a small silicon or Timepix-class detector and require heavy-ion beam calibration with an explicit high-LET response curve.
6. For wearable electronics: screen total ionizing dose first, then proton and heavy-ion SEE tests on the exact MCU and sensor parts, following the ECSS structure.
7. Estimate SEU rates with at least two tools using measured Weibull cross-sections; document orbit, shielding, and solar assumptions; compare to flight SRAM campaigns.
8. Mitigate in layers: ECC with scrubbing, watchdogs and safe-state resets, TMR for critical logic, sensor plausibility checks, and detector-triggered power cycling for latch-up risk.
9. Log and tag all faults (resets, bit flips, sensor glitches) with timestamps and orbital and SPE context so SEU effects can be separated from physiology.
10. Keep health claims aligned with evidence; record covariates (isolation, sleep, unloading).
11. Validate with public data (see [data-collection-sources](../data-collection-sources/space-radiation.md)); confirm portal terms before redistribution.
12. Before publishing, close the UNVERIFIED items and read NASA-STD-3001 Rev C and NCRP 183 directly.

## 7. Evaluation and robustness

- **Metrics:** detection efficiency versus source activity; dead-time correction accuracy; false-flag rate in a background run; fraction of physiological data flagged by the suppression rule.
- **Robustness checks:** detector gain versus temperature; power-supply noise coupling; agreement between two independent detectors; behavior at high count rates.
- **Reporting:** state calibration source, geometry, and uncertainty with every dose number.

## 8. Limitations and open questions

- A lab beta or gamma source does not reproduce heavy-ion energy deposition; SEU cross-sections need beam testing.
- Dose varies with shielding and location, so a single reading does not apply everywhere.
- No validated quantitative CNS risk model exists; rodent results do not establish a human behavioral signature.
- Does a body-worn detector add value beyond a cabin dosimeter for data-quality flagging? What flag window is justified by SEU physics?
- Does NASA-STD-3001 Rev C change any limit versus Rev B? Are there peer-reviewed M-42, HERA, and MARE papers beyond George 2024? What are the Hassler 2014 and Zeitlin 2013 daily dose-equivalent values from full text?
- Is a detector small and low-power enough for wearables with a validated high-LET response?
