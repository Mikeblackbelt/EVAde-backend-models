# Behavioral Health Data Sources

## GLOBEM (PhysioNet, credentialed access)
- **What:** multi-year passive sensing data from mobile and wearable devices, over 700 user-years from 497 unique users, collected over four years (10 weeks each year, 2018-2021), with surveys, features, and participant info. Features were extracted with RAPIDS. The paper benchmarks 18 algorithms on depression detection.
- **Access:** credentialed access on PhysioNet (the authors released feature-level data only, because of sensitivity). Approval timing was not stated; check the PhysioNet page for steps. Code is at github.com/UW-EXP/GLOBEM.
- **Use:** cross-year generalization tests for behavioral models, which the evaluation protocol requires.
- **Limit:** university student population; features, not raw sensor data.

## DAIC-WOZ (USC Institute for Creative Technologies)
- **What:** clinical interview recordings with depression labels; an extended version was used in the AVEC 2019 challenge.
- **Access:** through a data use agreement; the page has an "Apply Now" link. The request form's required fields were not confirmed.
- **Use:** only if audio/text modalities are added. Not central to a wearable-sensor design.
- **Limit:** interview setting, not daily life.

## DEAP (Queen Mary University of London)
- **What:** EEG and peripheral physiological signals recorded while participants watched music videos, with emotion ratings.
- **Access:** EULA signed and emailed back; academic research use only, and the download page says only a person with a permanent position at an academic or research institute can apply. Students may need a faculty sponsor (not verified).
- **Use:** EEG-based affect research and cross-checks against peripheral signals.
- **Limit:** lab stimulus design; small sample per person.

## WESAD
- Stress and affect benchmark with ECG, EDA, EMG, respiration, temperature, and accelerometer (15 subjects). Terms and notes are in [physiological-signals.md](physiological-signals.md).

## Mars-500 and other analog datasets
- **Status:** not checked for public availability. The Basner et al. papers analyze Mars-500 data; whether any dataset is shared was not found. Contact the authors or the sponsoring institutes.
- **Use if available:** actigraphy, sleep, and mood questionnaires over long confinement.

## NASA Life Sciences Data Archive (behavioral and sleep studies)
- **Status:** not checked for behavioral datasets. Access is by request through the NASA Life Sciences Portal; see [bone-and-spaceflight.md](bone-and-spaceflight.md) for notes on the portal.

## Driver stress data (Healey and Picard)
- **Status:** a PhysioNet-hosted driver stress dataset is commonly associated with this work, but this was not verified. Confirm the dataset name, license, and contents on PhysioNet before use.

## Instruments (questionnaires and tests)
- **PHQ-9 and GAD-7:** widely used; the sources disagree on how clearly their reuse terms are stated (GAD-7 is described as Pfizer-copyrighted with no reproduction charge, and sometimes as public domain). Confirm terms for the exact text you will embed.
- **PVT:** tests vary in implementation. Decide on the exact version (10-minute standard or a shorter variant) and cite it; effect sizes depend on duration and the outcome metric (Basner & Dinges, 2011).
