# Bone Health and Spaceflight Data Sources

Access terms change; confirm each before use. Status: **VERIFIED** = checked on the source page or a bibliographic record; **PARTIAL** = from search summaries; **UNVERIFIED** = not confirmed.

## NASA Open Science Data Repository (OSDR)
- **What:** multi-omics and physiological data from NASA spaceflight and ground studies, including bone-loss studies in rodents (for example Rodent Research-6). Combines GeneLab omics with data from the Ames Life Sciences Data Archive.
- **Access:** public at osdr.nasa.gov for processed data; also on AWS as a public bucket (`nasa-osdr`, listable without an account per the AWS registry). Controlled-access human data goes through a request on the study's Files tab, a curator, and a Data Access Committee (from a search summary; PARTIAL).
- **Use:** mostly animal and omics data; mechanism and countermeasure context, not human bone density.
- **Limit:** bone-specific accession numbers were not checked.

## NASA Life Sciences Portal / Life Sciences Data Archive (LSDA)
- **What:** human and animal spaceflight physiology data from 1961 onward.
- **Access:** request-based. The LSDA moved under the NASA Life Sciences Portal; the request page is nlsp.nasa.gov/explore/lsdahome/datarequest. A search summary indicates a request form, a review, and NASA IRB approval, with de-identified and possibly attributable astronaut data available for peer-reviewed research (PARTIAL; the page body could not be fetched).
- **Use:** the primary route to human in-flight bone density data.
- **Limit:** mostly animal bone datasets surfaced in searches; confirm human DXA and QCT availability directly.

## NASA Lifetime Surveillance of Astronaut Health (LSAH)
- **Status:** current request procedure not found (UNVERIFIED). Older NASA slides describe one request form across LSDA and LSAH, with Privacy Act constraints. Contact NASA directly.

## MrOS and SOF
- **What:** large prospective cohorts of older adults with DXA, QCT, and HR-pQCT scans and fracture outcomes. MrOS also has a sleep sub-study; NSRR lists 2,907 men with visit-1 polysomnography.
- **Access:** MrOS Online is public to registered users who accept its data use agreement. MrOS sleep data are hosted on sleepdata.org (NSRR). SOF materials are requested through the NIA Aging Research Biobank ("Request for Materials"); the ancillary-study process was not confirmed.
- **Use:** calibration of longitudinal bone-loss models and validation of proxies against DXA.
- **Limit:** not spaceflight cohorts; older men (MrOS) or women (SOF).

## NHANES (CDC)
- **Access:** public, no application. DXA files by cycle (verified on CDC pages):
  - 1999-2006: whole-body DXA (femur and spine for 2005-2006 not checked).
  - 2007-2008 and 2009-2010: femur and spine.
  - 2011-2012 and 2015-2016: whole body and android/gynoid only (no femur or spine seen).
  - 2013-2014: full set (femur, spine, FRAX, vertebral fracture assessment).
  - 2017-March 2020 (pre-pandemic): femur and spine.
  - 2021-2023: no DXA file on the examination list page.
- **Use:** reference distributions of BMD by age, sex, and ethnicity. **Limit:** cross-sectional, general US population.

## UK Biobank
- **What:** large cohort with an imaging enhancement that includes DXA (whole body, lumbar spine, hips, knees, vertebral fracture assessment) in a subset; current DXA n not verified (PARTIAL).
- **Access:** application via the Research Analysis Platform; downloads are exceptional. Fees quoted in search text: 9,000 GBP plus VAT for 3 years (imaging tier) and 500 GBP plus VAT for eligible students (data-only); confirm on the fees page. Whether DXA is platform-only is unclear.
