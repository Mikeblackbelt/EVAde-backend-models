# Bone Health and Spaceflight Data Sources

## NASA Open Science Data Repository (OSDR)
- **What:** multi-omics and physiological data from NASA spaceflight and ground studies, including bone-loss studies in rodents (for example, Rodent Research-6). Combines GeneLab omics with data from the Ames Life Sciences Data Archive (ALSDA).
- **Access:** public at osdr.nasa.gov; processed data are public. Also available on AWS as a public bucket (`nasa-osdr`, listable without an account per the AWS registry). Raw human data requires an application and board approval.
- **Use:** mostly animal and omics data. Not a direct source for human bone density, but useful for mechanism and countermeasure context.
- **Limit:** dataset accession numbers and bone-specific filters were not checked in this pass. Search osdr.nasa.gov for bone or skeletal terms.

## NASA Life Sciences Portal / Life Sciences Data Archive (LSDA)
- **What:** human and animal spaceflight physiology data from 1961 onward, accessed through the NASA Life Sciences Portal (NLSP).
- **Access:** a request is required. A single request form was used across repositories in a 2012 presentation; confirm the current process on the portal. Human data may need an IRB-equivalent review.
- **Use:** the primary source for human in-flight bone density data, if access is granted.
- **Limit:** the search results found mostly animal bone datasets. Human DXA or QCT availability needs direct confirmation.

## MrOS (Osteoporotic Fractures in Men) and SOF (Study of Osteoporotic Fractures)
- **What:** large prospective cohorts of older adults with DXA, QCT, and HR-pQCT scans, fracture outcomes, and in MrOS a sleep sub-study (n = 3,135 per the MrOS Online listing).
- **Access:** MrOS Online is public to registered users who accept its data use agreement. SOF access is through its own data-request process; not checked in this pass. NSRR hosting was not confirmed.
- **Use:** calibration of a longitudinal bone-loss model and validation of portable proxies against DXA. Older male population is the MrOS limit; SOF is women.
- **Limit:** these are not spaceflight cohorts. Use them for measurement validation, not for the mission-loss rate.

## NHANES (CDC)
- **What:** national survey with DXA scans in a subset of participants, plus demographic and fitness covariates.
- **Access:** public, no application needed. Confirm the DXA module years on the CDC site before use.
- **Use:** reference distributions of BMD by age, sex, and ethnicity for normative comparisons.
- **Limit:** cross-sectional, not longitudinal.

## UK Biobank
- **What:** large cohort with imaging and health records, including bone measures in a subset.
- **Access:** application required, with fees and data-use terms. Not checked in this pass.
- **Use:** large-sample validation of portable proxies, if access is granted.
