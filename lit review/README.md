# Literature Review

Working notes for **EVAde**, a space health monitoring platform, on the six areas the backend depends on (radiation, vital signs, ECG, EEG, bone health, behavioral health), a cross-cutting integration and evaluation document, model architectures, theory-only planning, and a folder of data sources. **No code yet.** Each topic folder holds a summary, labeled sources, and concrete implementation steps. `data-collection-sources/` lists where training and validation data can come from, with access terms. `planning/` holds the risk score, maximum days to return, data acquisition (including the 2-day hackathon scope), per-user baseline comparison, and Bluetooth plans. The whole set is also in [lit-review.pdf](lit-review.pdf).

| Folder | Topic |
|---|---|
| [space-radiation-detection](space-radiation-detection/README.md) | Detecting and quantifying ionizing radiation in spacecraft and on the body |
| [vital-sign-monitoring](vital-sign-monitoring/README.md) | Contactless and wearable heart rate, respiration, SpO2 |
| [ecg-applications](ecg-applications/README.md) | ECG signal processing, arrhythmia and stress applications |
| [eeg-applications](eeg-applications/README.md) | EEG signal processing, seizure and cognitive-state applications |
| [bone-health-metrics](bone-health-metrics/README.md) | Bone density, fracture risk, and bone loss in spaceflight |
| [behavioral-health](behavioral-health/README.md) | Mood, anxiety, stress, sleep, and alertness monitoring in isolation and confinement |
| [model-architectures](model-architectures/README.md) | Candidate model architectures per module, with verified sources and the `EVA-[n]` ID scheme |
| [planning](planning/README.md) | Theoretical design for an automatic risk score in (0,1) and maximum days to return (under 5 to over 100), plus the implementation pipeline (ingestion, training, FastAPI serving) |
| [integration-and-evaluation.md](integration-and-evaluation.md) | Shared schema, evaluation protocol, risks, and phasing across all areas |
| [data-collection-sources](data-collection-sources/README.md) | Public and controlled datasets, with license and access notes |

## Verification status

The topic sections were rewritten from six parallel literature searches that prioritized recent peer-reviewed sources (2019-2026: meta-analyses, systematic reviews, large studies, guidelines). Bibliographic fields were checked against Crossref, PubMed, or Europe PMC, and abstract-level numbers against PubMed or Europe PMC abstracts. Every source is labeled **VERIFIED** (record and abstract checked), **PARTIAL** (record checked, abstract or numbers not read), or **UNVERIFIED**. Full texts were mostly not read, and publisher pages often returned 403s, so numbers must be re-checked against the paper before they go into a report. Press releases are no longer used as number sources.

Known gaps are listed in each section, including: no verified depression-specific EEG-ECG fusion study; no ICE-validated questionnaire cutoff; no 2020+ Mars-500, NEK, LunAres, or HI-SEAS results found; no validated portable bone proxy for spaceflight; and unconfirmed items such as NASA-STD-3001 Rev C text and the final FDA pulse oximeter guidance.
