# Literature Review

Working notes on the six areas Blindsight depends on (radiation, vital signs, ECG, EEG, bone health, behavioral health), a cross-cutting integration and evaluation document, and a folder of data sources. No code yet. Each topic folder holds a summary, verified sources, and concrete implementation steps. `data-collection-sources/` lists where training and validation data can come from, with access terms.

| Folder | Topic |
|---|---|
| [space-radiation-detection](space-radiation-detection/README.md) | Detecting and quantifying ionizing radiation in spacecraft and on the body |
| [vital-sign-monitoring](vital-sign-monitoring/README.md) | Contactless and wearable heart rate, respiration, SpO2 |
| [ecg-applications](ecg-applications/README.md) | ECG signal processing, arrhythmia and stress applications |
| [eeg-applications](eeg-applications/README.md) | EEG signal processing, seizure and cognitive-state applications |
| [bone-health-metrics](bone-health-metrics/README.md) | Bone density, fracture risk, and bone loss in spaceflight |
| [behavioral-health](behavioral-health/README.md) | Mood, anxiety, stress, sleep, and alertness monitoring in isolation and confinement |
| [model-architectures](model-architectures/README.md) | Candidate model architectures per module, with verified sources and the `EVA-[n]` ID scheme |
| [integration-and-evaluation.md](integration-and-evaluation.md) | Shared schema, evaluation protocol, risks, and phasing across all areas |
| [data-collection-sources](data-collection-sources/README.md) | Public and controlled datasets, with license and access notes |

## Verification status

A verification pass (second round) checked the key citations and dataset access terms against search results, abstracts, and press releases. Full texts were mostly not accessible (publisher pages returned 403s or redirects), and some DOIs remain unconfirmed; the bibliography entries still say so where that applies. Specific numbers must be re-checked against the paper before they go into a report. Items marked **UNRESOLVED** or **UNVERIFIABLE** have conflicting or missing sources.
