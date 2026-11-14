# Model Architectures

This document surveys candidate model architectures for each module in the review. It is theoretical: no model has been built or benchmarked here. Every candidate that is actually created gets an ID from the registry (see [MODEL_REGISTRY.md](../../MODEL_REGISTRY.md)); the candidates below are unnumbered until then.

Status labels: **VERIFIED** = bibliographic record and abstract (or proceedings page) checked; **PARTIAL** = record checked but abstract or some fields not read, or venue details from secondary sources; **UNVERIFIED** = not confirmed. Numbers are as reported by the cited papers and were not reproduced.

## 1. Selection principles

1. **Start with a simple baseline per module** and require every learned model to beat it under subject-wise evaluation. Trees beat deep networks on typical medium-sized tabular data (Grinsztajn et al. 2022: 45 datasets, tree-based models state-of-the-art around 10,000 samples), and crew-scale datasets are far smaller than that.
2. **Match capacity to data.** Deep time-series models need large labeled sets. Crew and analog datasets are small, so prefer classical signal processing plus simple models, or pretrained models fine-tuned lightly.
3. **Evaluate for unseen people.** Segment-wise or record-wise splits overestimate performance (Saeb 2017, Brookshire 2024); see [behavioral-health/ecg-hrv.md](../behavioral-health/ecg-hrv.md) and [behavioral-health/eeg.md](../behavioral-health/eeg.md).
4. **Output calibrated uncertainty**, not just a class. Calibration and conformal methods are in section 9.
5. **Respect edge constraints.** Wearable and habitat hardware limits compute and power; an architecture that needs a GPU is a research baseline, not a deployment candidate.
6. **Version and trace everything.** Each retained architecture and trained model gets a registry ID with data version, split version, and metrics with confidence intervals.

## 2. General building blocks

- **Residual networks.** He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770-778. VERIFIED (record). Residual connections are the backbone of the strongest ECG classifiers below.
- **LSTM.** Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780. VERIFIED (record). Recurrent baseline for physiological sequences.
- **Transformer.** Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems* (NIPS 2017). VERIFIED (arXiv 1706.03762 and venue; the volume number was not confirmed).
- **Structured state-space models.** Gu, A., Goel, K., & Ré, C. (2022). Efficiently modeling long sequences with structured state spaces. *ICLR 2022* (S4). PARTIAL (secondary summaries). Gu, A., & Dao, T. (2024). Mamba: linear-time sequence modeling with selective state spaces. arXiv 2312.00752; listed as COLM 2024 in secondary sources, so the venue is PARTIAL. Candidate for long physiological sequences, but no verified physiological-signal benchmark was reviewed.

## 3. ECG

**Task:** QRS and beat detection, rhythm classification, and HRV feature extraction (see [ecg-applications](../ecg-applications/README.md)).

| Candidate | Evidence | Notes |
|---|---|---|
| Pan-Tompkins detector | Pan & Tompkins (1985), *IEEE TBME*, BME-32(3), 230-236; VERIFIED; 99.3% QRS detection on MIT-BIH per the abstract | Non-learned reference; keep as the fallback and as the HRV source |
| 1D residual and Inception CNNs | Strodthoff et al. (2021), *IEEE J Biomed Health Inform*, 25(5), 1519-1528, DOI 10.1109/JBHI.2020.3022989; VERIFIED; ResNet- and Inception-style CNNs strongest across PTB-XL tasks | Default supervised baseline |
| Large residual network, 12-lead | Ribeiro et al. (2020), *Nat Commun*, 11, 1760, DOI 10.1038/s41467-020-15432-4; VERIFIED; over 2 million labeled exams, F1 above 80% and specificity above 99% for six abnormalities | Single health-system source; shift not addressed |
| Single-lead deep network | Hannun et al. (2019), *Nat Med*, 25, 65-69, DOI 10.1038/s41591-018-0268-3; VERIFIED (citation); 12 rhythm classes, AUROC 0.97 | Architecture details were not verified here; the funding relationship with the device maker is unverified |
| Contrastive pretraining across patients | Kiyasseh, Zhu & Clifton, CLOCS, *ICML 2021*, PMLR 139, 5606-5615; PARTIAL (venue year varies by source; arXiv 2005.13249) | Uses patient-level positives; outperformed SimCLR and BYOL in the authors' linear evaluations |
| Self-supervised ECG | Mehari & Strodthoff (2022), *Comput Biol Med*, 141, 105114, DOI 10.1016/j.compbiomed.2021.105114; VERIFIED (abstract); first comprehensive assessment of self-supervised learning on clinical 12-lead ECG | Label-efficient pretraining |
| ECG foundation model | McKeen et al. (2025), ECG-FM, *JAMIA Open*, 8(5), ooaf122, DOI 10.1093/jamiaopen/ooaf122; VERIFIED (abstract); open weights, code, and benchmark | Training data not verified; clinical, not wearable |
| ECG foundation model (large) | ECGFounder, *NEJM AI*, 2025, 2(10), AIoa2500913, DOI 10.1056/AIx2500913; PARTIAL; over 10 million ECGs per search summary | Author list unverified; a benchmark preprint reports no single foundation model consistently excelled (preprint, not peer reviewed) |
| Stress detection with contrastive learning | "Contrastive self-supervised learning for stress detection from ECG data" (PMC9404921); PARTIAL (found in search; reports a 9% accuracy gain on WESAD, which has only 15 subjects) | Smoke-test level evidence |
| LSTM on wearable HRV | Coutts et al. (2020), *J Biomed Inform*, 112, 103610; VERIFIED (core fields); 652 participants, accuracy up to 83% from 5-minute HRV | Split scheme not stated in the abstract |

**Recommendation.** Stage 1: Pan-Tompkins plus HRV feature extraction. Stage 2: a 1D ResNet baseline for rhythm flags, trained on PTB-XL-style data and externally tested. Stage 3: lightly fine-tune an open ECG foundation model, and measure shift to wearable single-lead data before trusting it. PhysioNet challenge results show cross-site shift dominates lead count (Reyna et al. 2022, *Physiol Meas*, 43(8), 084001), so plan multi-site external validation.

## 4. EEG

**Task:** sleep staging, seizure detection, and exploratory state tracking (see [eeg-applications](../eeg-applications/README.md) and [behavioral-health/eeg.md](../behavioral-health/eeg.md)).

| Candidate | Evidence | Notes |
|---|---|---|
| Band-power and line-length features with gradient boosting | classical baseline; Shoeb & Guttag (2010), *ICML 2010*, 975-982, VERIFIED, patient-specific classifiers on CHB-MIT | Strong, cheap baseline; boosted trees are favored on small tabular data (Grinsztajn 2022) |
| EEGNet | Lawhern et al. (2018), *J Neural Eng*, 15(5), 056013, DOI 10.1088/1741-2552/aace8c; VERIFIED | Compact CNN; default deep baseline |
| EEG Conformer | Song et al. (2023), *IEEE TNSRE*, 31, 710-719, DOI 10.1109/TNSRE.2022.3230250; VERIFIED | Convolution plus self-attention; the reported accuracies (for example 78.66% on BCI IV-2a, 95.30% on SEED) are from the code repository README and were not reproduced |
| Graph neural network with self-supervised pretraining | Tang et al. (2022), *ICLR 2022*; VERIFIED via the ICLR and arXiv records; TUSZ v1.5.2, AUROC 0.875 for seizure detection and weighted F1 0.749 for seizure classification, with five patients overlapping the official split excluded | Models electrode connectivity; improved rare seizure types |
| EEG foundation models | LaBraM (Jiang, Zhao & Lu, ICLR 2024, arXiv 2405.18765; PARTIAL), BIOT (Yang, Westover & Sun, NeurIPS 2023; PARTIAL), EEGPT (NeurIPS 2024; PARTIAL) | Kuruppu et al. (2026), *J Neural Eng*, 23(2), 021001, VERIFIED: evaluations are "heterogeneous and largely limited" |
| Deep-learning EEG reviews | Roy et al. (2019), Craik et al. (2019); PARTIAL | Background only |

**Recommendation.** Baseline with features plus boosted trees, then EEGNet. Treat Conformer, GNN, and foundation models as research arms, always evaluated patient-disjoint and scored with event-based SzCORE metrics (Dan et al. 2025, *Epilepsia*, 66(Suppl 3), 14-24). Wearable montages (behind-the-ear, in-ear) need their own validation; do not assume transfer from 10-20 scalp models.

## 5. PPG and remote PPG (rPPG)

**Task:** heart rate, respiration, and SpO2 (see [vital-sign-monitoring](../vital-sign-monitoring/README.md)).

| Candidate | Evidence | Notes |
|---|---|---|
| Plane-Orthogonal-to-Skin (POS) | Wang et al. (2017), *IEEE TBME*, 64(7), 1479-1491, DOI 10.1109/TBME.2016.2609282; VERIFIED | Unsupervised, training-free baseline |
| ICA-based extraction | Poh, McDuff & Picard (2010), *Optics Express*, 18(10), 10762-10774; VERIFIED | Historical baseline |
| DeepPhys | Chen & McDuff (2018), *ECCV*, LNCS 11206, 356-373; VERIFIED (record) | Attention CNN; lineage |
| MTTS-CAN and TS-CAN | Liu, Fromm, Patel & McDuff (2020), *NeurIPS 33*; VERIFIED (proceedings page, arXiv 2006.03790) | Joint pulse and respiration; authors report over 150 frames per second on an ARM CPU (authors' claim) |
| PhysNet | Yu, Li & Zhao (2019), BMVC, arXiv 1905.02419; **UNVERIFIED venue** | Confirm on the BMVC site before citing |
| PhysFormer and PhysFormer++ | Yu et al. (2022), *CVPR*, 4176-4186, DOI 10.1109/CVPR52688.2022.00415; VERIFIED (record). PhysFormer++: *Int J Comput Vis*, 131(6), 1307-1330, DOI 10.1007/s11263-023-01758-1; VERIFIED (record) | Temporal-difference transformer; compute-heavy |
| RhythmFormer | Zou et al. (2025), *Pattern Recognition*, 164, 111511, DOI 10.1016/j.patcog.2025.111511; VERIFIED (venue) | Reported results are author claims on controlled datasets; do not treat as deployment accuracy |
| Synthetic pretraining | McDuff et al. (2022), SCAMPS, *NeurIPS 35*, 3744-3757, DOI 10.52202/068431-0271; VERIFIED (record) | Adds skin-tone and pose diversity; validate on real data |
| Standard harness | Liu et al. (2023), rPPG-Toolbox, *NeurIPS 36*, 68485-68510; VERIFIED (venue) | Use to compare all candidates under identical preprocessing |

**Recommendation.** Compare POS, TS-CAN, and one transformer in rPPG-Toolbox on identical splits, with a tone-stratified evaluation set. See the skin-tone evidence in the vital-sign document before choosing a winner on mean error alone.

## 6. Behavioral trend model

**Task:** per-person baseline, drift, and flag from sleep, HRV, cognition, and optional EEG features (see [behavioral-health](../behavioral-health/README.md)).

| Candidate | Evidence | Notes |
|---|---|---|
| Per-person baseline with z-score or robust deviation | Nunan et al. (2010), *PACE*, 33(11), 1407-1417: very large inter-individual variation in short-term HRV; VERIFIED | Simplest design; the reference arm |
| Anomaly detection on individualized baselines | Henson et al. (2021), *Transl Psychiatry*, 11, 28; VERIFIED; 89% sensitivity and 75% specificity in a small clinical cohort | Expect many false alarms |
| Gradient-boosted trees on windowed features | Grinsztajn et al. (2022), *NeurIPS Datasets and Benchmarks*, arXiv 2207.08815; VERIFIED (preprint, poster page) | Strong default for tabular features; the paper's finding applies to typical medium-sized tabular data |
| LSTM or Transformer on feature sequences | Coutts et al. (2020), above | Needs subject-wise validation and large data |
| Mixed-effects and state-space (Kalman) trend models | **No verified citation in this review**; standard statistical methods | Natural fit for slow drift with time-in-mission terms; cite a textbook or paper once chosen |

**Recommendation.** Baseline: per-person robust z-score plus a time-in-mission term. Next: mixed-effects regression and boosted trees. Deep sequence models only if data volume justifies them. All outputs carry calibrated intervals.

## 7. Multimodal fusion (ECG with EEG and others)

- **Taxonomy.** Baltrusaitis, Ahuja & Morency (2019). Multimodal machine learning: a survey and taxonomy. *IEEE TPAMI*, 41(2), 423-443. VERIFIED. Five challenges: representation, translation, alignment, fusion, and co-learning.
- **Practical choices** (see [behavioral-health/multimodal-ecg-eeg.md](../behavioral-health/multimodal-ecg-eeg.md)): decision-level fusion first because it degrades gracefully when one stream drops; feature-level fusion only when an ablation shows added value; quality-gate every stream.
- **Clinical example.** Zhang et al. (2024), *Physiol Meas*, 45(6), DOI 10.1088/1361-6579/ad4e94; VERIFIED. Fusing behind-the-ear EEG with other wearable signals improved tonic-clonic seizure detection over unimodal.

## 8. Radiation event flagging and bone trend

- **Radiation and SEU flags.** A rule-based architecture is the supported choice: threshold detection on a particle detector, a fault log, and detector-triggered power cycling. Simhony et al. (2026), *Aerosp Sci Technol*, 169, 111344 (arXiv 2502.04504; PARTIAL), report a detector array that power-cycles COTS electronics with 95% detection accuracy. No learned model is needed or justified for the first version.
- **Bone trend.** Empirical longitudinal models with population priors. Stavnichuk et al. (2020), *npj Microgravity*, 6, 13, DOI 10.1038/s41526-020-0103-2 (VERIFIED), give site-specific rates and marker kinetics. Kemp et al. (2023), *Life*, 13(11), 2183, DOI 10.3390/life13112183 (VERIFIED), found a participant-specific mechanistic model reproduced static morphometry (R2 above 0.94) but estimated dynamic rates poorly, so prefer empirical trend estimation with uncertainty.

## 9. Calibration and uncertainty (applies to every module)

- **Calibration.** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*, PMLR 70, 1321-1330. VERIFIED. Modern networks are poorly calibrated; temperature scaling is a one-parameter fix.
- **Conformal prediction.** Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: a gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494-591. DOI 10.1561/2200000101. VERIFIED. Distribution-free intervals, with extensions to time series and distribution shift. Note that exchangeability assumptions need care for longitudinal per-person data.

## 10. Candidate shortlist by module

| Module | Stage 1 (baseline) | Stage 2 | Stage 3 (research) |
|---|---|---|---|
| Radiation / SEU | threshold detector and fault log | detector-triggered power cycling | none |
| PPG / rPPG | POS | TS-CAN | PhysFormer or RhythmFormer |
| ECG rhythm | Pan-Tompkins plus rules | 1D ResNet | fine-tuned ECG foundation model |
| HRV features | RMSSD and related, standardized window | quality gating | none |
| EEG seizure | features plus boosted trees | EEGNet | GNN with self-supervision |
| EEG sleep or drowsiness | features plus boosted trees | EEGNet | foundation model arm |
| Behavioral trend | per-person robust z-score | mixed-effects or boosted trees | LSTM or Transformer |
| Fusion | decision-level | feature-level with ablation | learned multimodal encoder |
| Bone trend | empirical longitudinal model | mechanistic arm (comparison) | none |
| Uncertainty | temperature scaling | conformal intervals | none |

## 10a. Serving constraints

Models exposed through the FastAPI service must fit the latency and memory budget in [planning/implementation-pipeline.md](../planning/implementation-pipeline.md) (design: ECG and EEG windows at or below 500 ms median on the target host). Prefer the Stage 1 and 2 entries above for first deployment; Stage 3 models are research arms until they meet the budget and external validation. Each served model needs an `EVA-[n]` registry entry.

## 11. Limitations

- No architecture was trained or evaluated here; all performance numbers are the cited papers' own and mostly come from non-wearable data.
- Several entries are PARTIAL or UNVERIFIED (PhysNet venue, S4 and Mamba venues, foundation-model details, ECGFounder authors); check before citing.
- A verified citation for mixed-effects and state-space trend models was not found.
- Foundation models for EEG and ECG have limited, heterogeneous evaluations and are not validated for wearable or behavioral use.
