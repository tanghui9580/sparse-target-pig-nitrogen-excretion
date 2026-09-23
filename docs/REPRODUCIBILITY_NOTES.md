# Numerical reproducibility notes

## Executable analyses

The repository contains executable code for the selected MLR+RF refit and direct transfer, target measurement selection, model updating, the 504 balanced outcome-record allocation analysis, the four-architecture downstream extension, gestating-sow evaluation, and the classical domain-adaptation/transfer benchmark. Pre-update predictions for the four starting architectures are included as analysis inputs. TabM training used during model development is represented by retained predictions and model-selection evidence rather than by a separate end-to-end retraining pipeline.

## Model-development analysis coverage

The P0–P3 predictor-information comparison and ten-model architecture-selection analysis belong to the model-development stage. Their machine-readable results and verification materials are retained in `results/predictor_information/` and `results/model_selection/`. Target-domain analyses can be reproduced from the selected model specification without end-to-end retraining of every historical candidate model from raw development inputs.

The 233-record confirmation set and the 546-record target system were not used to select the predictor-information tier or winning starting architecture.

## Verification tiers

`python scripts/reproduce.py verify` is the fast integrity/numerical contract check. It now includes manifest verification, core numerical contracts, historical provenance, byte-identity checks for intentionally duplicated canonical files, and public-release hygiene.

`python scripts/reproduce.py release-audit --jobs 1` is the GitHub CI gate. It adds selector reconstruction, light-method smoke testing, a representative four-cell heavy-method smoke test, the classical benchmark/Table S5 rebuild, prediction-level RN/TNE verification, and the complete publication build.

`derived-endpoints` is included in that CI gate through an optimized verification path: it recomputes the required selected-procedure updates across all 504 allocations, four architectures, and four prespecified budgets, but directly summarizes RN/TNE R² instead of materializing the large per-source prediction file. It also omits only the unrelated 10,000-resample paired AUBC bootstrap; the scientific script retains its full bootstrap by default. `primary-factorial` and `full-extension` remain full analyses outside the default CI gate.

## 504 balanced outcome-record allocations

The 504 allocations are fixed-data scenarios constructed as 84 blocks × 6 rotations. Within each source-specific diet, the six pig records are balanced across the role of supplying revealed target FN/UN values. Percentile ranges across these scenarios describe sensitivity to the selected pig record; they are not confidence intervals or independent biological replicates.

## Measurement budgets

Budgets 3, 6, 9, and 12 constitute the prespecified analysis. Budgets 15, 18, and 21 are an extended-budget analysis and are reported separately.

## Figure artwork

Machine-readable source tables are included for Figures 1–5 and Supplementary Figures S1–S10 together with `scripts/build_publication_figures.py`, which regenerates all 15 figures in PNG and EPS. Exact locked submission artwork and styled Excel workbooks are retained under `publication_reference/` and re-materialized with SHA-256 verification by the one-command publication build.

For Figure S2, the 546-target starch row represents the prespecified historical-training-median input used by the fixed prediction model because raw target starch was structurally unavailable. The target-specific P2-PAM measurement-allocation representation used ash rather than starch, as documented in the supplementary methods.

## Historical-source provenance

All 170 Study_ID values in `historical_906.csv` are mapped in `data/provenance/HISTORICAL_170_STUDY_PROVENANCE.csv`. The resolution hierarchy is: V44 reviewer-ready Study Index → verified v35 identifier fallback → publisher/journal metadata for six remaining identities. No DOI is inferred from Study_ID text. The resulting `Best_Available_Source_Identifier` field is nonblank for 170/170 Study_ID values.

The provenance table also carries `CLEAR_records`, `TRACEABLE_CAVEAT_records`, and `REVIEW_FLAG_records` from the broader reviewer-facing data audit. These are **record/field-level QA counts**, not paper-identity confidence labels. A study can have a securely mapped source identity while some nutrient fields retain a traceable caveat or review flag.

## Self-contained subpackages and duplicate contracts

Several canonical inputs are intentionally copied into `analysis/architecture_specific/` and `analysis/classical_da_transfer_benchmark/` so those subpackages can be inspected or executed independently. `scripts/verify_duplicate_contracts.py` hashes the canonical copies and fails if they drift apart. This preserves self-containment without allowing silent version divergence.
