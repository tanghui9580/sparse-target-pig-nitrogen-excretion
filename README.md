# Sparse target measurements restore pig nitrogen-excretion predictions across independent settings

Numerical reproducibility package for the manuscript **“Sparse target measurements restore pig nitrogen-excretion predictions across independent settings.”** It contains analysis-ready data, executable downstream analyses, retained model-development evidence, machine-readable results, figure-source tables, and publication-layer reference assets.

## Repository map

- `data/analysis_matrices/`: 906-record literature matrix, independent 546-record growing-pig target system, and 80-record gestating-sow evaluation dataset (internal domain/file code: SOW80).
- `data/design/`: deterministic measurement-allocation ledgers, including the 504 balanced outcome-record allocations.
- `data/provenance/`: research-level source mapping for all 170 historical `Study_ID` values.
- `data/pre_update_predictions/`: predictions generated before target FN/UN labels are revealed.
- `analysis/route/`: selected-model refit, direct transfer, target updating, architecture robustness, and SOW80 evaluation.
- `analysis/architecture_specific/`: architecture-specific response-surface-guided sampling across four starting architectures, seven budgets, eight update methods, and 504 balanced allocations.
- `analysis/classical_da_transfer_benchmark/`: classical domain-adaptation/transfer benchmark underlying Supplementary Table S5.
- `results/`: retained machine-readable results used by the manuscript and supplement.
- `figure_sources/`: numerical source tables for Figures 1–5 and Supplementary Figures S1–S10.
- `publication_reference/`: exact locked submission artwork, publication-source table workbooks, and figure-caption files.
- `docs/`: data dictionary, manuscript/asset mapping, and reproducibility-scope notes.

The manuscript-to-asset mapping is in `docs/FIGURE_TABLE_ALIGNMENT.md`.

## Reference environment

Python **3.12** is the reference environment. Package versions are pinned in both `requirements.txt` and `environment.yml`.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Text files are normalized to LF by `.gitattributes`; binary publication assets, including EPS, are explicitly protected from Git line-ending conversion.

## Verification tiers

### 1. Quick integrity/numerical verification

```bash
python scripts/reproduce.py verify
```

This checks:

- package manifest integrity;
- 906 = 673 development + 233 confirmation and 170 historical Study_ID values;
- 546 = 91 source-specific diets × 6 pig records;
- nitrogen mass-balance contracts;
- the 504 balanced allocation ledger;
- P0–P3/six-family retained evidence and selected-model contracts;
- key growing-pig and SOW80 manuscript metrics;
- Supplementary Table S5 and Figure 5 source contracts;
- 170/170 historical Study_ID provenance and 170/170 best-available source identifiers;
- byte identity of intentionally duplicated canonical files across self-contained subpackages;
- release hygiene (no packaged caches/bytecode, pinned environment, GitHub-size guard, publication-reference counts, and release metadata structure).

### 2. Release audit used by CI

```bash
python scripts/reproduce.py release-audit --jobs 1
```

This adds to the quick verification:

- exact reproduction of all 336 architecture-specific selected-source entries;
- one-realization light-method smoke comparison;
- a representative heavy-method smoke comparison at MLR+RF / 12-diet budget (4 heavy update cells);
- reconstruction of Supplementary Table S5 source tables;
- prediction-level regeneration and exact 64-row verification of Figure 5 RN/TNE derived endpoints;
- the complete one-command publication-artifact build.

The heavy smoke test is intentionally representative rather than exhaustive. The complete **112,896-cell** architecture-specific extension remains available through `full-extension`.

### 3. Full analyses beyond the CI gate

```bash
python scripts/reproduce.py primary-factorial
python scripts/reproduce.py full-extension --jobs 8
```

The prediction-level `derived-endpoints` verification is now part of `release-audit`. Its optimized verification path still recomputes all required 504-allocation × 4-architecture × 4-budget selected-procedure updates, but avoids materializing a large per-source prediction file and omits the unrelated 10,000-resample AUBC bootstrap. The scientific architecture-robustness analysis retains that bootstrap by default.

`primary-factorial` and `full-extension` are computationally heavier full analyses. The reported realization-level matrices are included so manuscript values remain inspectable without rerunning the most expensive analyses.

## One-command publication-artifact build

```bash
python scripts/reproduce.py publication --jobs 1
```

The command creates `generated/` containing:

- `rebuilt_figures/`: all 15 figures regenerated from machine-readable figure-source tables as PNG and EPS;
- `locked_submission_artwork/`: exact locked 15 PNG + 15 EPS submission artwork, re-materialized and SHA-256 checked;
- `tables/`: three locked publication-source Excel workbooks, re-materialized and SHA-256 checked;
- `captions/`: main and supplementary figure-caption files;
- `PUBLICATION_ARTIFACT_HASH_CHECK.csv` and `PUBLICATION_BUILD_REPORT.txt`.

Statistical content and publication formatting are deliberately separated. Figure statistics can be replotted from machine-readable source tables; exact final typography, spacing, and styled Excel layout are preserved as publication-layer reference assets and byte-verified rather than presented as statistical computations.

## Analysis conventions

The literature dataset contains **673 development records from 122 Study_ID groups** plus **233 confirmation records from 48 previously unseen Study_ID groups**. After model selection, the selected model specification was refitted on all 906 records before independent-target evaluation.

The growing-pig target system contains **546 pig-period records from 91 source-specific diets**, six records per diet. Budgets 3, 6, 9, and 12 are prespecified; 15, 18, and 21 belong to the extended-budget analysis.

The 504 allocations are fixed-data robustness scenarios (84 blocks × 6 rotations) balancing which of six pig records supplies the revealed FN/UN values within a selected diet. They are not 504 independent biological replicates.

**P2-PAM** represents the candidate-diet space from premeasurement descriptors. **Response-surface-guided sampling (RSGS)** uses only predictions generated before target FN/UN labels are revealed. **CG-HPP** uses revealed target FN/UN values during updating. Detailed variables and execution order are documented in `docs/METHODS_MAPPING.md`.

## Reproducibility boundary

Downstream target-domain analyses after selection of the MLR+RF starting model are executable from this repository. The P0–P3 predictor-information comparison and ten-model architecture-selection stage are represented by retained machine-readable results and verification materials. This repository does **not** claim end-to-end retraining of every historical candidate model from raw model-development inputs. See `docs/REPRODUCIBILITY_NOTES.md`.

## Historical-source provenance

`data/provenance/HISTORICAL_170_STUDY_PROVENANCE.csv` maps all 170 historical Study_ID values. Source resolution follows a documented hierarchy: V44 reviewer-ready Study Index first, verified v35 identifier fallback second, and publisher/journal verification for the six identities not resolved by those identifier fields. No DOI was inferred from Study_ID text.

`REVIEW_FLAG_records` in that table is a **record/field-level QA count**, not a statement that the study identity is unverified. See `data/provenance/README.md`.

## Licensing, data rights, and citation

Original software code is released under the MIT License (`LICENSE`). That license does **not** automatically apply to data matrices, literature-derived values, publication artwork, or third-party source material; see `DATA_RIGHTS.md`.

`CITATION.cff` contains release metadata but intentionally omits a repository URL and archive DOI until those identifiers actually exist. After the public GitHub release and permanent archive are created, add the real repository URL and version DOI to `CITATION.cff` and the manuscript. Do not invent placeholders.

See `GITHUB_RELEASE_CHECKLIST.md` for the final release sequence and `CHANGELOG.md` for package-hardening history.
