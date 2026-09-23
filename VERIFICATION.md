# Verification record — V6.3.2

Package-hardening audit date: 2026-09-22.

This build preserves manuscript-facing numerical results and locked publication assets while strengthening public-repository robustness, provenance, and release checks.

| Check | Expected / status |
|---|---|
| Manifest integrity | every packaged file must match `MANIFEST_SHA256.csv` |
| Historical matrix | 906 = 673 development + 233 confirmation; 170 Study_ID total |
| Historical provenance | 170/170 Study_ID mapped; 170/170 best-available source identifiers; record counts/splits checked automatically |
| Target system | 546 = 91 source-specific diets × 6 pig records |
| Balanced allocation ledger | 504 realizations; each pig ordinal used 84 times per diet |
| P0–P3 / six-family evidence | checked by `scripts/verify_predictor_information.py` |
| Selected architecture / key target metrics | checked by `scripts/verify_reproducibility.py` |
| Architecture-specific retained matrix | 112,896 cells retained |
| Selector reproduction | 336/336 selected-source entries |
| Light smoke | 1 realization, 4 light methods across the packaged architecture/budget scope used by the smoke runner |
| Heavy smoke | representative 4 cells: MLR+RF, budget 12, realization 0, four heavy methods |
| Table S5 benchmark | retained-result verification plus table-source rebuild |
| Figure 5 RN/TNE | source contract in quick verification; prediction-level 64-row rebuild included in `release-audit` |
| Duplicate contracts | canonical files copied into self-contained subpackages must remain byte-identical |
| Release hygiene | no caches/bytecode/100-MiB files; pinned environment; publication-reference counts; Git attributes; release metadata structure |
| Publication rebuild | 15 PNG + 15 EPS regenerated from `figure_sources/` |
| Locked publication assets | 15 PNG + 15 EPS + 3 XLSX + 2 caption files re-materialized and SHA-256 checked |
| Reference environment | Python 3.12; exact package versions in `requirements.txt` / `environment.yml` |
| GitHub CI | `release-audit` on Python 3.12, including prediction-level RN/TNE verification |

## Reproducibility boundary

The P0–P3 predictor-information comparison and ten-model architecture-selection stage are represented by retained machine-readable results and verification materials rather than a claim of end-to-end retraining of every candidate model from raw model-development inputs. Downstream target-domain analyses after model selection remain executable as documented in `README.md`.

The `derived-endpoints` profile is included in the release CI through a dedicated summary path. It still recomputes the selected-procedure updates across all 504 allocations, four architectures, and four prespecified budgets, but avoids materializing the large per-source prediction file. It skips only the unrelated 10,000-resample paired AUBC bootstrap; normal scientific-analysis defaults remain unchanged.

## Public-release identifiers still external to the package

`CITATION.cff` intentionally omits the public repository URL and permanent archive DOI until those identifiers actually exist. After the GitHub release and permanent archive are created, add the real identifiers and synchronize them with the manuscript. Do not fabricate placeholders.
