# Manuscript, figure, table, and numerical-source alignment

This file records the publication-facing asset alignment used for the 2026-09-21 content-synchronized submission package. The final EPS/PNG artwork is retained in-package under `publication_reference/`; this repository contains the numerical sources used to check the plotted values.

## Main figures

| Figure | Numerical source(s) |
|---|---|
| Figure 1 | `figure_sources/Figure1_nodes.csv`; `figure_sources/Figure1_edges.csv` |
| Figure 2 | `figure_sources/Figure2A_Benchmark.csv`; `Figure2B_Transport.csv`; `Figure2C_UN.csv` |
| Figure 3 | `figure_sources/Figure3_budget_recovery_3_21.csv` |
| Figure 4 | `figure_sources/Figure4A_selection_all_budgets.csv`; `Figure4B_sampling_curves.csv`; `Figure4C_updating_curves.csv` |
| Figure 5 | `figure_sources/Figure5A_546_AUBC.csv`; `Figure5B_SOW80_AUBC.csv`; `Figure5C_EndpointR2.csv`; `Figure5D_DerivedEndpointR2.csv` |

Figure 5 RN/TNE values in `Figure5D_DerivedEndpointR2.csv` are mass-balance-derived, not separately fitted. They are rebuilt from regenerated selected-procedure FN/UN predictions by `scripts/verify_derived_endpoints.py` using RN = N intake - FN - UN and TNE = FN + UN.

## Supplementary figures

| Figure | Numerical source(s) |
|---|---|
| Figure S1 | `figure_sources/FigureS01_DomainPCA.csv` |
| Figure S2 | `figure_sources/FigureS02_DomainShift.csv` |
| Figure S3 | `figure_sources/FigureS03_ExternalAgreement.csv` |
| Figure S4 | `figure_sources/FigureS04_preupdate_ingredient_heterogeneity.csv` |
| Figure S5 | `figure_sources/FigureS05_Selection_AllBudgets.csv` |
| Figure S6 | `figure_sources/FigureS06_SamplingCurves.csv` |
| Figure S7 | `figure_sources/FigureS07_UpdatingCurves.csv` |
| Figure S8 | `figure_sources/FigureS08_MeasurementDepth.csv` (Mean NRMSE, FN R², and UN R² panels) |
| Figures S9–S10 | `figure_sources/FigureS09_S10_full_matrix_summary.csv`; `FigureS09_S10_full_matrix_aubc.csv` |

## Tables

Main Tables 1–4 and Supplementary Tables S1–S17 report the same packaged numerical results referenced in `docs/RESULTS_MAP.md`. Supplementary Table S8 is the tabular authority for the Figure S8 measurement-depth analysis. Supplementary Table S5 is reproduced by:

```bash
python scripts/reproduce.py classical-benchmark
```

The publication package changes made during asset synchronization were confined to reader-facing labels, caption-to-panel alignment, and table panel/index headings. Analysis code, input data, selected measurement sets, prediction values, and reported statistics were not changed.
