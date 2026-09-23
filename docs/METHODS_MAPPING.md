# Analysis-to-code map

| Manuscript analysis | Primary inputs | Executable code or retained evidence | Main outputs |
|---|---|---|---|
| Dataset roles and study splits | `data/analysis_matrices/historical_906.csv` | `scripts/verify_reproducibility.py` | 673/122 development; 233/48 confirmation |
| P0–P3 predictor-information comparison | 906-record matrix | retained machine-readable results | `results/predictor_information/` |
| Ten-family model selection | 673-record development split | retained ranking; selected-route refit is executable | `results/model_selection/NESTED_10_MODEL_RANKING.csv` |
| MLR+RF refit, confirmation, and all-906 refit | 906- and 546-record matrices | `analysis/route/run_nested_primary_route.py` | confirmation and target predictions |
| Four starting architectures | packaged pre-update predictions for MLR, RF, TabM, and MLR+RF | architecture-specific RSGS and updating scripts | `data/pre_update_predictions/four_arch_*` |
| Target sampling × updating, budgets 3/6/9/12 | 546-record target system, pre-update predictions, 504 ledger | `run_546_sampling_updating_factorial.py` | `results/target_domain/FACTORIAL_*` |
| P2-PAM representative sampling | pre-measurement P2-tier information | target factorial route | selection ledgers and Table S16 inputs |
| Response-surface-guided sampling (RSGS) | pre-update FN/UN predictions | target factorial and architecture-specific selector | selected-source ledgers |
| Balanced outcome-record allocation analysis | 504 allocation ledger | target factorial and architecture-specific extension | realization-level and percentile summaries |
| Architecture-specific extension, budgets 3–21 | four-architecture pre-update predictions | `analysis/architecture_specific/scripts/` | `results/architecture_robustness/` |
| Gestating-sow cross-physiology evaluation (internal code SOW80) | SOW80 and four-architecture predictions | `analysis/route/run_sow80_evaluation.py` | `results/target_domain/SOW80_*` |
| Classical DA/transfer benchmark | 906, 546, and pre-update predictions | `analysis/classical_da_transfer_benchmark/code/` | `results/domain_adaptation/` |

## P2 model inputs and P2-PAM sampling representation

The fitted P2 prediction model and the PAM sampling step use related but nonidentical pre-measurement representations. The fitted model uses the specified P2 predictive variables. PAM additionally uses routinely available pre-measurement descriptors to represent candidate-diet geometry; in the 546-record target system this includes dietary ash because starch is structurally unavailable for candidate-space representation. Dietary ash is used only for measurement allocation and does not replace starch in the fitted prediction model.

RSGS uses only predictions generated before target FN/UN values are revealed. CG-HPP is applied after the selected target FN/UN values become available.

| Module | Information available before target measurement | Uses measured target FN/UN? |
|---|---|---:|
| Fitted P2 prediction model | specified P2 predictive variables | No |
| P2-PAM | P2-tier pre-measurement descriptors used to represent candidate-diet space | No |
| RSGS | pre-update FN/UN predictions for candidate diets | No |
| CG-HPP | selected target records and measured FN/UN | Yes |
