# Data dictionary

## Analysis populations

| File | Records | Analysis unit | Role |
|---|---:|---|---|
| `historical_906.csv` | 906 | literature treatment/observation record | model development, confirmation, and all-906 refit |
| `target_546_animal.csv` | 546 = 91 diets × 6 pigs | pig-period record | independent growing-pig target analysis |
| `target_91_diet_means.csv` | 91 | source-specific diet | diet-level candidate-space and source-level summaries |
| `sow80.csv` | 80 from 8 studies | treatment/stage record | cross-physiology evaluation |
| `BALANCED_504_ANIMAL_ROTATION_LEDGER.csv` | 45,864 = 91 × 504 | diet × allocation scenario | identifies the pig record supplying the revealed target FN/UN values |

## Main variables

- `Analysis_N_Intake_g_d`: nitrogen intake (g/d) used in the analysis.
- `Analysis_Fecal_N_g_d`: fecal nitrogen, FN (g/d).
- `Analysis_Urinary_N_g_d`: urinary nitrogen, UN (g/d).
- `Analysis_Retained_N_g_d`: retained nitrogen, RN (g/d), derived by nitrogen mass balance where applicable.
- `Analysis_Total_N_Excretion_g_d`: total nitrogen excretion, TNE (g/d), calculated as FN + UN.
- `Source_unit`: source-specific diet identifier in the 546-record target system.
- `Pig_Ordinal`: pig index within each source-specific diet.
- `Realization`: one of the 504 balanced allocation scenarios.
- `Budget`: number of source-specific diets measured for target FN/UN.
- `Primary_NRMSE`: mean of the endpoint-specific normalized RMSE values for FN and UN.

Columns prefixed with `Analysis_` are the finalized harmonized values used by the executable analyses. `Analysis_Split` identifies the development and confirmation subsets of the 906-record literature dataset.
