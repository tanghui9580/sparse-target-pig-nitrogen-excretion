#!/usr/bin/env python3
"""Verify intentionally duplicated canonical inputs/results have not drifted."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
GROUPS=[
    [
        'data/analysis_matrices/historical_906.csv',
        'analysis/architecture_specific/inputs/historical_906.csv',
        'analysis/classical_da_transfer_benchmark/inputs/historical_906.csv',
    ],
    [
        'data/design/BALANCED_504_ANIMAL_ROTATION_LEDGER.csv',
        'analysis/architecture_specific/inputs/BALANCED_504_ANIMAL_ROTATION_LEDGER.csv',
        'analysis/classical_da_transfer_benchmark/inputs/BALANCED_504_ANIMAL_ROTATION_LEDGER.csv',
    ],
    [
        'data/pre_update_predictions/four_arch_target546_predictions.csv',
        'analysis/architecture_specific/inputs/four_arch_target546_predictions.csv',
    ],
    [
        'data/analysis_matrices/target_546_animal.csv',
        'analysis/classical_da_transfer_benchmark/inputs/target_546_animal.csv',
    ],
    [
        'data/analysis_matrices/target_91_diet_means.csv',
        'analysis/classical_da_transfer_benchmark/inputs/target_91_diet_means.csv',
    ],
    [
        'data/pre_update_predictions/tabm_673_oof.csv',
        'analysis/architecture_specific/inputs/tabm_673_oof.csv',
    ],
    [
        'data/pre_update_predictions/mlr_rf_673_oof.csv',
        'analysis/architecture_specific/inputs/mlr_rf_673_oof.csv',
    ],
    [
        'config/primary_model_specification.json',
        'analysis/architecture_specific/inputs/primary_model_specification.json',
    ],
    [
        'analysis/architecture_specific/results/FULL_MATRIX_504_METRICS.csv.gz',
        'analysis/classical_da_transfer_benchmark/inputs/FULL_MATRIX_504_METRICS.csv.gz',
    ],
    [
        'analysis/architecture_specific/results/FULL_MATRIX_CELL_SUMMARY.csv',
        'results/architecture_robustness/FULL_MATRIX_CELL_SUMMARY.csv',
    ],
    [
        'analysis/architecture_specific/results/FULL_MATRIX_AUBC_SUMMARY.csv',
        'results/architecture_robustness/FULL_MATRIX_AUBC_SUMMARY.csv',
        'figure_sources/FigureS09_S10_full_matrix_aubc.csv',
    ],
    [
        'results/target_domain/FOUR_ARCH_EXTERNAL_ZERO_SHOT_METRICS.csv',
        'analysis/classical_da_transfer_benchmark/inputs/FOUR_ARCH_EXTERNAL_ZERO_SHOT_METRICS.csv',
    ],
]

def sha(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    checked=0
    for group in GROUPS:
        paths=[ROOT/x for x in group]
        missing=[str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
        if missing: raise AssertionError(f'missing duplicate-contract file(s): {missing}')
        hashes=[sha(p) for p in paths]
        if len(set(hashes))!=1:
            raise AssertionError({'group':group,'hashes':dict(zip(group,hashes))})
        checked += len(paths)
    print(f'PASS: duplicate-contract integrity; {len(GROUPS)} groups / {checked} file instances are byte-identical')

if __name__=='__main__': main()
