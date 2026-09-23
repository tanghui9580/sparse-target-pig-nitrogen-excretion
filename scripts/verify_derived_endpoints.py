#!/usr/bin/env python3
"""Rebuild and verify Figure 5 mass-balance-derived RN/TNE R² values.

RN and TNE are not independently fitted endpoints.  This check regenerates the
selected-procedure FN/UN predictions for the 546 target system and the
cross-physiology gestating-sow evaluation, derives

    RN  = N intake - FN - UN
    TNE = FN + UN

and verifies the published Figure5D_DerivedEndpointR2.csv values.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def derive_546(pred_path: Path) -> pd.DataFrame:
    pred = pd.read_csv(pred_path)
    target = pd.read_csv(ROOT/'data/analysis_matrices/target_546_animal.csv')
    source = (target.groupby('Source_unit', as_index=False).mean(numeric_only=True)[
        ['Source_unit','Analysis_N_Intake_g_d','Analysis_Retained_N_g_d','Analysis_Total_N_Excretion_g_d']
    ])
    d = pred.merge(source, on='Source_unit', validate='many_to_one')
    d['RN_pred'] = d.Analysis_N_Intake_g_d - d.FN_predicted - d.UN_predicted
    d['TNE_pred'] = d.FN_predicted + d.UN_predicted
    rows=[]
    for (arch,budget,realization), g in d.groupby(['Architecture','Budget','Realization']):
        rows.append({'architecture':arch,'domain':'546','measurement_budget':budget,'endpoint':'RN',
                     'value':float(r2_score(g.Analysis_Retained_N_g_d,g.RN_pred))})
        rows.append({'architecture':arch,'domain':'546','measurement_budget':budget,'endpoint':'TNE',
                     'value':float(r2_score(g.Analysis_Total_N_Excretion_g_d,g.TNE_pred))})
    return pd.DataFrame(rows).groupby(['architecture','domain','measurement_budget','endpoint'],as_index=False).value.mean()


def derive_sow(pred_path: Path) -> pd.DataFrame:
    pred = pd.read_csv(pred_path)
    sow = pd.read_csv(ROOT/'data/analysis_matrices/sow80.csv')
    d = pred.merge(sow[['Record_ID','Analysis_N_Intake_g_d','Analysis_Retained_N_g_d','Analysis_Total_N_Excretion_g_d']],
                   on='Record_ID', validate='many_to_one')
    d['RN_pred'] = d.Analysis_N_Intake_g_d - d.FN_predicted - d.UN_predicted
    d['TNE_pred'] = d.FN_predicted + d.UN_predicted
    rows=[]
    for (arch,budget,realization), g in d.groupby(['Architecture','Budget','Realization']):
        rows.append({'architecture':arch,'domain':'SOW80','measurement_budget':budget,'endpoint':'RN',
                     'value':float(r2_score(g.Analysis_Retained_N_g_d,g.RN_pred))})
        rows.append({'architecture':arch,'domain':'SOW80','measurement_budget':budget,'endpoint':'TNE',
                     'value':float(r2_score(g.Analysis_Total_N_Excretion_g_d,g.TNE_pred))})
    return pd.DataFrame(rows).groupby(['architecture','domain','measurement_budget','endpoint'],as_index=False).value.mean()


def main():
    # Verify observed mass-balance fields before deriving predictions.
    for rel in ['data/analysis_matrices/target_546_animal.csv','data/analysis_matrices/sow80.csv']:
        d=pd.read_csv(ROOT/rel)
        rn_err=(d.Analysis_N_Intake_g_d-d.Analysis_Fecal_N_g_d-d.Analysis_Urinary_N_g_d-d.Analysis_Retained_N_g_d).abs().max()
        tne_err=(d.Analysis_Fecal_N_g_d+d.Analysis_Urinary_N_g_d-d.Analysis_Total_N_Excretion_g_d).abs().max()
        if float(rn_err)>1e-5 or float(tne_err)>1e-5:
            raise AssertionError((rel,float(rn_err),float(tne_err)))

    work_root=ROOT/'work'; work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='aia_derived_', dir=str(work_root)) as td:
        td=Path(td); a546=td/'arch546'; asow=td/'sow80'; a546.mkdir(); asow.mkdir(); derived546=td/'derived_546.csv'
        run([sys.executable,ROOT/'analysis/route/run_546_architecture_robustness.py',
             '--predictions-546',ROOT/'data/pre_update_predictions/four_arch_target546_predictions.csv',
             '--primary-oof',ROOT/'data/pre_update_predictions/mlr_rf_673_oof.csv',
             '--tabm-oof',ROOT/'data/pre_update_predictions/tabm_673_oof.csv',
             '--rotation-ledger',ROOT/'data/design/BALANCED_504_ANIMAL_ROTATION_LEDGER.csv',
             '--structured-ledger',ROOT/'data/design/STRUCTURED_SELECTION_LEDGERS.csv',
             '--existing-factorial',ROOT/'results/target_domain/FACTORIAL_504_REALIZATION_METRICS.csv.gz',
             '--target-spec',ROOT/'config/target_updating_specification.json',
             '--output-dir',a546,
             '--skip-bootstrap',
             '--derived-endpoint-summary',derived546])
        run([sys.executable,ROOT/'analysis/route/run_sow80_evaluation.py',
             '--matrix-906',ROOT/'data/analysis_matrices/historical_906.csv',
             '--sow-predictions',ROOT/'data/pre_update_predictions/four_arch_sow80_predictions.csv',
             '--primary-oof',ROOT/'data/pre_update_predictions/mlr_rf_673_oof.csv',
             '--tabm-oof',ROOT/'data/pre_update_predictions/tabm_673_oof.csv',
             '--model-spec',ROOT/'config/primary_model_specification.json',
             '--target-spec',ROOT/'config/target_updating_specification.json',
             '--output-dir',asow])
        d546=pd.read_csv(derived546); d546['domain']='546'
        rebuilt=pd.concat([
            d546,
            derive_sow(asow/'SOW80_SELECTED_PROCEDURE_PREDICTIONS.csv.gz')
        ],ignore_index=True)

    reported=pd.read_csv(ROOT/'figure_sources/Figure5D_DerivedEndpointR2.csv')
    keys=['architecture','domain','measurement_budget','endpoint']
    m=reported[keys+['value']].merge(rebuilt,on=keys,suffixes=('_reported','_rebuilt'),validate='one_to_one')
    if len(m)!=len(reported) or len(m)!=64:
        raise AssertionError((len(reported),len(rebuilt),len(m)))
    maxdiff=float((m.value_reported-m.value_rebuilt).abs().max())
    if maxdiff>1e-12:
        bad=m.loc[(m.value_reported-m.value_rebuilt).abs().idxmax()].to_dict()
        raise AssertionError((maxdiff,bad))
    print(f'PASS: Figure 5 RN/TNE derived endpoints rebuilt exactly; 64 rows, max abs difference={maxdiff:.3g}')

if __name__=='__main__':
    main()
