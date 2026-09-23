#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess, sys

ROOT=Path(__file__).resolve().parents[1]

def close(a,b,tol=5e-4):
    if not np.isclose(float(a),float(b),atol=tol,rtol=0):
        raise AssertionError(f"Expected {b}, got {a}")

hist=pd.read_csv(ROOT/'data/analysis_matrices/historical_906.csv')
target=pd.read_csv(ROOT/'data/analysis_matrices/target_546_animal.csv')
sow=pd.read_csv(ROOT/'data/analysis_matrices/sow80.csv')
assert len(hist)==906
assert (hist.Analysis_Split=='development').sum()==673
assert (hist.Analysis_Split=='confirmation').sum()==233
assert hist.loc[hist.Analysis_Split.eq('development'),'Study_ID'].nunique()==122
assert hist.loc[hist.Analysis_Split.eq('confirmation'),'Study_ID'].nunique()==48
assert len(target)==546 and target.Source_unit.nunique()==91
assert target.groupby('Source_unit').size().eq(6).all()
assert len(sow)==80 and sow.Study_ID.nunique()==8

# Nitrogen mass balance in the target analysis matrix.
mb=target.Analysis_N_Intake_g_d-target.Analysis_Fecal_N_g_d-target.Analysis_Urinary_N_g_d-target.Analysis_Retained_N_g_d
assert float(mb.abs().max()) < 1e-5

rot=pd.read_csv(ROOT/'data/design/BALANCED_504_ANIMAL_ROTATION_LEDGER.csv')
assert rot.Realization.nunique()==504
assert set(rot.Rotation.unique())==set(range(6))
assert rot.groupby(['Source_unit','Pig_Ordinal']).size().nunique()==1
assert int(rot.groupby(['Source_unit','Pig_Ordinal']).size().iloc[0])==84

rank=pd.read_csv(ROOT/'results/model_selection/NESTED_10_MODEL_RANKING.csv')
assert rank.iloc[0].Model=='MLR+RF'
close(rank.iloc[0].Primary_NRMSE,0.523065,1e-6)

cell=pd.read_csv(ROOT/'results/target_domain/FACTORIAL_CELL_SUMMARY.csv')
hpp=cell[(cell.Sampling=='RSGS')&(cell.Method=='CG-HPP')].set_index('Budget')
for b,v in {3:0.626688,6:0.548895,9:0.534391,12:0.518991}.items(): close(hpp.loc[b,'Primary_mean'],v,1e-6)
close(hpp.loc[12,'FN_RMSE_mean'],1.238048,1e-6)
close(hpp.loc[12,'FN_R2_mean'],0.846686,1e-6)
close(hpp.loc[12,'UN_RMSE_mean'],2.817830,1e-6)
close(hpp.loc[12,'UN_R2_mean'],0.523948,1e-6)

arch=pd.read_csv(ROOT/'results/architecture_robustness/FULL_MATRIX_AUBC_SUMMARY.csv')
# Prespecified-budget AUBCs are reported in the manuscript-facing figure sources; verify key MLR+RF extension is present.
assert {'MLR','RF','TabM','MLR+RF'}.issubset(set(arch.Architecture))
assert len(arch)==32

sowres=pd.read_csv(ROOT/'results/target_domain/SOW80_SELECTED_PROCEDURE_SUMMARY.csv')
r=sowres[(sowres.Architecture=='MLR+RF')&(sowres.Method=='CG-HPP')&(sowres.Budget==12)].iloc[0]
close(r.Primary_mean,0.596487,1e-6); close(r.FN_RMSE_mean,1.149744,1e-6); close(r.UN_RMSE_mean,4.602975,1e-6)

da=pd.read_csv(ROOT/'results/domain_adaptation/TABLE_S5_PANEL_B_SAME_LABEL.csv')
assert len(da)==8
r=da[(da.Architecture=='MLR+RF')&(da.Method=='CG-HPP')].iloc[0]
close(r.AUBC_3_12,0.552042,1e-6)
r=da[(da.Architecture=='MLR+RF')&(da.Method=='Two-stage TrAdaBoost.R2')].iloc[0]
close(r.AUBC_3_12,0.797213,1e-6)

# 112,896 architecture-specific cells are present in the packaged compressed result matrix.
full=pd.read_csv(ROOT/'analysis/architecture_specific/results/FULL_MATRIX_504_METRICS.csv.gz')
assert len(full)==112896

# Verify the P0–P3 information-tier evidence and six-family bootstrap contract.
subprocess.run([sys.executable, ROOT/'scripts/verify_predictor_information.py'], check=True, cwd=ROOT)

# Quick Figure 5 RN/TNE source-contract check. A full prediction-level rebuild remains available via
# `python scripts/reproduce.py derived-endpoints`.
derived=pd.read_csv(ROOT/'figure_sources/Figure5D_DerivedEndpointR2.csv')
keys=['architecture','domain','measurement_budget','endpoint']
assert len(derived)==64 and len(derived[keys].drop_duplicates())==64
r=derived[(derived.architecture=='MLR+RF')&(derived.domain=='546')&(derived.measurement_budget==12)&(derived.endpoint=='RN')].iloc[0]
close(r.value,0.938,1e-3)
r=derived[(derived.architecture=='MLR+RF')&(derived.domain=='546')&(derived.measurement_budget==12)&(derived.endpoint=='TNE')].iloc[0]
close(r.value,0.794,1e-3)


# Public-facing reproducibility contracts introduced by the final audit.
sow_summary = pd.read_csv(ROOT/'results/target_domain/SOW80_SELECTED_PROCEDURE_SUMMARY.csv')
sow_aubc = pd.read_csv(ROOT/'results/target_domain/SOW80_AUBC_SUMMARY.csv')
assert 'formal_selected_updater' not in set(sow_summary.Role.astype(str))
assert 'formal_selected_updater' not in set(sow_aubc.Role.astype(str))
assert 'prespecified_updater' not in set(sow_aubc.Role.astype(str))
assert 'selected_updater' in set(sow_summary.Role.astype(str))
assert 'selected_updater' in set(sow_aubc.Role.astype(str))

s2 = pd.read_csv(ROOT/'figure_sources/FigureS02_DomainShift.csv')
assert not s2.Variable.isin(['Fecal N, g/d', 'Urinary N, g/d']).any()
assert s2.Variable.nunique() == 11

fig_manifest = pd.read_csv(ROOT/'figure_sources/FIGURE_SOURCE_MANIFEST.csv')
assert fig_manifest.active_script.eq('scripts/build_publication_figures.py').all()
assert fig_manifest.rebuild_mode.str.contains('reproducible plotting script').all()

print('PASS: paper reproducibility package')
print('  historical matrix: 906 = 673 development + 233 confirmation')
print('  target system: 546 = 91 diets x 6 pigs')
print('  balanced allocations: 504; every pig ordinal used 84 times per diet')
print('  selected architecture: MLR+RF')
print('  P0–P3 information-tier evidence and six-family bootstrap contract verified')
print('  primary CG-HPP budget metrics and SOW80 metrics verified')
print('  architecture-specific extension: 112,896 cells')
print('  Table S5 domain-adaptation benchmark verified')
print('  Figure 5 RN/TNE source contract verified; full prediction-level rebuild available via derived-endpoints')
