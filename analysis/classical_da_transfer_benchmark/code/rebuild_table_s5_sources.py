#!/usr/bin/env python3
"""Rebuild the Table S5 source CSVs from the packaged benchmark results.

This script reconstructs the reported Table S5 summaries from the packaged
benchmark result tables and the paired 504-allocation data.
"""
from pathlib import Path
import argparse
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=ROOT/'work'/'table_s5_rebuild')
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Panel A: B3 matched-unmeasured, zero-target-Y UDA plus labeled CG-HPP anchor.
    uda = pd.read_csv(ROOT/'results'/'uda'/'UDA_MATCHED_UNMEASURED_BY_BUDGET.csv')
    a = uda[uda['Budget'].eq(3)].copy()
    # Use the order reported in Table S5.
    order = ['NoAdapt','KMM','RuLSIF','TCA','CORAL']
    arch_order = ['MLR','RF','TabM','MLR+RF']
    a['Method'] = pd.Categorical(a['Method'], order, ordered=True)
    a['Architecture'] = pd.Categorical(a['Architecture'], arch_order, ordered=True)
    a = a.sort_values(['Architecture','Method'])
    a = a[['Architecture','Method','FN_RMSE','UN_RMSE','Primary_NRMSE','n_eval']]
    a.columns = ['Architecture','Method','FN_RMSE','UN_RMSE','Composite_NRMSE','n_eval_diets']
    a.insert(2,'Target_Y_labels',0)
    a['Information_layer'] = 'Unlabeled target-X adaptation'

    bsum = pd.read_csv(ROOT/'results'/'supervised_transfer'/'CROSSARCH_TRADABOOSTR2_VS_CGHPP_BUDGET_SUMMARY.csv')
    cghpp_b3 = bsum[bsum['Budget'].eq(3)][['Architecture','FN_RMSE_CGHPP','UN_RMSE_CGHPP','CGHPP_Primary']].copy()
    cghpp_b3.columns = ['Architecture','FN_RMSE','UN_RMSE','Composite_NRMSE']
    cghpp_b3['Method'] = 'CG-HPP'
    cghpp_b3['Target_Y_labels'] = 3
    cghpp_b3['n_eval_diets'] = 88
    cghpp_b3['Information_layer'] = 'Sparse labeled target updating'
    cghpp_b3 = cghpp_b3[['Architecture','Method','Target_Y_labels','FN_RMSE','UN_RMSE','Composite_NRMSE','n_eval_diets','Information_layer']]
    a = pd.concat([a, cghpp_b3], ignore_index=True)
    a['Architecture'] = pd.Categorical(a['Architecture'], arch_order, ordered=True)
    method_order = ['NoAdapt','KMM','RuLSIF','TCA','CORAL','CG-HPP']
    a['Method'] = pd.Categorical(a['Method'], method_order, ordered=True)
    a = a.sort_values(['Architecture','Method']).reset_index(drop=True)
    a['Architecture'] = a['Architecture'].astype(str)
    a['Method'] = a['Method'].astype(str)
    a.to_csv(args.outdir/'TABLE_S5_PANEL_A_B3_MATCHED.csv', index=False)

    # Panel B: same-label paired comparison. Budget means come from 504 allocations;
    # AUBC is the paired-allocation mean; lower-scenario counts are computed paired.
    paired = pd.read_csv(ROOT/'results'/'supervised_transfer'/'CROSSARCH_TRADABOOSTR2_VS_CGHPP_AUBC_504.csv')
    rows=[]
    for arch in arch_order:
        g = bsum[bsum['Architecture'].eq(arch)].set_index('Budget')
        p = paired[paired['Architecture'].eq(arch)]
        tr_aubc = float(p['TrAda_AUBC'].mean())
        cg_aubc = float(p['CGHPP_AUBC'].mean())
        rows.append(dict(Architecture=arch, Method='Two-stage TrAdaBoost.R2',
                         B3=g.loc[3,'TrAda_Primary'], B6=g.loc[6,'TrAda_Primary'],
                         B9=g.loc[9,'TrAda_Primary'], B12=g.loc[12,'TrAda_Primary'],
                         AUBC_3_12=tr_aubc, Delta_AUBC_vs_CGHPP=tr_aubc-cg_aubc,
                         Lower_AUBC_scenarios_vs_paired=int((p['TrAda_AUBC'] < p['CGHPP_AUBC']).sum())))
        rows.append(dict(Architecture=arch, Method='CG-HPP',
                         B3=g.loc[3,'CGHPP_Primary'], B6=g.loc[6,'CGHPP_Primary'],
                         B9=g.loc[9,'CGHPP_Primary'], B12=g.loc[12,'CGHPP_Primary'],
                         AUBC_3_12=cg_aubc, Delta_AUBC_vs_CGHPP=0.0,
                         Lower_AUBC_scenarios_vs_paired=int((p['CGHPP_AUBC'] < p['TrAda_AUBC']).sum())))
    pd.DataFrame(rows).to_csv(args.outdir/'TABLE_S5_PANEL_B_SAME_LABEL.csv', index=False)
    print(f'Wrote Table S5 source CSVs to {args.outdir}')

if __name__ == '__main__':
    main()
