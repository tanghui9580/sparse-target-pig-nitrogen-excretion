#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'analysis/architecture_specific/results/FULL_MATRIX_504_METRICS.csv.gz'

def summarize(metrics:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    m=pd.read_csv(metrics)
    keys=['Realization','Architecture','Budget','Method']
    if len(m)!=112896 or len(m[keys].drop_duplicates())!=112896:
        raise AssertionError(f'Expected 112,896 unique cells; got {len(m)} rows')
    counts=m.groupby(['Architecture','Budget','Method']).Realization.nunique()
    if not (counts.eq(504).all() and len(counts)==224):
        raise AssertionError('Expected 224 architecture×budget×method cells, each with 504 allocations')
    agg=m.groupby(['Architecture','Method','Budget']).agg(
        n_realizations=('Realization','nunique'),
        Primary_mean=('Primary_NRMSE','mean'), Primary_SD=('Primary_NRMSE','std'),
        Primary_q025=('Primary_NRMSE',lambda x:x.quantile(.025)), Primary_q975=('Primary_NRMSE',lambda x:x.quantile(.975)),
        FN_RMSE_mean=('FN_RMSE','mean'), FN_R2_mean=('FN_R2','mean'),
        UN_RMSE_mean=('UN_RMSE','mean'), UN_R2_mean=('UN_R2','mean')).reset_index()
    agg.to_csv(out/'FULL_MATRIX_CELL_SUMMARY.csv',index=False)
    rows=[]
    for (arch,meth,r),g in m.groupby(['Architecture','Method','Realization']):
        g=g.sort_values('Budget')
        main=g[g.Budget.isin([3,6,9,12])]
        ext=g[g.Budget.isin([3,6,9,12,15,18,21])]
        rows.append({'Architecture':arch,'Method':meth,'Realization':int(r),
                     'AUBC_3_12':float(np.trapezoid(main.Primary_NRMSE,main.Budget)/9.0),
                     'AUBC_3_21':float(np.trapezoid(ext.Primary_NRMSE,ext.Budget)/18.0)})
    a=pd.DataFrame(rows)
    a.to_csv(out/'FULL_MATRIX_AUBC_BY_REALIZATION.csv',index=False)
    s=a.groupby(['Architecture','Method']).agg(
        n_realizations=('Realization','size'),
        AUBC_3_12_mean=('AUBC_3_12','mean'), AUBC_3_12_q025=('AUBC_3_12',lambda x:x.quantile(.025)), AUBC_3_12_q975=('AUBC_3_12',lambda x:x.quantile(.975)),
        AUBC_3_21_mean=('AUBC_3_21','mean'), AUBC_3_21_q025=('AUBC_3_21',lambda x:x.quantile(.025)), AUBC_3_21_q975=('AUBC_3_21',lambda x:x.quantile(.975))).reset_index()
    s.to_csv(out/'FULL_MATRIX_AUBC_SUMMARY.csv',index=False)
    c=s[s.Method.eq('CG-HPP')].set_index('Architecture').AUBC_3_12_mean
    expected={'MLR':0.587939,'RF':0.577197,'TabM':0.572708,'MLR+RF':0.552042}
    for k,v in expected.items():
        if not np.isclose(c[k],v,atol=1e-6,rtol=0): raise AssertionError((k,c[k],v))
    print('PASS: reported summaries rebuilt')
    print(s[s.Method.eq('CG-HPP')][['Architecture','AUBC_3_12_mean','AUBC_3_21_mean']].to_string(index=False))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--metrics',type=Path,default=DEFAULT); p.add_argument('--out',type=Path,default=ROOT/'work/summaries'); a=p.parse_args(); summarize(a.metrics,a.out)
if __name__=='__main__': main()
