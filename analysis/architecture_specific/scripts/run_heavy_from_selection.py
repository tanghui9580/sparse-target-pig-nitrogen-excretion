#!/usr/bin/env python3
"""Recompute the heavier target-only and target-weighted update methods from the packaged inputs.

This runner is self-contained inside the architecture_specific_full_matrix folder.
It regenerates target-only Ridge/ElasticNet/RF and target-weighted TargetWeightedFullRF for any
subset of realizations across all four response-surface architectures and all
seven budgets. Architecture affects source selection only for these downstream
comparators; the fitted target-only/full-RF algorithms are otherwise common.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ARCHS=("MLR","RF","TabM","MLR+RF")
BUDGETS=(3,6,9,12,15,18,21)
ENDPOINTS=("FN","UN")
TARGETS={"FN":"Analysis_Fecal_N_g_d","UN":"Analysis_Urinary_N_g_d"}
FULLRF_SEED=20260524
LOCAL_RF_SEED=20260524
TREES=60


def metrics(obs,pred,ref_sd):
    out={}
    for j,e in enumerate(ENDPOINTS):
        y=obs[:,j]; p=pred[:,j]; rr=p-y
        out[f'{e}_RMSE']=float(np.sqrt(np.mean(rr*rr)))
        out[f'{e}_MAE']=float(np.mean(np.abs(rr)))
        out[f'{e}_Bias']=float(np.mean(rr))
        out[f'{e}_R2']=float(r2_score(y,p))
    out['Primary_NRMSE']=float(.5*(out['FN_RMSE']/ref_sd[0]+out['UN_RMSE']/ref_sd[1]))
    return out


def main():
    here=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,default=here/'inputs')
    ap.add_argument('--selection',type=Path,default=here/'selection_reference'/'reproduced_rsgs_selections_3_21.csv')
    ap.add_argument('--out',type=Path,default=here/'checkpoints'/'rev5_self_contained_heavy')
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--end',type=int,default=504,help='exclusive; 0<=start<end<=504')
    ap.add_argument('--jobs',type=int,default=8)
    ap.add_argument('--architectures',nargs='*',choices=ARCHS,default=list(ARCHS))
    ap.add_argument('--budgets',nargs='*',type=int,choices=BUDGETS,default=list(BUDGETS))
    a=ap.parse_args()
    if not (0 <= a.start < a.end <= 504):
        raise ValueError('Require 0 <= start < end <= 504')
    a.out.mkdir(parents=True,exist_ok=True)

    animal=pd.read_csv(a.input_dir/'four_arch_target546_predictions.csv')
    historical=pd.read_csv(a.input_dir/'historical_906.csv')
    rot=pd.read_csv(a.input_dir/'BALANCED_504_ANIMAL_ROTATION_LEDGER.csv')
    sel=pd.read_csv(a.selection)
    with open(a.input_dir/'primary_model_specification.json',encoding='utf-8') as fh:
        model_spec=json.load(fh)

    source_order=sorted(animal.Source_unit.astype(str).unique())
    slookup={s:i for i,s in enumerate(source_order)}
    animal_by_source={s:animal.index[animal.Source_unit.astype(str).eq(s)].tolist() for s in source_order}
    for s in source_order:
        animal_by_source[s].sort(key=lambda i:int(animal.loc[i,'Pig_Ordinal']))
    source=(animal.groupby(['Source_unit','Ingredient_Class'],as_index=False).mean(numeric_only=True)
            .set_index('Source_unit').loc[source_order].reset_index())
    obs_source=np.column_stack([source[TARGETS[e]].to_numpy(float) for e in ENDPOINTS])
    obs_animal=np.column_stack([animal[TARGETS[e]].to_numpy(float) for e in ENDPOINTS])
    ref_sd=np.array([source[TARGETS[e]].std(ddof=1) for e in ENDPOINTS],float)
    animal_idx_by_record={str(r.Record_ID):i for i,r in animal.iterrows()}
    rotation={(int(r.Realization),str(r.Source_unit)):animal_idx_by_record[str(r.Record_ID)] for r in rot.itertuples(index=False)}

    selected={}
    for arch in a.architectures:
        selected[arch]={}
        for b in a.budgets:
            d=sel[(sel.Architecture.eq(arch))&(sel.Budget.eq(b))].sort_values('Rank')
            if len(d)!=b:
                raise ValueError(f'Selection ledger has {len(d)} rows for {arch}, B{b}; expected {b}')
            selected[arch][b]=[slookup[str(x)] for x in d.Source_unit]

    features=list(model_spec['features'])
    imp=SimpleImputer(strategy='median').fit(historical[features].to_numpy(float))
    hx=imp.transform(historical[features].to_numpy(float))
    ax=imp.transform(animal[features].to_numpy(float))
    hy={e:historical[TARGETS[e]].to_numpy(float) for e in ENDPOINTS}
    config={e:model_spec['selected'][e]['config'] for e in ENDPOINTS}

    def task(realization,arch,b):
        sidx=selected[arch][b]
        selected_set=set(sidx)
        eidx=[i for i in range(91) if i not in selected_set]
        cal_rows=np.array([rotation[(realization,source_order[i])] for i in sidx],int)
        preds={k:np.empty((len(eidx),2),float) for k in ('TargetOnlyRidge','TargetOnlyElasticNet','TargetOnlyRF','TargetWeightedFullRF')}
        for j,e in enumerate(ENDPOINTS):
            xcal=ax[cal_rows]; ycal=obs_animal[cal_rows,j]
            scaler=StandardScaler().fit(xcal)
            xcs=scaler.transform(xcal); xas=scaler.transform(ax)
            ridge=Ridge(alpha=1.0).fit(xcs,ycal)
            elastic=ElasticNet(alpha=.1,l1_ratio=.5,max_iter=100000,random_state=LOCAL_RF_SEED).fit(xcs,ycal)
            lrf=RandomForestRegressor(n_estimators=TREES,min_samples_leaf=1,max_features=1.0,
                                      random_state=LOCAL_RF_SEED+j*100,n_jobs=1).fit(xcal,ycal)
            indiv={
                'TargetOnlyRidge':ridge.predict(xas),
                'TargetOnlyElasticNet':elastic.predict(xas),
                'TargetOnlyRF':lrf.predict(ax),
            }
            cx=np.vstack([hx,ax[cal_rows]])
            cy=np.concatenate([hy[e],ycal])
            weights=np.concatenate([np.ones(len(historical)),np.full(b,len(historical)/b)])
            cfg=config[e]
            frf=RandomForestRegressor(
                n_estimators=TREES,
                min_samples_leaf=int(cfg['min_samples_leaf']),
                max_features=float(cfg['max_features']),
                max_depth=cfg.get('max_depth'),
                random_state=FULLRF_SEED+j*100,
                n_jobs=1,
            ).fit(cx,cy,sample_weight=weights)
            indiv['TargetWeightedFullRF']=frf.predict(ax)
            for meth,ip in indiv.items():
                for oi,si in enumerate(eidx):
                    preds[meth][oi,j]=float(np.mean(ip[animal_by_source[source_order[si]]]))
        common={'Realization':realization,'Block':realization//6,'Rotation':realization%6,
                'Architecture':arch,'Geometry':arch,'Budget':b,
                'n_calibration_sources':b,'n_evaluation_sources':len(eidx)}
        return [{**common,'Method':meth,**metrics(obs_source[eidx],p,ref_sd)} for meth,p in preds.items()]

    tasks=[(r,arch,b) for r in range(a.start,a.end) for arch in a.architectures for b in a.budgets]
    nested=Parallel(n_jobs=a.jobs,backend='threading',batch_size=1,verbose=0)(delayed(task)(*t) for t in tasks)
    m=pd.DataFrame([x for g in nested for x in g])
    out=a.out/f'HEAVY_ARCH_SPEC_{a.start:03d}_{a.end-1:03d}.csv.gz'
    m.to_csv(out,index=False,compression='gzip')
    print(out)
    print(f'rows={len(m)}; architectures={len(a.architectures)}; budgets={len(a.budgets)}; realizations={a.end-a.start}')

if __name__=='__main__':
    main()
