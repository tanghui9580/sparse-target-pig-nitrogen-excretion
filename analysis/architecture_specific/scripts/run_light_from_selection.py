#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import r2_score

ARCHS=('MLR','RF','TabM','MLR+RF')
BUDGETS=(3,6,9,12,15,18,21)
ENDPOINTS=('FN','UN')
TARGETS={'FN':'Analysis_Fecal_N_g_d','UN':'Analysis_Urinary_N_g_d'}

def ridge_residual_update(cal_pred,cal_obs,eval_pred,center,scale,sigma,domain_cal=None,domain_eval=None,domains=None):
    scale=max(float(scale),1e-8); sigma=max(float(sigma),1e-8)
    # Historical residual sigma defines the Gaussian residual scale. The fixed
    # prior precisions below are parameterized in residual-variance-scaled units,
    # so the common 1/sigma^2 factor cancels from both sides of the MAP normal
    # equations. sigma is retained explicitly as the prespecified update-scale parameter
    # field; omitting it from the final linear solve is therefore algebraic, not
    # an unintentional loss of weighting.
    zc=(cal_pred-center)/scale; ze=(eval_pred-center)/scale; residual=cal_obs-cal_pred
    if domain_cal is None:
        design=np.column_stack([np.ones(len(zc)),zc]); edesign=np.column_stack([np.ones(len(ze)),ze]); precision=np.diag([1/4,1.0])
    else:
        lookup={d:i for i,d in enumerate(domains)}
        design=np.zeros((len(zc),2+2*len(domains))); edesign=np.zeros((len(ze),2+2*len(domains)))
        design[:,0]=1; design[:,1]=zc; edesign[:,0]=1; edesign[:,1]=ze
        for r,d in enumerate(domain_cal):
            i=lookup[str(d)]; design[r,2+2*i]=1; design[r,2+2*i+1]=zc[r]
        for r,d in enumerate(domain_eval):
            i=lookup[str(d)]; edesign[r,2+2*i]=1; edesign[r,2+2*i+1]=ze[r]
        pv=[1/4,1.0]
        for _ in domains: pv += [1.0,4.0]
        precision=np.diag(pv)
    coef=np.linalg.solve(design.T@design+precision,design.T@residual)
    return eval_pred+edesign@coef

def metric(obs,pred,ref_sd):
    out={}
    for j,e in enumerate(ENDPOINTS):
        y=obs[:,j]; p=pred[:,j]; r=p-y
        out[f'{e}_RMSE']=float(np.sqrt(np.mean(r*r))); out[f'{e}_MAE']=float(np.mean(np.abs(r))); out[f'{e}_Bias']=float(np.mean(r)); out[f'{e}_R2']=float(r2_score(y,p))
    out['Primary_NRMSE']=float(.5*(out['FN_RMSE']/ref_sd[0]+out['UN_RMSE']/ref_sd[1])); return out

def summarize(m, keys):
    return m.groupby(keys).agg(n_realizations=('Realization','size'),Primary_mean=('Primary_NRMSE','mean'),Primary_SD=('Primary_NRMSE','std'),Primary_q025=('Primary_NRMSE',lambda x:x.quantile(.025)),Primary_median=('Primary_NRMSE','median'),Primary_q975=('Primary_NRMSE',lambda x:x.quantile(.975)),FN_RMSE_mean=('FN_RMSE','mean'),FN_R2_mean=('FN_R2','mean'),UN_RMSE_mean=('UN_RMSE','mean'),UN_R2_mean=('UN_R2','mean')).reset_index()

def main():
    here=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,default=here/'inputs')
    ap.add_argument('--selection',type=Path,default=here/'selection_reference'/'reproduced_rsgs_selections_3_21.csv')
    ap.add_argument('--out',type=Path,default=here/'checkpoints'/'rev5_self_contained_light')
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--end',type=int,default=504)
    a=ap.parse_args()
    if not (0 <= a.start < a.end <= 504): raise ValueError('Require 0 <= start < end <= 504')
    a.out.mkdir(parents=True,exist_ok=True)
    animal=pd.read_csv(a.input_dir/'four_arch_target546_predictions.csv')
    poof=pd.read_csv(a.input_dir/'mlr_rf_673_oof.csv')
    toof=pd.read_csv(a.input_dir/'tabm_673_oof.csv')
    rot=pd.read_csv(a.input_dir/'BALANCED_504_ANIMAL_ROTATION_LEDGER.csv')
    sel=pd.read_csv(a.selection)
    source_order=sorted(animal.Source_unit.astype(str).unique()); slookup={s:i for i,s in enumerate(source_order)}
    source=(animal.groupby(['Source_unit','Ingredient_Class'],as_index=False).mean(numeric_only=True).set_index('Source_unit').loc[source_order].reset_index())
    domains=source.Ingredient_Class.astype(str).to_numpy(); all_domains=sorted(set(domains))
    obs_source=np.column_stack([source[TARGETS[e]].to_numpy(float) for e in ENDPOINTS]); obs_animal=np.column_stack([animal[TARGETS[e]].to_numpy(float) for e in ENDPOINTS])
    prior_source={arch:np.column_stack([source[f'{arch}_{e}_pred'].to_numpy(float) for e in ENDPOINTS]) for arch in ARCHS}
    prior_animal={arch:np.column_stack([animal[f'{arch}_{e}_pred'].to_numpy(float) for e in ENDPOINTS]) for arch in ARCHS}
    ref_sd=np.array([source[TARGETS[e]].std(ddof=1) for e in ENDPOINTS],float)
    # OOF hyperparameters
    hyper={}
    for arch in ('MLR','RF','MLR+RF'):
        hyper[arch]={}
        for e in ENDPOINTS:
            y=poof[TARGETS[e]].to_numpy(float); p=poof[f'{arch}_{e}_pred'].to_numpy(float)
            hyper[arch][e]={'center':float(p.mean()),'scale':float(p.std(ddof=1)),'sigma':float((y-p).std(ddof=1))}
    hyper['TabM']={}
    for e in ENDPOINTS:
        d=toof.loc[toof.Endpoint.eq(e)]; y=d.Observed.to_numpy(float); p=d.Predicted.to_numpy(float)
        hyper['TabM'][e]={'center':float(p.mean()),'scale':float(p.std(ddof=1)),'sigma':float((y-p).std(ddof=1))}
    animal_idx={str(r.Record_ID):i for i,r in animal.iterrows()}
    rotation={(int(r.Realization),str(r.Source_unit)):animal_idx[str(r.Record_ID)] for r in rot.itertuples(index=False)}
    selected={arch:{} for arch in ARCHS}
    for arch in ARCHS:
        for b in BUDGETS:
            d=sel[(sel.Architecture.eq(arch))&(sel.Budget.eq(b))].sort_values('Rank'); assert len(d)==b
            selected[arch][b]=[slookup[str(x)] for x in d.Source_unit]
    rows=[]
    for r in range(a.start,a.end):
        for arch in ARCHS:
            for b in BUDGETS:
                sidx=selected[arch][b]; eidx=[i for i in range(91) if i not in set(sidx)]; cal=np.array([rotation[(r,source_order[i])] for i in sidx],int)
                for meth in ('NoUpdate','Mean','Affine','CG-HPP'):
                    pred=np.empty((len(eidx),2))
                    for j,e in enumerate(ENDPOINTS):
                        base=prior_source[arch][eidx,j]
                        if meth=='NoUpdate': pred[:,j]=base
                        elif meth=='Mean': pred[:,j]=base+float(np.mean(obs_animal[cal,j]-prior_animal[arch][cal,j]))
                        else:
                            h=hyper[arch][e]
                            pred[:,j]=ridge_residual_update(prior_animal[arch][cal,j],obs_animal[cal,j],base,h['center'],h['scale'],h['sigma'],domains[sidx] if meth=='CG-HPP' else None,domains[eidx] if meth=='CG-HPP' else None,all_domains if meth=='CG-HPP' else None)
                    rows.append({'Realization':r,'Block':r//6,'Rotation':r%6,'Architecture':arch,'Geometry':arch,'Budget':b,'Method':meth,'n_calibration_sources':b,'n_evaluation_sources':len(eidx),**metric(obs_source[eidx],pred,ref_sd)})
    m=pd.DataFrame(rows); m.to_csv(a.out/'ARCH_SPECIFIC_LIGHT_504_METRICS.csv.gz',index=False,compression='gzip')
    s=summarize(m,['Architecture','Geometry','Method','Budget']); s.to_csv(a.out/'ARCH_SPECIFIC_LIGHT_CELL_SUMMARY.csv',index=False)
    ar=[]
    for (arch,meth,r),g in m.groupby(['Architecture','Method','Realization']):
        g=g.sort_values('Budget'); assert tuple(g.Budget)==BUDGETS
        ar.append({'Architecture':arch,'Method':meth,'Realization':r,'AUBC_3_21':float(np.trapezoid(g.Primary_NRMSE,g.Budget)/18)})
    ad=pd.DataFrame(ar); ad.to_csv(a.out/'ARCH_SPECIFIC_LIGHT_AUBC_BY_REALIZATION.csv',index=False)
    aus=ad.groupby(['Architecture','Method']).agg(n_realizations=('Realization','size'),AUBC_mean=('AUBC_3_21','mean'),AUBC_SD=('AUBC_3_21','std'),AUBC_q025=('AUBC_3_21',lambda x:x.quantile(.025)),AUBC_median=('AUBC_3_21','median'),AUBC_q975=('AUBC_3_21',lambda x:x.quantile(.975))).reset_index(); aus.to_csv(a.out/'ARCH_SPECIFIC_LIGHT_AUBC_SUMMARY.csv',index=False)
    print(s[(s.Method.eq('CG-HPP'))&(s.Budget.isin([15,18,21]))][['Architecture','Budget','Primary_mean','FN_R2_mean','UN_R2_mean']].to_string(index=False))
    print('\nAUBC 3-21 CG-HPP\n',aus[aus.Method.eq('CG-HPP')].sort_values('AUBC_mean').to_string(index=False))
if __name__=='__main__': main()
