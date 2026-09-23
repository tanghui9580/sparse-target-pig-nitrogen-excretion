#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

BUDGETS=(3,6,9,12)
ARCHITECTURES=("MLR","RF","TabM","MLR+RF")
ENDPOINTS=("FN","UN")
TARGETS={"FN":"Analysis_Fecal_N_g_d","UN":"Analysis_Urinary_N_g_d"}
BOOTSTRAP_SEED=20260816
BOOTSTRAP_REPS=10000

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def ridge_residual_update(cal_pred, cal_obs, eval_pred, center, scale, sigma, domain_cal=None, domain_eval=None, domains=None):
    scale=max(float(scale),1e-8); sigma=max(float(sigma),1e-8)
    zc=(cal_pred-center)/scale; ze=(eval_pred-center)/scale
    residual=cal_obs-cal_pred
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
    coef=np.linalg.solve(design.T@design+precision, design.T@residual)
    return eval_pred+edesign@coef

def metrics(obs,pred,ref_sd):
    out={}
    for j,e in enumerate(ENDPOINTS):
        y=obs[:,j]; p=pred[:,j]; r=p-y
        out[f'{e}_RMSE']=float(np.sqrt(np.mean(r*r)))
        out[f'{e}_MAE']=float(np.mean(np.abs(r)))
        out[f'{e}_Bias']=float(np.mean(r))
        out[f'{e}_R2']=float(r2_score(y,p))
    out['Primary_NRMSE']=float(0.5*(out['FN_RMSE']/ref_sd[0]+out['UN_RMSE']/ref_sd[1]))
    return out

def oof_hyper(primary_oof,tabm_oof):
    result={}
    for arch in ('MLR','RF','MLR+RF'):
        result[arch]={}
        for e in ENDPOINTS:
            y=primary_oof[TARGETS[e]].to_numpy(float); p=primary_oof[f'{arch}_{e}_pred'].to_numpy(float)
            result[arch][e]={'center':float(np.mean(p)),'scale':float(np.std(p,ddof=1)),'sigma':float(np.std(y-p,ddof=1))}
    result['TabM']={}
    for e in ENDPOINTS:
        d=tabm_oof.loc[tabm_oof.Endpoint.eq(e)].copy()
        assert len(d)==673 and d.Record_ID.astype(str).nunique()==673
        y=d.Observed.to_numpy(float); p=d.Predicted.to_numpy(float)
        result['TabM'][e]={'center':float(np.mean(p)),'scale':float(np.std(p,ddof=1)),'sigma':float(np.std(y-p,ddof=1))}
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--predictions-546',type=Path,required=True)
    ap.add_argument('--primary-oof',type=Path,required=True)
    ap.add_argument('--tabm-oof',type=Path,required=True)
    ap.add_argument('--rotation-ledger',type=Path,required=True)
    ap.add_argument('--structured-ledger',type=Path,required=True)
    ap.add_argument('--existing-factorial',type=Path,required=True)
    ap.add_argument('--target-spec',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--skip-bootstrap',action='store_true',help='Skip the 10,000-resample paired AUBC bootstrap when only prediction-level reconstruction is required.')
    ap.add_argument('--derived-endpoint-summary',type=Path,default=None,help='Fast verification mode: write mean RN/TNE R2 by architecture/budget and skip large per-source prediction/result outputs.')
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    animal=pd.read_csv(a.predictions_546)
    poof=pd.read_csv(a.primary_oof); toof=pd.read_csv(a.tabm_oof)
    rot=pd.read_csv(a.rotation_ledger); sel=pd.read_csv(a.structured_ledger)
    existing=pd.read_csv(a.existing_factorial)
    target_spec=json.loads(a.target_spec.read_text(encoding='utf-8'))
    assert target_spec['status']=='COMPLETE' and target_spec['selected_sampling']=='RSGS' and target_spec['selected_updater']=='CG-HPP'
    assert len(animal)==546 and animal.Source_unit.nunique()==91 and animal.groupby('Source_unit').size().eq(6).all()
    source_order=sorted(animal.Source_unit.astype(str).unique())
    src_index={s:i for i,s in enumerate(source_order)}
    animal_idx_by_record={str(r.Record_ID):i for i,r in animal.iterrows()}
    source=(animal.groupby(['Source_unit','Ingredient_Class'],as_index=False).mean(numeric_only=True).set_index('Source_unit').loc[source_order].reset_index())
    domains=source.Ingredient_Class.astype(str).to_numpy(); all_domains=sorted(set(domains))
    obs_source=np.column_stack([source[TARGETS[e]].to_numpy(float) for e in ENDPOINTS])
    obs_animal=np.column_stack([animal[TARGETS[e]].to_numpy(float) for e in ENDPOINTS])
    ref_sd=np.array([source[TARGETS[e]].std(ddof=1) for e in ENDPOINTS],float)
    prior_source={arch:np.column_stack([source[f'{arch}_{e}_pred'].to_numpy(float) for e in ENDPOINTS]) for arch in ARCHITECTURES}
    prior_animal={arch:np.column_stack([animal[f'{arch}_{e}_pred'].to_numpy(float) for e in ENDPOINTS]) for arch in ARCHITECTURES}
    hyper=oof_hyper(poof,toof)
    selected={}
    ss=sel.loc[sel.Sampling.eq('RSGS')].copy()
    for b in BUDGETS:
        d=ss.loc[ss.Budget.eq(b)].sort_values('Rank')
        assert len(d)==b
        selected[b]=[src_index[str(x)] for x in d.Source_unit]
    rotation={}
    for r in rot.itertuples(index=False):
        rotation[(int(r.Realization),str(r.Source_unit))]=animal_idx_by_record[str(r.Record_ID)]
    metric_rows=[]; pred_rows=[]; derived_rows=[]
    nint_source=source['Analysis_N_Intake_g_d'].to_numpy(float) if a.derived_endpoint_summary else None
    rn_obs_source=source['Analysis_Retained_N_g_d'].to_numpy(float) if a.derived_endpoint_summary else None
    tne_obs_source=source['Analysis_Total_N_Excretion_g_d'].to_numpy(float) if a.derived_endpoint_summary else None
    for realization in range(504):
        for b in BUDGETS:
            sidx=selected[b]; sset=set(sidx); eidx=[i for i in range(len(source_order)) if i not in sset]
            cal_rows=np.array([rotation[(realization,source_order[i])] for i in sidx],int)
            for arch in ARCHITECTURES:
                pred=np.empty((len(eidx),2),float)
                for j,e in enumerate(ENDPOINTS):
                    hp=hyper[arch][e]
                    pred[:,j]=ridge_residual_update(
                        prior_animal[arch][cal_rows,j], obs_animal[cal_rows,j], prior_source[arch][eidx,j],
                        hp['center'],hp['scale'],hp['sigma'], domains[sidx],domains[eidx],all_domains)
                common={'Realization':realization,'Block':realization//6,'Rotation':realization%6,'Sampling':'RSGS','Method':'CG-HPP','Architecture':arch,'Budget':b,'n_calibration_sources':b,'n_evaluation_sources':len(eidx)}
                if a.derived_endpoint_summary:
                    rn_pred=nint_source[eidx]-pred[:,0]-pred[:,1]
                    tne_pred=pred[:,0]+pred[:,1]
                    derived_rows.append({'architecture':arch,'domain':'546','measurement_budget':b,'endpoint':'RN','realization':realization,'value':float(r2_score(rn_obs_source[eidx],rn_pred))})
                    derived_rows.append({'architecture':arch,'domain':'546','measurement_budget':b,'endpoint':'TNE','realization':realization,'value':float(r2_score(tne_obs_source[eidx],tne_pred))})
                    continue
                metric_rows.append({**common,'Role':'selected_procedure_architecture_robustness',**metrics(obs_source[eidx],pred,ref_sd)})
                zero=prior_source[arch][eidx]
                metric_rows.append({**common,'Method':'NoUpdate','Role':'matched_no_update_architecture_control',**metrics(obs_source[eidx],zero,ref_sd)})
                for k,si in enumerate(eidx):
                    pred_rows.append({**common,'Role':'selected_procedure_architecture_robustness','Source_unit':source_order[si],'Ingredient_Class':domains[si],'FN_observed':obs_source[si,0],'UN_observed':obs_source[si,1],'FN_predicted':pred[k,0],'UN_predicted':pred[k,1]})
    if a.derived_endpoint_summary:
        out=(pd.DataFrame(derived_rows).groupby(['architecture','domain','measurement_budget','endpoint'],as_index=False).value.mean())
        a.derived_endpoint_summary.parent.mkdir(parents=True,exist_ok=True)
        out.to_csv(a.derived_endpoint_summary,index=False)
        print(f'Wrote fast derived-endpoint summary: {a.derived_endpoint_summary}; rows={len(out)}')
        return
    m=pd.DataFrame(metric_rows); m.to_csv(a.output_dir/'ARCHITECTURE_ROBUSTNESS_504_METRICS.csv',index=False)
    pd.DataFrame(pred_rows).to_csv(a.output_dir/'ARCHITECTURE_ROBUSTNESS_SOURCE_PREDICTIONS.csv.gz',index=False,compression='gzip')
    formal=m.loc[m.Role.eq('selected_procedure_architecture_robustness')].copy()
    cell=(formal.groupby(['Architecture','Budget']).agg(n_realizations=('Realization','size'),Primary_mean=('Primary_NRMSE','mean'),Primary_SD=('Primary_NRMSE','std'),Primary_q025=('Primary_NRMSE',lambda x:x.quantile(.025)),Primary_median=('Primary_NRMSE','median'),Primary_q975=('Primary_NRMSE',lambda x:x.quantile(.975)),FN_RMSE_mean=('FN_RMSE','mean'),FN_R2_mean=('FN_R2','mean'),UN_RMSE_mean=('UN_RMSE','mean'),UN_R2_mean=('UN_R2','mean')).reset_index())
    cell.to_csv(a.output_dir/'ARCHITECTURE_ROBUSTNESS_CELL_SUMMARY.csv',index=False)
    aubc_rows=[]
    for (arch,r),g in formal.groupby(['Architecture','Realization']):
        g=g.sort_values('Budget'); assert tuple(g.Budget)==BUDGETS
        aubc_rows.append({'Architecture':arch,'Realization':r,'Block':int(r)//6,'AUBC_3_12':float(np.trapezoid(g.Primary_NRMSE,g.Budget)/9)})
    au=pd.DataFrame(aubc_rows); au.to_csv(a.output_dir/'ARCHITECTURE_ROBUSTNESS_AUBC_BY_REALIZATION.csv',index=False)
    aus=(au.groupby('Architecture').agg(n_realizations=('Realization','size'),AUBC_mean=('AUBC_3_12','mean'),AUBC_SD=('AUBC_3_12','std'),AUBC_q025=('AUBC_3_12',lambda x:x.quantile(.025)),AUBC_median=('AUBC_3_12','median'),AUBC_q975=('AUBC_3_12',lambda x:x.quantile(.975))).reset_index().sort_values('AUBC_mean'))
    aus.to_csv(a.output_dir/'ARCHITECTURE_ROBUSTNESS_AUBC_SUMMARY.csv',index=False)
    wide=au.pivot(index='Realization',columns='Architecture',values='AUBC_3_12')
    rows=[]
    if not a.skip_bootstrap:
        rng=np.random.default_rng(BOOTSTRAP_SEED); blocks=np.arange(84)
        for arch in ARCHITECTURES:
            if arch=='MLR+RF': continue
            d=(wide[arch]-wide['MLR+RF']).to_numpy(); observed=float(np.mean(d)); boots=np.empty(BOOTSTRAP_REPS)
            for i in range(BOOTSTRAP_REPS):
                bs=rng.choice(blocks,size=84,replace=True); idx=np.concatenate([np.arange(b*6,b*6+6) for b in bs]); boots[i]=np.mean(d[idx])
            rows.append({'Comparator':arch,'Reference':'MLR+RF','Delta_AUBC_comparator_minus_reference':observed,'CI95_low':float(np.quantile(boots,.025)),'CI95_high':float(np.quantile(boots,.975)),'P_comparator_better_lower_AUBC':float(np.mean(boots<0)),'Bootstrap_unit':'84 blocks (6 rotations kept together)','Bootstrap_reps':BOOTSTRAP_REPS,'Seed':BOOTSTRAP_SEED})
        pd.DataFrame(rows).to_csv(a.output_dir/'ARCHITECTURE_ROBUSTNESS_PAIRED_BLOCK_BOOTSTRAP.csv',index=False)
    # Exact reproduction check against original MLR+RF RSGS CG-HPP cells.
    orig=existing.loc[(existing.Sampling.isin(['RSGS','Model-shape'])) & (existing.Method.isin(['CG-HPP','HPP']))].copy()
    new=formal.loc[formal.Architecture.eq('MLR+RF')].copy()
    keys=['Realization','Budget']; cols=['Primary_NRMSE','FN_RMSE','FN_R2','UN_RMSE','UN_R2']
    z=orig[keys+cols].merge(new[keys+cols],on=keys,suffixes=('_orig','_new'),validate='one_to_one')
    diffs={c:float(np.nanmax(np.abs(z[f'{c}_orig']-z[f'{c}_new']))) for c in cols}
    check_pass=all(v<1e-12 for v in diffs.values())
    check_table=pd.DataFrame([{'Check':'MLR+RF selected-procedure exact reproduction vs original factorial','Status':'PASS' if check_pass else 'FAIL',**{f'max_abs_diff_{k}':v for k,v in diffs.items()}}])
    check_table.to_csv(a.output_dir/'ARCHITECTURE_ROBUSTNESS_REPRODUCTION_CHECK.csv',index=False)
    if not check_pass: raise AssertionError(diffs)
    outputs={p.name:sha256(p) for p in sorted(a.output_dir.iterdir()) if p.is_file()}
    metadata={'status':'PASS','date':'2026-08-16','purpose':'Prespecified architecture-robustness analysis using the same target-measurement and updating procedure.','selected_procedure':{'sampling':'RSGS','updater':'CG-HPP','budgets':list(BUDGETS),'realizations':504},'selector_geometry':'All-906 MLR+RF target-response geometry; identical selected sources across architectures for this robustness analysis.','architectures':list(ARCHITECTURES),'architecture_specific_CG_HPP_hyperparameters_from':'673-record development OOF predictions only.','label_rule':'One rotated outcome record per selected source-specific diet; selected source entirely excluded from evaluation.','metric_denominator':{'FN_source_sample_SD':float(ref_sd[0]),'UN_source_sample_SD':float(ref_sd[1])},'primary_reproduction_check':{'pass':check_pass,'max_abs_diffs':diffs},'result_summary':aus.to_dict(orient='records'),'paired_block_bootstrap':'skipped_for_prediction_verification' if a.skip_bootstrap else {'reps':BOOTSTRAP_REPS,'seed':BOOTSTRAP_SEED},'interpretation':'Architecture comparisons are reported as robustness analyses; the starting model remains the MLR+RF model selected during development.','inputs':{str(a.predictions_546.name):sha256(a.predictions_546),str(a.primary_oof.name):sha256(a.primary_oof),str(a.tabm_oof.name):sha256(a.tabm_oof),str(a.rotation_ledger.name):sha256(a.rotation_ledger),str(a.structured_ledger.name):sha256(a.structured_ledger),str(a.target_spec.name):sha256(a.target_spec)},'outputs':outputs}
    (a.output_dir/'ARCHITECTURE_ROBUSTNESS_METADATA.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(aus.to_string(index=False));
    if rows: print(pd.DataFrame(rows).to_string(index=False))
    else: print('paired block bootstrap: SKIPPED by request')
    print('reproduction',diffs)

if __name__=='__main__': main()
