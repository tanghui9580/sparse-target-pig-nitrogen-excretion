#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, sys, shutil
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
ARCH=ROOT/'analysis/architecture_specific'
WORK=ROOT/'work'

def run(cmd):
    print('+',' '.join(map(str,cmd)))
    subprocess.run(list(map(str,cmd)),check=True,cwd=ROOT)

def selectors():
    out=WORK/'selectors'; out.mkdir(parents=True,exist_ok=True)
    led=out/'reproduced_rsgs_selections_3_21.csv'
    diag=out/'rsgs_selection_diagnostics_3_21.csv'
    run([sys.executable,ARCH/'scripts/generate_arch_specific_selections.py','--predictions-546',ARCH/'inputs/four_arch_target546_predictions.csv','--output-ledger',led,'--output-diagnostics',diag])
    a=pd.read_csv(ARCH/'inputs/reported_rsgs_selections_3_21.csv')
    b=pd.read_csv(led)
    cols=['Architecture','Budget','Rank','Source_unit']
    if not a[cols].sort_values(cols).reset_index(drop=True).equals(b[cols].sort_values(cols).reset_index(drop=True)):
        raise AssertionError('Selected-source ledger differs from the packaged reference')
    print('PASS: 336/336 architecture-specific selections reproduced exactly')
    return led

def light(start=0,end=1):
    led=selectors(); out=WORK/'light_smoke'; out.mkdir(parents=True,exist_ok=True)
    run([sys.executable,ARCH/'scripts/run_light_from_selection.py','--input-dir',ARCH/'inputs','--selection',led,'--out',out,'--start',start,'--end',end])
    return out/'ARCH_SPECIFIC_LIGHT_504_METRICS.csv.gz'

def heavy(start=0,end=1,jobs=4,architectures=None,budgets=None):
    led=selectors(); out=WORK/'heavy_smoke'; out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,ARCH/'scripts/run_heavy_from_selection.py','--input-dir',ARCH/'inputs','--selection',led,'--out',out,'--start',start,'--end',end,'--jobs',jobs]
    if architectures: cmd += ['--architectures',*architectures]
    if budgets: cmd += ['--budgets',*map(str,budgets)]
    run(cmd)
    files=sorted(out.glob('HEAVY_ARCH_SPEC_*.csv.gz'))
    if len(files)!=1: raise AssertionError(files)
    return files[0]

def compare(path, methods, start=0,end=1):
    new=pd.read_csv(path); reference=pd.read_csv(ARCH/'results/FULL_MATRIX_504_METRICS.csv.gz')
    reference=reference[reference.Realization.between(start,end-1)&reference.Method.isin(methods)]
    keys=['Realization','Architecture','Budget','Method']; metrics=['FN_RMSE','FN_MAE','FN_Bias','FN_R2','UN_RMSE','UN_MAE','UN_Bias','UN_R2','Primary_NRMSE']
    m=new.merge(reference[keys+metrics],on=keys,suffixes=('_new','_reference'),validate='one_to_one')
    mx=max(float((m[f'{x}_new']-m[f'{x}_reference']).abs().max()) for x in metrics)
    if len(m)!=len(new) or mx>1e-10: raise AssertionError((len(new),len(m),mx))
    print(f'PASS: {len(m)} cells; max abs difference={mx:.3g}')

def main():
    p=argparse.ArgumentParser(); p.add_argument('profile',choices=['verify','release-audit','selectors','light-smoke','heavy-smoke','primary-factorial','sow80','summaries','classical-benchmark','derived-endpoints','publication','full-extension']); p.add_argument('--jobs',type=int,default=1); a=p.parse_args()
    if a.profile=='verify':
        run([sys.executable,ROOT/'scripts/verify_manifest.py'])
        run([sys.executable,ROOT/'scripts/verify_reproducibility.py'])
        run([sys.executable,ROOT/'scripts/verify_provenance.py'])
        run([sys.executable,ROOT/'scripts/verify_duplicate_contracts.py'])
        run([sys.executable,ROOT/'scripts/verify_release_readiness.py'])
    elif a.profile=='release-audit':
        run([sys.executable,ROOT/'scripts/reproduce.py','verify'])
        selectors()
        compare(light(),{'NoUpdate','Mean','Affine','CG-HPP'})
        compare(heavy(jobs=a.jobs,architectures=['MLR+RF'],budgets=[12]),{'TargetOnlyRidge','TargetOnlyElasticNet','TargetOnlyRF','TargetWeightedFullRF'})
        run([sys.executable,ROOT/'scripts/reproduce.py','classical-benchmark'])
        run([sys.executable,ROOT/'scripts/reproduce.py','derived-endpoints'])
        run([sys.executable,ROOT/'scripts/reproduce.py','publication','--jobs',a.jobs])
        print('PASS: release-audit (quick contracts + selectors + light smoke + representative heavy smoke + classical benchmark + prediction-level RN/TNE verification + publication build)')
    elif a.profile=='selectors': selectors()
    elif a.profile=='light-smoke': compare(light(),{'NoUpdate','Mean','Affine','CG-HPP'})
    elif a.profile=='heavy-smoke': compare(heavy(jobs=a.jobs,architectures=['MLR+RF'],budgets=[12]),{'TargetOnlyRidge','TargetOnlyElasticNet','TargetOnlyRF','TargetWeightedFullRF'})
    elif a.profile=='primary-factorial':
        out=WORK/'primary_factorial'; out.mkdir(parents=True,exist_ok=True)
        run([sys.executable,ROOT/'analysis/route/run_546_sampling_updating_factorial.py','--matrix-906',ROOT/'data/analysis_matrices/historical_906.csv','--predictions-546',ROOT/'data/pre_update_predictions/four_arch_target546_predictions.csv','--oof-673',ROOT/'data/pre_update_predictions/mlr_rf_673_oof.csv','--model-spec',ROOT/'config/primary_model_specification.json','--output-dir',out])
    elif a.profile=='sow80':
        out=WORK/'sow80'; out.mkdir(parents=True,exist_ok=True)
        run([sys.executable,ROOT/'analysis/route/run_sow80_evaluation.py','--matrix-906',ROOT/'data/analysis_matrices/historical_906.csv','--sow-predictions',ROOT/'data/pre_update_predictions/four_arch_sow80_predictions.csv','--primary-oof',ROOT/'data/pre_update_predictions/mlr_rf_673_oof.csv','--tabm-oof',ROOT/'data/pre_update_predictions/tabm_673_oof.csv','--model-spec',ROOT/'config/primary_model_specification.json','--target-spec',ROOT/'config/target_updating_specification.json','--output-dir',out])
    elif a.profile=='summaries':
        run([sys.executable,ROOT/'scripts/summarize_reported_results.py'])
    elif a.profile=='derived-endpoints':
        run([sys.executable,ROOT/'scripts/verify_derived_endpoints.py'])
    elif a.profile=='publication':
        run([sys.executable,ROOT/'scripts/build_publication_artifacts.py','--output-dir',ROOT/'generated','--jobs',a.jobs])
    elif a.profile=='classical-benchmark':
        base=ROOT/'analysis/classical_da_transfer_benchmark'; out=WORK/'classical_benchmark'; out.mkdir(parents=True,exist_ok=True)
        run([sys.executable,base/'code/verify_benchmark.py'])
        run([sys.executable,base/'code/rebuild_table_s5_sources.py','--outdir',out])
        for name in ['TABLE_S5_PANEL_A_B3_MATCHED.csv','TABLE_S5_PANEL_B_SAME_LABEL.csv']:
            new=pd.read_csv(out/name); reference=pd.read_csv(base/'publication_table'/name)
            if not new.equals(reference):
                # Numeric files may differ only at floating-point string precision; compare numerics tightly.
                if list(new.columns)!=list(reference.columns) or len(new)!=len(reference): raise AssertionError(name)
                for c in new.columns:
                    if pd.api.types.is_numeric_dtype(new[c]):
                        if (new[c]-reference[c]).abs().max()>1e-12: raise AssertionError((name,c))
                    elif not new[c].astype(str).equals(reference[c].astype(str)): raise AssertionError((name,c))
        print('PASS: Table S5 source tables rebuilt from the packaged benchmark results')
    elif a.profile=='full-extension':
        lp=light(0,504); hp=heavy(0,504,a.jobs); out=WORK/'full_extension'; out.mkdir(parents=True,exist_ok=True)
        full=pd.concat([pd.read_csv(lp),pd.read_csv(hp)],ignore_index=True).sort_values(['Realization','Architecture','Budget','Method']).reset_index(drop=True)
        if len(full)!=112896: raise AssertionError(len(full))
        path=out/'FULL_MATRIX_504_METRICS.csv.gz'; full.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
        run([sys.executable,ROOT/'scripts/summarize_reported_results.py','--metrics',path,'--out',out])

if __name__=='__main__': main()
