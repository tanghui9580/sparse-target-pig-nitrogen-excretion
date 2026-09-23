#!/usr/bin/env python3
from pathlib import Path
import csv, json, math, sys
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results/predictor_information'

def rd(name):
    with open(D/name,encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
evidence=json.loads((D/'PREDICTOR_INFORMATION_673_FULLP3_SIXMODEL.json').read_text(encoding='utf-8'))
assert evidence['selection_data']['records']==673 and evidence['selection_data']['study_groups']==122
assert evidence['confirmation_data']['records']==233 and evidence['confirmation_data']['study_groups']==48
assert evidence['study_id_overlap']==0
assert evidence['P3_coverage']['development'].startswith('673/673')
assert evidence['P3_coverage']['confirmation'].startswith('233/233')
models={'MLR','ElasticNet','SVR','RandomForest','ExtraTrees','XGBoost'}
s=rd('PREDICTOR_INFORMATION_673_FULLP3_SIX_MODEL_SCREEN.csv')
for layer in ('P0','P1','P2','P3'):
    assert {r['Model'] for r in s if r['Layer']==layer}==models
b=rd('PREDICTOR_INFORMATION_673_FULLP3_SIX_MODEL_P3P2_BOOTSTRAP.csv')
assert len(b)==6 and {r['Model'] for r in b}==models
assert all(float(r['CI2.5']) < 0 < float(r['CI97.5']) for r in b)
c=rd('PREDICTOR_LAYER_233_FULLP3_SIX_MODEL_CONFIRMATION_ONLY.csv')
assert {r['Model'] for r in c if r['Layer']=='P2'}==models and {r['Model'] for r in c if r['Layer']=='P3'}==models
cb=rd('PREDICTOR_LAYER_233_FULLP3_SIX_MODEL_P3P2_BOOTSTRAP.csv')
assert len(cb)==6 and all(r['Role']=='confirmation only' for r in cb)
assert all(float(r['CI2.5']) < 0 < float(r['CI97.5']) for r in cb)
ranking=rd('PREDICTOR_INFORMATION_TEN_MODEL_RANKING_CHECK.csv')
score={r['Model']:float(r['Primary_NRMSE']) for r in ranking if r.get('Model')}
assert abs(score['MLR+RF']-0.5230654982536393)<1e-12
assert abs(score['MLR+ET']-0.5242440174628202)<1e-12
assert score['MLR+RF'] < score['MLR+ET']
print('PASS: 673-record predictor-information verification')
print('673/122 selection; 233/48 confirmation-only; P3 673/673 + 233/233; 6 model families; 6/6 P3-P2 intervals span zero; locked predictor-information benchmark unchanged.')
