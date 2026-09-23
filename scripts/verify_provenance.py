#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
hist=pd.read_csv(ROOT/'data/analysis_matrices/historical_906.csv')
prov=pd.read_csv(ROOT/'data/provenance/HISTORICAL_170_STUDY_PROVENANCE.csv')

assert len(hist)==906
assert hist.Study_ID.nunique()==170
assert len(prov)==170 and prov.Study_ID.nunique()==170
assert set(hist.Study_ID)==set(prov.Study_ID)
assert prov.Author_Year.notna().all()
assert prov.Analysis_Split.isin(['development','confirmation']).all()

counts=hist.groupby('Study_ID').size().sort_index()
pcounts=prov.set_index('Study_ID').Records_in_historical_906.sort_index()
assert counts.equals(pcounts)
splits=hist[['Study_ID','Analysis_Split']].drop_duplicates().set_index('Study_ID').Analysis_Split.sort_index()
psplits=prov.set_index('Study_ID').Analysis_Split.sort_index()
assert splits.equals(psplits)

best=prov.Best_Available_Source_Identifier.fillna('').astype(str).str.strip()
if not best.ne('').all():
    raise AssertionError('Every historical Study_ID must have a traceable best-available DOI/link or source identifier')
allowed={'V44_reviewer_ready_study_index','v35_verified_metadata_registry','publisher_metadata_2026-09-22','journal_metadata_2026-09-22'}
if not set(prov.Best_Identifier_Source).issubset(allowed):
    raise AssertionError(sorted(set(prov.Best_Identifier_Source)-allowed))

n_v44=prov.DOI_or_Link.fillna('').astype(str).str.strip().ne('').sum()
n_v35_union=((prov.DOI_or_Link.fillna('').astype(str).str.strip()!='') | (prov.Canonical_DOI_v35.fillna('').astype(str).str.strip()!='')).sum()
n_external=prov.Best_Identifier_Source.isin(['publisher_metadata_2026-09-22','journal_metadata_2026-09-22']).sum()
print('PASS: historical-study provenance')
print('  Study_ID mapped: 170/170')
print(f'  V44 DOI/link available: {n_v44}/170')
print(f'  V44 + verified v35 identifier coverage: {n_v35_union}/170')
print(f'  externally resolved remaining identities: {n_external}; best-available source-identifier coverage: 170/170')
print('  REVIEW_FLAG_records is a field/record-level QA count, not a study-identity confidence label')
