#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
A=pd.read_csv(ROOT/'publication_table'/'TABLE_S5_PANEL_A_B3_MATCHED.csv')
B=pd.read_csv(ROOT/'publication_table'/'TABLE_S5_PANEL_B_SAME_LABEL.csv')
P=pd.read_csv(ROOT/'results'/'supervised_transfer'/'CROSSARCH_TRADABOOSTR2_VS_CGHPP_AUBC_504.csv')
assert len(A)==24 and len(B)==8 and len(P)==2016
assert set(A.loc[A.Method.isin(['KMM','RuLSIF','TCA','CORAL']),'Target_Y_labels'])=={0}
for arch in ['MLR','RF','TabM','MLR+RF']:
    p=P[P.Architecture.eq(arch)]
    assert len(p)==504
    assert (p.CGHPP_AUBC < p.TrAda_AUBC).sum()==504
print('PASS: classical DA/transfer benchmark')
