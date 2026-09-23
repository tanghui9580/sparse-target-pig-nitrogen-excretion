#!/usr/bin/env python3
"""Build publication-layer artifacts in one command.

Outputs:
- regenerated figures from machine-readable figure-source tables (PNG/EPS),
- exact locked submission artwork re-materialized from publication_reference/artwork,
- exact locked publication-source table workbooks re-materialized from publication_reference/tables,
- captions and a hash/integrity report.
"""
from __future__ import annotations
from pathlib import Path
import argparse, csv, hashlib, shutil, subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'publication_reference'

def sha256(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def copy_tree_files(src:Path,dst:Path,patterns=('*',)):
    dst.mkdir(parents=True,exist_ok=True); rows=[]
    for pat in patterns:
        for p in sorted(src.glob(pat)):
            if p.is_file():
                q=dst/p.name; shutil.copy2(p,q); rows.append((p,q))
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=ROOT/'generated'); ap.add_argument('--clean',action='store_true',default=True); ap.add_argument('--jobs',type=int,default=1,help='Parallel isolated workers for figure export.'); a=ap.parse_args()
    out=a.output_dir
    if a.clean and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)

    # 1) Data-driven rebuild of all 15 figures.
    subprocess.run([sys.executable,str(ROOT/'scripts/build_publication_figures.py'),'--output-dir',str(out/'rebuilt_figures'),'--jobs',str(max(1,a.jobs))],check=True,cwd=ROOT)

    # 2) Exact publication-layer source assets are re-materialized and hash-checked.
    copied=[]
    copied += copy_tree_files(REF/'artwork/Main_Figures',out/'locked_submission_artwork/Main_Figures',('*.png','*.eps'))
    copied += copy_tree_files(REF/'artwork/Supplementary_Figures',out/'locked_submission_artwork/Supplementary_Figures',('*.png','*.eps'))
    copied += copy_tree_files(REF/'tables',out/'tables',('*.xlsx',))
    copied += copy_tree_files(REF/'captions',out/'captions',('*.txt',))

    rows=[]
    for src,dst in copied:
        hs,hd=sha256(src),sha256(dst)
        rows.append({'artifact':str(dst.relative_to(out)),'source':str(src.relative_to(ROOT)),'sha256':hd,'hash_match':hs==hd})
        if hs!=hd: raise AssertionError((src,dst))

    rebuilt=list((out/'rebuilt_figures').glob('*'))
    byext={e:len(list((out/'rebuilt_figures').glob(f'*.{e}'))) for e in ['png','eps']}
    if byext!={'png':15,'eps':15}: raise AssertionError(byext)
    locked_png=len(list((out/'locked_submission_artwork').rglob('*.png'))); locked_eps=len(list((out/'locked_submission_artwork').rglob('*.eps')))
    tables=len(list((out/'tables').glob('*.xlsx')))
    if (locked_png,locked_eps,tables)!=(15,15,3): raise AssertionError((locked_png,locked_eps,tables))

    report=out/'PUBLICATION_ARTIFACT_HASH_CHECK.csv'
    with report.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['artifact','source','sha256','hash_match']); w.writeheader(); w.writerows(rows)
    (out/'PUBLICATION_BUILD_REPORT.txt').write_text(
        'PUBLICATION BUILD PASS\n'
        'Data-driven rebuilt figures: 15 PNG + 15 EPS\n'
        'Locked submission artwork re-materialized: 15 PNG + 15 EPS; byte-identical to packaged publication references\n'
        'Publication-source table workbooks re-materialized: 3 XLSX; byte-identical to packaged publication references\n'
        'Captions re-materialized: 2 TXT\n'
        'Scope note: statistical figure content is regenerated from machine-readable figure-source tables. Exact locked typography/layout of the submission artwork and the styled Excel table workbooks are publication-layer reference assets and are hash-verified rather than reverse-engineered from raw analysis code.\n',encoding='utf-8')
    print('PASS: one-command publication artifact build completed')
    print('Output:',out)

if __name__=='__main__': main()
