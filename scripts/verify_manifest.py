#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, sys
ROOT=Path(__file__).resolve().parents[1]
MAN=ROOT/'MANIFEST_SHA256.csv'
EXCLUDE_PREFIX=('.git/','generated/','work/')
EXCLUDE_NAMES={'MANIFEST_SHA256.csv'}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def current_files():
    out=[]
    for p in ROOT.rglob('*'):
        if not p.is_file(): continue
        rel=p.relative_to(ROOT).as_posix()
        if rel in EXCLUDE_NAMES or rel.startswith(EXCLUDE_PREFIX) or '/__pycache__/' in '/'+rel or rel.endswith('.pyc'): continue
        out.append((rel,p))
    return dict(out)

def main():
    expected={}
    with MAN.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f): expected[r['path']]=(r['sha256'],int(r['size_bytes']))
    cur=current_files(); missing=sorted(set(expected)-set(cur)); extra=sorted(set(cur)-set(expected)); bad=[]
    for rel,(hs,sz) in expected.items():
        if rel not in cur: continue
        p=cur[rel]
        if p.stat().st_size!=sz or sha(p)!=hs: bad.append(rel)
    if missing or extra or bad:
        print('FAIL manifest'); print('missing',missing[:10]); print('extra',extra[:10]); print('bad',bad[:10]); sys.exit(1)
    print(f'PASS: manifest {len(expected)}/{len(cur)} files exact')
if __name__=='__main__': main()
