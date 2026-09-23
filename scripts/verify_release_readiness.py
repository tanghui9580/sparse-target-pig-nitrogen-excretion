#!/usr/bin/env python3
"""Repository-hygiene checks for a GitHub/archival release."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]

def noncomment_lines(path):
    return [x.strip() for x in path.read_text(encoding='utf-8').splitlines() if x.strip() and not x.lstrip().startswith('#')]

def main():
    # Check the tracked/package manifest rather than the live working tree: Python may legitimately
    # create __pycache__ after execution, but those runtime files must never be part of the release.
    import csv
    with (ROOT/'MANIFEST_SHA256.csv').open(encoding='utf-8',newline='') as f:
        manifest_rows=list(csv.DictReader(f))
    tracked=[r['path'] for r in manifest_rows]
    forbidden=[x for x in tracked if '/__pycache__/' in '/'+x or x.endswith('.pyc') or x.endswith('/.DS_Store') or x=='.DS_Store']
    if forbidden: raise AssertionError(f'forbidden packaged artifacts: {forbidden[:20]}')
    big=[(r['path'],int(r['size_bytes'])) for r in manifest_rows if int(r['size_bytes'])>=100*1024*1024]
    if big: raise AssertionError(f'GitHub >=100 MiB packaged files: {big}')

    ga=(ROOT/'.gitattributes').read_text(encoding='utf-8')
    if '* text=auto eol=lf' not in ga or '*.eps binary' not in ga:
        raise AssertionError('.gitattributes must normalize text to LF and mark EPS binary')

    req=noncomment_lines(ROOT/'requirements.txt')
    if not req or any('==' not in x for x in req): raise AssertionError('requirements.txt is not fully pinned')
    env=(ROOT/'environment.yml').read_text(encoding='utf-8')
    missing=[x for x in req if x not in env]
    if missing: raise AssertionError(f'environment.yml missing pinned requirements: {missing}')
    if 'python=3.12' not in env: raise AssertionError('reference Python 3.12 missing from environment.yml')

    cff=(ROOT/'CITATION.cff').read_text(encoding='utf-8')
    required_patterns=(r'^cff-version:\s*1\.2\.0\s*$',r'^version:\s*[\"\']?1\.0\.0[\"\']?\s*$',r'^license:\s*[\"\']?MIT[\"\']?\s*$')
    for pattern in required_patterns:
        if not re.search(pattern,cff,re.M): raise AssertionError(f'CITATION.cff missing pattern {pattern}')
    if re.search(r'\[to be completed\]|TODO|TBD',cff,re.I): raise AssertionError('CITATION.cff contains unresolved placeholder')

    pref=ROOT/'publication_reference'
    counts={ext:len(list(pref.rglob(f'*.{ext}'))) for ext in ('png','eps','xlsx','txt')}
    expected={'png':15,'eps':15,'xlsx':3,'txt':2}
    if counts!=expected: raise AssertionError((counts,expected))

    # Figure-source contract: exactly 15 named figures and an existing active plotting script/source table.
    mf=ROOT/'figure_sources/FIGURE_SOURCE_MANIFEST.csv'
    with mf.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    if len(rows)!=15: raise AssertionError(f'figure source manifest rows={len(rows)}')
    for r in rows:
        src_field=(r.get('source_csvs') or r.get('Source_file') or r.get('source_file') or r.get('Source') or r.get('source') or '').strip()
        script=(r.get('active_script') or r.get('Script') or r.get('script') or r.get('Plotting_script') or r.get('plotting_script') or '').strip()
        for src in [x.strip() for x in src_field.split(';') if x.strip()]:
            if not (ROOT/'figure_sources'/src).is_file() and not (ROOT/src).is_file():
                raise AssertionError(f'missing figure source: {src}')
        if not script:
            raise AssertionError(f'figure manifest has no active script for {r.get("figure")}')
        if not (ROOT/script).is_file() and not (ROOT/'scripts'/script).is_file():
            raise AssertionError(f'missing figure script: {script}')

    for rel in ('LICENSE','DATA_RIGHTS.md','README.md','README_CN.md','VERIFICATION.md','GITHUB_RELEASE_CHECKLIST.md','data/provenance/HISTORICAL_170_STUDY_PROVENANCE.csv'):
        if not (ROOT/rel).is_file(): raise AssertionError(f'missing release file: {rel}')
    print('PASS: release-readiness hygiene')
    print(f'  publication_reference counts: {counts}')
    print('  no packaged caches/bytecode/100-MiB files; dependencies pinned; release metadata structurally complete')

if __name__=='__main__': main()
