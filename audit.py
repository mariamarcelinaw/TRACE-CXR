"""Artifact and split/role auditing; no patient-level independence claims."""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path
from collections import Counter
from pipeline import rows


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_manifest(root: Path, checksum_filename: str='evidence/checksums.sha256') -> dict:
    root=root.resolve(); path=root/checksum_filename
    if not path.is_file(): raise FileNotFoundError(path)
    checked=0
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        expected,space,relative=line.partition('  ')
        if not space or not expected or not relative: raise ValueError('Invalid checksum record')
        raw=root/relative
        if raw.is_symlink(): raise ValueError('Symlinked evidence not permitted')
        file=raw.resolve()
        if not file.is_relative_to(root): raise ValueError('Path traversal in manifest')
        if not file.is_file(): raise FileNotFoundError(relative)
        if sha256_file(file)!=expected: raise ValueError(f'Checksum mismatch: {relative}')
        checked += 1
    return {'status':'PASS','files_verified':checked}


def require_disjoint_by_role(groups: dict[str,set[str]]) -> None:
    """Raise even with Python -O; avoid role overlap for fitting/evaluation."""
    items=list(groups.items())
    for i,(name,ids) in enumerate(items):
        for other,other_ids in items[i+1:]:
            if ids & other_ids:
                raise ValueError(f'Role leakage detected: {name} intersects {other}')


def check_split_csv(path: Path) -> dict:
    records=list(rows(path))
    if not records: raise ValueError('Empty split manifest')
    required={'id','split','fold'}
    if not required.issubset(records[0]): raise ValueError('Missing required split columns')
    has_label='label' in records[0]
    groups={}; all_ids=set();foldcounts=Counter()
    for record in records:
        identifier=record['id'];role=record['split']
        if role not in ('development','calibration','locked_test'):
            raise ValueError('Unexpected split role')
        if has_label and record['label'] not in ('0','1'): raise ValueError('Unexpected label')
        if identifier in all_ids: raise ValueError('Duplicate competition image identifier')
        all_ids.add(identifier);groups.setdefault(role,set()).add(identifier)
        if role=='development':
            fold=record['fold']
            if fold not in ('0','1','2','3','4','0.0','1.0','2.0','3.0','4.0'):
                raise ValueError('Invalid development fold')
            foldcounts[str(int(float(fold)))] += 1
        elif record['fold'].strip() and record['fold'].lower() not in ('nan','none'):
            raise ValueError('Fold assigned outside development')
    require_disjoint_by_role(groups)
    return {'status':'PASS_IDENTIFIER_LEVEL','n_total':len(records),
            'split_counts':{k:len(v) for k,v in sorted(groups.items())},
            'development_folds':dict(sorted(foldcounts.items())),
            'biological_patient_independence':'NOT_VERIFIED'}


def verify_frozen_roles(config: dict) -> dict:
    roles=config['fitting_roles']
    if roles['fusion']!='development_oof': raise ValueError('Fusion fitting role not development OOF')
    if roles['temperature_threshold']!='rsna_calibration': raise ValueError('Calibration fitting role invalid')
    if roles['external_kermany']!='evaluation_only': raise ValueError('External Kermany fitting prohibited')
    if config.get('preprocessing',{}).get('size')!=320:
        raise ValueError('Unexpected input size')
    if 'historical' not in config.get('experiment_scopes',{}):
        raise ValueError('Historical scope not declared')
    return {'status':'PASS_CONFIG_ROLE_POLICY','external_label_fitting':False,
            'historical_leakage_risk':'DOCUMENTED_NOT_PROVEN_ABSENT'}


def public_safety_scan(root: Path) -> dict:
    suspicious=[]; count=0
    deny_extensions={'.dcm','.pt','.pth','.npz','.key','.pem'}
    for f in root.rglob('*'):
        if not f.is_file():continue
        count+=1
        if f.is_symlink() or f.suffix.lower() in deny_extensions or f.name in ('.env','id_rsa'):
            suspicious.append(str(f.relative_to(root)))
        if f.stat().st_size > 15_000_000:
            suspicious.append(str(f.relative_to(root))+' (>15MB)')
    return {'status':'PASS_TECHNICAL_SCAN' if not suspicious else 'FAIL_PUBLIC_SAFETY',
            'files_examined':count,'suspicious_paths':sorted(set(suspicious)),
            'license_authorization':'NOT_VERIFIED','dataset_rights':'NOT_VERIFIED'}
