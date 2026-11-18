#!/usr/bin/env python3
"""TRACE-CXR reproducibility CLI: safe, CPU-first evidence verification.

Does not download data, train a model, invoke CUDA, or modify source evidence.
This evidence checker is not the original training or inference code.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from pipeline import rows, binary_metrics, summarize_timings, reliability_bins, ece_from_bins
from audit import verify_manifest, check_split_csv, verify_frozen_roles, public_safety_scan

ROOT=Path(__file__).resolve().parent
# Raw per-image timing file of the interleaved six-candidate benchmark.
# The public name is tried first; archived copies may carry a prefix, so any
# file ending in INTERLEAVED_TIMING_SUFFIX is accepted as well.
INTERLEAVED_TIMING_FILES=('INTERLEAVED_LATENCY_RAW_TIMINGS.csv',)
INTERLEAVED_TIMING_SUFFIX='SIX_CANDIDATE_RAW_TIMINGS.csv'
OLDER_TIMING_FILE='TRACE_CXR_INFERENCE_TIME.csv'


def search_dirs(root: Path) -> list[Path]:
    return [root,root/'outputs'/'trace_cxr_inception_swinb_v2',root/'immutable_source_evidence']


def locate(root: Path | None, filename: str) -> Path | None:
    if root is None:return None
    for folder in search_dirs(root):
        candidate=folder/filename
        if candidate.is_file():return candidate
    return None


def locate_interleaved_timing(root: Path | None) -> Path | None:
    """Find the interleaved benchmark file: exact public name, then a unique suffix match."""
    if root is None:return None
    for name in INTERLEAVED_TIMING_FILES:
        found=locate(root,name)
        if found is not None:return found
    for folder in search_dirs(root):
        if not folder.is_dir():continue
        matches=sorted(p for p in folder.glob('*'+INTERLEAVED_TIMING_SUFFIX) if p.is_file())
        if len(matches)>1:
            raise ValueError(f'Several interleaved timing files in {folder.name}; keep exactly one')
        if matches:return matches[0]
    return None


def check_metrics(root: Path | None, expected: dict) -> dict:
    path=locate(root,'TRACE_CXR_INTERNAL_PREDICTIONS.csv')
    if path is None:
        return {'status':'SKIPPED_CASE_LEVEL_EVIDENCE_NOT_PROVIDED',
                'reason':'Aggregate archived results are evidence records, not independent metric recomputation.'}
    items=list(rows(path)); labels=[int(x['label']) for x in items]
    values=[float(x['gated_calibrated']) for x in items]
    threshold=float(expected['control_lock']['models']['gated_fusion']['threshold'])
    result=binary_metrics(labels,values,threshold)
    if len(items)!=5337:raise ValueError('Expected 5337 RSNA locked-test rows')
    bins=reliability_bins(labels,values)
    result['ece15']=ece_from_bins(bins,len(labels))
    archived=[r for r in rows(ROOT/'evidence'/'reliability_bins.csv')
              if r['cohort']=='RSNA locked test' and r['model']=='gated_fusion']
    for mine,ref in zip(bins,archived):
        if mine['n_images']!=int(ref['n_images']):
            raise ValueError(f"Reliability bin count mismatch in bin {mine['bin']}")
        if mine['n_images'] and abs(mine['observed_rate']-float(ref['observed_rate']))>5e-5:
            raise ValueError(f"Reliability observed rate mismatch in bin {mine['bin']}")
    return {'status':'PASS_RECOMPUTED_FROM_LOCAL_CASE_SCORES','source':path.name,'result':result,
            'reliability_bins':'MATCH_EVIDENCE_RELIABILITY_BINS_CSV'}


def check_xai(root: Path | None, config: dict) -> dict:
    path=locate(root,'TRACE_CXR_XAI_FAITHFULNESS.csv')
    if path is None:
        return {'status':'SKIPPED_CASE_LEVEL_EVIDENCE_NOT_PROVIDED',
                'archived_values':config['archived_aopc'],
                'limitation':'Report of archived values only; no raw attribution replay.'}
    groups={}
    for line in rows(path):
        if line['cohort']=='Kermany':
            groups.setdefault(line['branch'],[]).append(float(line['aopc_gain_over_random']))
    outcome={}
    for model,target in config['archived_aopc'].items():
        arr=groups.get(model,[])
        if len(arr)!=100:raise ValueError(f'Incorrect Kermany AOPC count for {model}')
        observed=sum(arr)/len(arr)
        if abs(observed-float(target))>0.00005:
            raise ValueError(f'Archived AOPC mismatch for {model}')
        outcome[model]={'n':len(arr),'mean':observed,'archived_rounded':target}
    return {'status':'PASS_MEANS_FROM_LOCAL_ARCHIVED_CASE_SCORES',
            'results':outcome,'confidence_intervals':'archived image-level bootstrap (docs/results.md); not regenerated here'}


def check_timing(root: Path | None, config: dict) -> dict:
    path=locate_interleaved_timing(root)
    interleaved=path is not None
    if path is None:path=locate(root,OLDER_TIMING_FILE)
    if path is None:
        return {'status':'SKIPPED_RAW_TIMING_NOT_PROVIDED',
                'archived_benchmark_summary':config['archived_latency_benchmark']}
    result=summarize_timings(path)
    if not interleaved:
        return {'status':'PASS_OLDER_COMPONENT_TIMING_ONLY','source':path.name,'result':result,
                'warning':'Interleaved six-candidate timing file not found; this older two-backbone '
                          'record is not the matched latency comparison reported in the paper.'}
    expected_n=int(config['archived_latency_benchmark']['n_per_candidate'])
    if set(result)!=set(config['candidate_names']) or any(v['n']!=expected_n for v in result.values()):
        raise ValueError(f'Interleaved benchmark mismatch: six candidate groups with n={expected_n} required')
    archived=config['archived_latency_benchmark']['candidate_means_ms']
    for name,value in archived.items():
        if abs(result[name]['mean_ms']-float(value))>0.001:
            raise ValueError(f'Interleaved latency mean mismatch for {name}')
    return {'status':'PASS_INTERLEAVED_TIMING_RECOMPUTED_FROM_LOCAL_ROWS','source':path.name,'result':result}


def check_calibration(config: dict) -> dict:
    from pipeline import calibrate, fuse
    lock=config['control_lock']
    for method in config['candidate_names']:
        entry=lock['models'][method]
        if not (0 < float(entry['temperature']) and 0 <= float(entry['threshold']) <= 1):
            raise ValueError(f'Invalid T/threshold for {method}')
        if not 0 <= calibrate(0.23,float(entry['temperature'])) <= 1:
            raise ValueError('Invalid calibration result')
    if abs(fuse(0.6,0.2,'mean_fusion',lock)-0.4)>1e-12:
        raise ValueError('Fusion equation unexpectedly changed')
    return {'status':'PASS_FROZEN_PARAMETER_SCHEMA_AND_FORMULA',
            'n_candidates':len(config['candidate_names']),
            'warning':'This is not a replay of 6-candidate GPU inference.'}


def report(cmd: str, args, config: dict) -> dict:
    data_root=Path(args.data_root).resolve() if args.data_root else None
    if cmd=='verify': return verify_manifest(ROOT)
    if cmd=='splits':
        file=locate(data_root,'TRACE_CXR_RSNA_SPLIT_MANIFEST.csv')
        source='private'
        if not file:
            file=ROOT/'data'/'rsna_split_manifest.csv';source='public'
        if not file.is_file():return {'status':'SKIPPED_SPLIT_MANIFEST_NOT_PROVIDED',
                            'n_archived':config['n_rsna'],
                            'warning':'No image-ID-level verification performed.'}
        result=check_split_csv(file)
        expected=config['split_sizes']
        for role,n in result['split_counts'].items():
            if role in expected and int(expected[role])!=n:
                raise ValueError(f'Split size mismatch for {role}')
        if result['n_total']!=int(config['n_rsna']):raise ValueError('RSNA manifest size mismatch')
        result['manifest_source']=source
        return result
    if cmd=='metrics':return check_metrics(data_root,config)
    if cmd=='calibration':return check_calibration(config)
    if cmd=='xai':return check_xai(data_root,config)
    if cmd=='timing':return check_timing(data_root,config)
    if cmd=='audit':
        checks=verify_frozen_roles(config)
        safe=public_safety_scan(ROOT)
        if safe['status'].startswith('FAIL'):raise ValueError('Unsafe public repository content')
        return {'status':'PASS_DOCUMENTED_ROLE_POLICY_AND_TECHNICAL_SCAN',
                'role_policy':checks,'public_safety':safe,
                'limitation':'Policy checks do not prove all original training runs leakage-free.'}
    raise ValueError(f'Unsupported command: {cmd}')


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','splits','metrics','calibration','xai','timing','audit','all'])
    parser.add_argument('--data-root',help='Private local archive directory; no upload occurs')
    parser.add_argument('--checkpoint-root',help='Private checkpoint location (declared, not loaded by safe CLI)')
    parser.add_argument('--output-root',default='reports',help='Directory for machine-readable audit reports')
    parser.add_argument('--config',default='configs/historical.json',help='Read-only JSON configuration')
    args=parser.parse_args(argv)
    try:
        conf=Path(args.config)
        if not conf.is_absolute():conf=ROOT/conf
        config=json.loads(conf.read_text(encoding='utf-8'))
        out=Path(args.output_root)
        if not out.is_absolute():out=ROOT/out
        out=out.resolve()
        # forbid destructive output locations under source evidence
        protected=(ROOT/'evidence').resolve()
        if out == protected or protected in out.parents:
            raise ValueError('Output cannot overwrite evidence directory')
        out.mkdir(parents=True,exist_ok=True)
        commands=['verify','splits','metrics','calibration','xai','timing','audit'] if args.command=='all' else [args.command]
        results={}
        for command in commands:
            results[command]=report(command,args,config)
            print(f"{command}: {results[command]['status']}")
        output={'command':args.command,'results':results,'gpu_used':False,'training_performed':False,
                'medical_data_transmitted':False,'checkpoint_root_supplied':bool(args.checkpoint_root),
                'note':'No raw model forward or GPU inference performed by this CLI.'}
        target=out/('trace_cxr_'+args.command+'.json')
        target.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        print('Report:',target)
        return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
