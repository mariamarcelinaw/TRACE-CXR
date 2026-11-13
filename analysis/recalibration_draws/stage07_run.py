"""Stage 07: read-only audit/design by default. Historical extension requires explicit acknowledgement."""
import argparse,csv,json,sys
from pathlib import Path
from stage07_input_validation import validate,six_scores,MODELS
from stage07_release_gate import corrected_release,historical_extension_release
from stage07_sampling import draw_seed
from stage07_simulation import run_one
from stage07_provenance import sha256

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['design','historical-audit','synthetic-test','historical-replay','historical-exploratory','corrected-analysis'],default='design')
    ap.add_argument('--predictions'); ap.add_argument('--lock'); ap.add_argument('--output-dir');ap.add_argument('--acknowledge-posthoc',action='store_true')
    ap.add_argument('--reps',type=int,default=500);ap.add_argument('--seed',type=int,default=20260908)
    args=ap.parse_args(argv)
    if args.mode=='design':print('STAGE07_DESIGN_ONLY: no real target recalibration performed');return 0
    if args.mode=='corrected-analysis':corrected_release()
    if args.mode=='synthetic-test':print('Run: PYTHONPATH=code python -m unittest discover -s tests -p "test_stage07_*.py" -v');return 0
    if not args.predictions or not args.lock:ap.error('historical mode requires --predictions and --lock')
    import pandas as pd
    d=pd.read_csv(args.predictions);lock=json.loads(Path(args.lock).read_text());validate(d)
    q=six_scores(d,lock)
    report={'status':'HISTORICAL_AUDIT_ONLY','cohort':'KermanyNorm','rows':len(d),'positive':int(d.label.sum()),'negative':int((1-d.label).sum()),'candidates':list(q),'prediction_sha256':sha256(args.predictions),'lock_sha256':sha256(args.lock)}
    if args.mode=='historical-audit':print(json.dumps(report,indent=2));return 0
    if args.mode=='historical-replay':raise NotImplementedError('Exact historical replay is NOT implemented; use historical-exploratory only. Do not label extension as historical replay.')
    historical_extension_release(args.acknowledge_posthoc,source_version='KermanyNorm')
    if report['prediction_sha256']!='6deaf0f52c69c068dacfcad3ffe720a9624834ce75008d119f8631da9650c271' or report['lock_sha256']!='a34b7bbb061a0c928aad7f60045feb92359b039bb762568d89a2d6ebd4d881b7':
        raise PermissionError('BLOCKED_PREDICTION_PROVENANCE: source SHA256 does not match frozen KermanyNorm and control lock')
    if not args.output_dir:ap.error('retrospective execution requires a NEW --output-dir')
    out=Path(args.output_dir)
    if out.exists():raise FileExistsError('output directory must not already exist')
    if args.reps<=0:raise ValueError('reps must be positive')
    out.mkdir(parents=True)
    y=d.label.to_numpy(); rows=[];strategies=['stratified','unstratified']
    for n in [10,20,50,100,200]:
        for strategy in strategies:
            for i in range(args.reps):
                # Shared sampling across all models and methods, modified vs original historical replay RNG behavior.
                seed=draw_seed(args.seed,'KermanyNorm',strategy,n,i)
                for m in MODELS:
                    for row in run_one(y,q[m],lock['models'][m]['threshold'],n,strategy,seed,draw_id=i): rows.append({**row,'model':m,'evidence_tier':'HISTORICAL_POST_HOC_EXTENSION'})
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'STAGE07_POSTHOC_DRAW_LEVEL_RESULTS.csv',index=False)
    from stage07_statistics import describe
    summaries=[];failures=[]
    for (model,n,strategy,method),g in frame.groupby(['model','n','strategy','method'],sort=True):
        entry={'model':model,'n':int(n),'strategy':strategy,'method':method,
               'attempted':len(g),'successful':int((g.status=='SUCCESS').sum()),
               'failed':int((g.status!='SUCCESS').sum()),
               'failure_rate':float((g.status!='SUCCESS').mean())}
        for metric in ['ECE15','BA','Se','Sp','Brier','NLL']:
            for k,v in describe(g.loc[g.status=='SUCCESS',metric].to_numpy(dtype=float)).items():entry[metric+'_'+k]=v
        summaries.append(entry)
        for reason,count in g.loc[g.status!='SUCCESS','status'].value_counts().items():
            failures.append({'model':model,'n':n,'strategy':strategy,'method':method,'reason':reason,'count':int(count),'attempted':len(g)})
    pd.DataFrame(summaries).to_csv(out/'STAGE07_POSTHOC_METRIC_DISTRIBUTIONS.csv',index=False)
    pd.DataFrame(failures,columns=['model','n','strategy','method','reason','count','attempted']).to_csv(out/'STAGE07_POSTHOC_FIT_FAILURES.csv',index=False)
    comparison=[]
    key=['model','n','strategy','draw_id','eval_id_hash']
    fr=frame.loc[frame.method=='frozen',key+['BA','Se','Sp','ECE15','Brier']].rename(columns={m:m+'_frozen' for m in ['BA','Se','Sp','ECE15','Brier']})
    merged=frame.merge(fr,on=key,how='left',validate='many_to_one')
    for metric in ['BA','Se','Sp','ECE15','Brier']:
        merged['delta_'+metric]=merged[metric]-merged[metric+'_frozen']
    cols=key+['method','status']+['delta_'+m for m in ['BA','Se','Sp','ECE15','Brier']]
    merged[cols].to_csv(out/'STAGE07_POSTHOC_PAIRED_DELTAS.csv',index=False)

    report['status']='HISTORICAL_POST_HOC_EXTENSION';report['draw_attempts']=args.reps;report['strategies']=strategies;report['replay_exactness']='NOT_EXACT_ORIGINAL_RNG_MODEL_LOOP_ORDER';report['draw_csv_sha256']=sha256(out/'STAGE07_POSTHOC_DRAW_LEVEL_RESULTS.csv')
    (out/'STAGE07_POSTHOC_MANIFEST.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));return 0
if __name__=='__main__':sys.exit(main())
