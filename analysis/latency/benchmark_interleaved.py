#!/usr/bin/env python3
"""TRACE-CXR interleaved within-image six-candidate latency benchmark (Section IV-F).
This program never trains, optimizes, or modifies checkpoints or source evidence.
Historical best.pt fallback is explicit and separately identified; output parity is a gate.
"""
import argparse,csv,hashlib,json,os,platform,random,sys,time,traceback,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
REF=HERE/'reference'  # private inputs, see README.md
sys.path.insert(0,str(REF))
T=None  # deferred import: --help works before optional GPU/DICOM dependencies are installed

CANDIDATES=['inception_v3','swin_b','mean_fusion','const_alpha','gated_fusion','logit_stack']
HIST_SHA={'inception_v3':'94242722f60991d82cb114aba2c3d7f5f9ecb0af89845024b1ce69386b193c5b','swin_b':'6d6e8d821effa0fce07913e10f34671cf420fc9c0dceaea8c04bae4e1089f6b9','gate':'6897e8274bb34a5a3e5e695359b1d66ed39b72198dac497f6b7d3473383a50e0'}

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def atomic_json(path,obj):
 p=Path(path); p.write_text(json.dumps(obj,indent=2,default=str,ensure_ascii=False)+'\n')

def sync(dev):
 if dev.type=='cuda':torch.cuda.synchronize(dev)

def _ckpt_path(name,source,allow_best):
 primary=source/T.final_model_filename(name)
 if primary.is_file():
  h=digest(primary)
  if h!=HIST_SHA[name]:raise RuntimeError(f'Historical original file SHA mismatch: {primary}: {h}')
  return primary,'ORIGINAL_FINAL_PT_HASH_VERIFIED',h
 if not allow_best:raise FileNotFoundError(f'Missing historical original {primary}; add --allow-historical-best-pt only if choosing historical best.pt fallback explicitly')
 fallback=source/'checkpoints'/'final'/name/'best.pt'
 if not fallback.is_file():raise FileNotFoundError(f'Missing fallback {fallback}')
 done=json.loads((REF/f'done_{name}.json').read_text());h=digest(fallback)
 if h!=done['best_checkpoint_sha256']:raise RuntimeError(f'Best.pt SHA differs from archived done.json for {name}: {h}')
 return fallback,'HISTORICAL_BEST_PT_NOT_BINARY_EQUIVALENT_TO_FINAL_WRAPPER',h

def _safe_weights(path):
 # no use of historical T.safe_torch_load: it can rename a checkpoint upon exception
 return torch.load(path,map_location='cpu',weights_only=True)

def load_models(source,allow_best,dev):
 info=[]; mods={}
 for name in ['inception_v3','swin_b']:
  p,provenance,h=_ckpt_path(name,source,allow_best)
  model=T.build_model(name,pretrained=False)
  obj=_safe_weights(p)
  if not isinstance(obj,dict):raise RuntimeError(f'Checkpoint {name} must be dict')
  sd=obj.get('model_state_dict',obj)
  model.load_state_dict(sd,strict=True)
  if name=='inception_v3' and not model.transform_input:raise RuntimeError('transform_input must be True')
  model=model.to(dev).eval();mods[name]=model
  info.append(dict(name=name,path=str(p),sha256=h,provenance=provenance,parameters=sum(p.numel() for p in model.parameters())))
 gp=REF/'TRACE_CXR_RELIABILITY_GATE.pt'; gh=digest(gp)
 if gh!=HIST_SHA['gate']:raise RuntimeError('Historical gate archive hash mismatch')
 cfg=T.load_cfg(REF/'trace_cxr.yaml')
 gate=T.ReliabilityGate(cfg['fusion']['hidden_dim']).eval()
 gsd=_safe_weights(gp);gate.load_state_dict(gsd.get('model_state_dict',gsd),strict=True)
 info.append(dict(name='gate',path=str(gp),sha256=gh,provenance='ORIGINAL_GATE_PT_HASH_VERIFIED',parameters=sum(p.numel() for p in gate.parameters())))
 return mods,gate,info

def sample_paths(n,warmup):
 cfg=T.load_cfg(REF/'trace_cxr.yaml');man=pd.read_csv(REF/'TRACE_CXR_RSNA_SPLIT_MANIFEST.csv')
 locked=man.loc[man['split']=='locked_test'].copy()
 if n+warmup>len(locked):raise RuntimeError('n+warmup exceeds locked-test population')
 # exact sampling semantics of historical measure_inference_time.py
 drawn=locked['path'].sample(n=n+warmup,random_state=cfg['experiment']['seed']).tolist()
 by_path=locked.set_index('path'); rows=[]
 for ix,p in enumerate(drawn):
  obj=by_path.loc[p]
  rows.append(dict(sample_order=ix,image_id=str(obj['id']),path=p,warmup=ix<warmup,split='locked_test',source='RECONSTRUCTED_FROM_HISTORICAL_SAMPLING_CODE'))
 return rows

def calibration(raw,model,lock):
 z=np.log(np.clip(float(raw),1e-6,1-1e-6)/np.clip(1-float(raw),1e-6,1-1e-6))
 temp=float(lock['models'][model]['temperature']); th=float(lock['models'][model]['threshold'])
 q=1/(1+np.exp(-np.clip(z/temp,-80,80)))
 return float(q),int(q>=th)

def postprocess(pi,ps,candidate,lock,gate):
 if candidate=='inception_v3':p=float(pi)
 elif candidate=='swin_b':p=float(ps)
 elif candidate=='mean_fusion':p=0.5*(float(pi)+float(ps))
 elif candidate=='const_alpha':
  a=float(lock['const_alpha_inception']);p=a*float(pi)+(1-a)*float(ps)
 elif candidate=='gated_fusion':
  feats=torch.tensor(T.gate_features(np.array([pi]),np.array([ps])),dtype=torch.float32)
  with torch.inference_mode():a=float(gate(feats).item())
  p=a*float(pi)+(1-a)*float(ps)
 elif candidate=='logit_stack':
  d=lock['logit_stack'];lip=np.log(np.clip(pi,1e-6,1-1e-6)/np.clip(1-pi,1e-6,1-1e-6));lsp=np.log(np.clip(ps,1e-6,1-1e-6)/np.clip(1-ps,1e-6,1-1e-6))
  zp=d['intercept']+d['coef_logit_inception']*lip+d['coef_logit_swin']*lsp
  p=float(1/(1+np.exp(-zp)))
 else:raise ValueError(candidate)
 q,decision=calibration(p,candidate,lock)
 return p,q,decision

def process_one(path,candidate,models,gate,lock,tf,dev,round_no,image_id,timed=True):
 require_inc=candidate!='swin_b';require_swin=candidate!='inception_v3'
 sync(dev);t0=time.perf_counter_ns()
 image=T.read_cxr(path);t1=time.perf_counter_ns()
 x=tf(image).unsqueeze(0);t2=time.perf_counter_ns()
 xg=x.to(dev);sync(dev);t3=time.perf_counter_ns()
 gpu_events={};pi=ps=None
 with torch.inference_mode():
  if require_inc:
   start=end=None
   if dev.type=='cuda':start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
   zi,_=T.forward_logits(models['inception_v3'],xg,'inception_v3')
   if end is not None:end.record()
   sync(dev);te=time.perf_counter_ns()
   gpu_events['inception_cuda_event_ms']=start.elapsed_time(end) if end is not None else None
   inc_ms=(te-t3)/1e6
  else:
   te=t3;inc_ms=0.0;gpu_events['inception_cuda_event_ms']=None
  if require_swin:
   start=end=None
   if dev.type=='cuda':start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
   zs,_=T.forward_logits(models['swin_b'],xg,'swin_b')
   if end is not None:end.record()
   sync(dev);ts=time.perf_counter_ns()
   gpu_events['swin_cuda_event_ms']=start.elapsed_time(end) if end is not None else None
   swin_ms=(ts-te)/1e6
  else:
   ts=te;swin_ms=0.0;gpu_events['swin_cuda_event_ms']=None
  if require_inc:pi=float(torch.sigmoid(zi).float().cpu().numpy()[0])
  if require_swin:ps=float(torch.sigmoid(zs).float().cpu().numpy()[0])
  sync(dev);td=time.perf_counter_ns()
  p,q,decision=postprocess(pi,ps,candidate,lock,gate)
  sync(dev);tend=time.perf_counter_ns()
 components={'read_decode_ms':(t1-t0)/1e6,'preprocessing_ms':(t2-t1)/1e6,'h2d_transfer_ms':(t3-t2)/1e6,
             'inception_forward_ms':inc_ms,'swin_forward_ms':swin_ms,'d2h_transfer_ms':(td-ts)/1e6,
             'postprocessing_ms':(tend-td)/1e6}
 return dict(candidate=candidate,round=round_no,image_id=image_id,path=path,total_end_to_end_ms=(tend-t0)/1e6,
             **components,**gpu_events,raw_probability=p,calibrated_probability=q,binary_decision=decision,
             inception_raw=pi,swin_raw=ps,component_sum_ms=sum(components.values()))

def parity_check(rows,models,gate,lock,tf,dev,max_cases=12,tol=0.0005):
 archived=pd.read_csv(REF/'TRACE_CXR_INTERNAL_PREDICTIONS.csv').set_index('id')
 checks=[]
 for row in rows[:max_cases]:
  id=row['image_id'];path=row['path']
  x=tf(T.read_cxr(path)).unsqueeze(0).to(dev)
  with torch.inference_mode():
   zi,_=T.forward_logits(models['inception_v3'],x,'inception_v3')
   zs,_=T.forward_logits(models['swin_b'],x,'swin_b')
   sync(dev)
   pi=float(torch.sigmoid(zi).cpu().numpy()[0]);ps=float(torch.sigmoid(zs).cpu().numpy()[0])
   raw,_,_=postprocess(pi,ps,'gated_fusion',lock,gate)
  if id not in archived.index:raise RuntimeError(f'Archived reference missing id {id}')
  ref=archived.loc[id]
  for name,actual,col in [('inception_v3',pi,'inception_v3'),('swin_b',ps,'swin_b'),('gated_fusion',raw,'gated_fusion')]:
   d=abs(float(actual)-float(ref[col]))
   checks.append(dict(image_id=id,branch=name,measured=actual,archived=float(ref[col]),absolute_error=d,
                      tolerance=tol,status='PASS' if d<=tol else 'FAIL'))
 return checks

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--root',required=True,help='historical TRACE_CXR project directory containing outputs/trace_cxr_inception_swinb_v2')
 ap.add_argument('--out',default='STAGE11_INTERLEAVED_GPU_RUN_02')
 ap.add_argument('--n',type=int,default=200)
 ap.add_argument('--warmup',type=int,default=20)
 ap.add_argument('--rounds',type=int,default=3)
 ap.add_argument('--parity-cases',type=int,default=12)
 ap.add_argument('--parity-tolerance',type=float,default=5e-4)
 ap.add_argument('--allow-historical-best-pt',action='store_true',help='explicit fallback, sha checked against done.json; not binary-equivalent to published final pt')
 ap.add_argument('--preflight-only',action='store_true')
 args=ap.parse_args()
 global T
 import trace_cxr as T
 out=Path(args.out).resolve()
 if out.exists():raise FileExistsError(f'Output path already exists, refusing overwrite: {out}')
 if args.n<1 or args.rounds<1 or args.warmup<1:raise ValueError('n,rounds,warmup positive')
 out.mkdir(parents=True)
 try:
  src=Path(args.root).expanduser().resolve()/'outputs'/'trace_cxr_inception_swinb_v2'
  lock=json.loads((REF/'TRACE_CXR_CONTROL_LOCK.json').read_text())
  samples=sample_paths(args.n,args.warmup)
  pd.DataFrame(samples).to_csv(out/'STAGE11_IMAGE_SAMPLE_MANIFEST.csv',index=False)
  source_ckpt=[]
  for name in ['inception_v3','swin_b']:
   p,tag,h=_ckpt_path(name,src,args.allow_historical_best_pt)
   source_ckpt.append(dict(name=name,path=str(p),provenance=tag,sha256=h,bytes=p.stat().st_size))
  missing=[r for r in samples if not Path(r['path']).is_file()]
  atomic_json(out/'STAGE11_PREFLIGHT.json',dict(source_ckpt=source_ckpt,source_image_count=len(samples),missing_images=missing[:20],missing_n=len(missing),control_lock_sha256=digest(REF/'TRACE_CXR_CONTROL_LOCK.json'),gpu_available=torch.cuda.is_available()))
  if missing:raise FileNotFoundError(f'{len(missing)} input DICOM paths unavailable; see preflight.json')
  if args.preflight_only:
   atomic_json(out/'STAGE11_COMPLETION.json',{'status':'PREFLIGHT_ONLY_NO_GPU_BENCHMARK','message':'Checkpoint and image paths verified; run without --preflight-only'});return
  if not torch.cuda.is_available():raise RuntimeError('GPU with CUDA required for matched historical benchmark; no CPU timing substitution')
  dev=torch.device('cuda');cfg=T.load_cfg(REF/'trace_cxr.yaml');T.set_seed(cfg['experiment']['seed'])
  models,gate,info=load_models(src,args.allow_historical_best_pt,dev)
  tf=T.build_transforms(cfg['experiment']['image_size'],'inception_v3',train=False)
  hardware=dict(gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,torch=torch.__version__,python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,gpu_device_count=torch.cuda.device_count(),batch_size=1,precision='float32; no autocast',torch_inference_mode=True,model_allocation='both backbones resident for all candidates; single-branch path skips other forward',warmup_per_candidate=args.warmup,rounds=args.rounds,timed_n=args.n,parity_tolerance=args.parity_tolerance)
  try:
   smi=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,temperature.gpu,utilization.gpu,clocks.sm','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=12,check=False)
   hardware['nvidia_smi_snapshot']=smi.stdout.strip() if smi.returncode==0 else None
  except (OSError,subprocess.TimeoutExpired):hardware['nvidia_smi_snapshot']=None
  atomic_json(out/'STAGE11_HARDWARE_SOFTWARE_MANIFEST.json',hardware)
  pd.DataFrame(info).to_csv(out/'STAGE11_CHECKPOINT_VERIFICATION.csv',index=False)
  parity=parity_check([r for r in samples if not r['warmup']],models,gate,lock,tf,dev,args.parity_cases,args.parity_tolerance)
  pd.DataFrame(parity).to_csv(out/'STAGE11_MODEL_OUTPUT_PARITY.csv',index=False)
  if any(x['status']=='FAIL' for x in parity):raise RuntimeError('Model probability parity failed; timing not run, archived output semantics incompatible')
  warmups=[r for r in samples if r['warmup']]; images=[r for r in samples if not r['warmup']]
  order_register=[];out_rows=[];memrows=[];start=time.monotonic()
  # Follow-up to original matched blockwise benchmark: within-image interleaving.
  # Each candidate is in every within-image position ~equally often; all six
  # decode the SAME DICOM per image. Warm-ups still run outside timed data.
  for rnd in range(args.rounds):
   base_order=CANDIDATES.copy();random.Random(20260908+rnd).shuffle(base_order)
   for cand in base_order:
    for r in warmups:
     process_one(r['path'],cand,models,gate,lock,tf,dev,rnd,r['image_id'],timed=False)
   sync(dev)
   torch.cuda.reset_peak_memory_stats(dev)
   for img_ix,r in enumerate(images):
    shift=(rnd*len(images)+img_ix)%len(CANDIDATES)
    order=base_order[shift:]+base_order[:shift]
    for pos,cand in enumerate(order):
     order_register.append(dict(round=rnd,image_id=r['image_id'],candidate=cand,order_within_image=pos,rotation=shift))
     d=process_one(r['path'],cand,models,gate,lock,tf,dev,rnd,r['image_id'])
     out_rows.append(d)
   memrows.append(dict(round=rnd,candidate='ALL_CANDIDATES_SHARED_GPU_RESIDENCY',peak_allocated_bytes=torch.cuda.max_memory_allocated(dev),peak_reserved_bytes=torch.cuda.max_memory_reserved(dev),allocated_after_bytes=torch.cuda.memory_allocated(dev),note='whole_round; both_backbones_resident; NOT isolated model memory'))
   print(f'Completed INTERLEAVED benchmark round {rnd+1}/{args.rounds}',flush=True)
  pd.DataFrame(order_register).to_csv(out/'STAGE11_BENCHMARK_REPETITION_REGISTER.csv',index=False)
  pd.DataFrame(out_rows).to_csv(out/'STAGE11_SIX_CANDIDATE_RAW_TIMINGS.csv',index=False)
  pd.DataFrame(memrows).to_csv(out/'STAGE11_GPU_MEMORY_PROFILE.csv',index=False)
  if len(out_rows)!=len(CANDIDATES)*args.rounds*args.n:raise RuntimeError('Unexpected timing row count')
  atomic_json(out/'STAGE11_COMPLETION.json',dict(status='MATCHED_INTERLEAVED_GPU_BENCHMARK_EXECUTED_PENDING_INDEPENDENT_AUDIT',n_images=args.n,rounds=args.rounds,observations=len(out_rows),candidate_names=CANDIDATES,all_parity_pass=True,checkpoint_provenance=[x['provenance'] for x in info],runtime_seconds=time.monotonic()-start))
 except BaseException as e:
  atomic_json(out/'STAGE11_RUN_ERROR.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()[-9000:]})
  raise

if __name__=='__main__':main()
