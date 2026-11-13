"""Never infer corrected pipeline identity from a historical file path."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
MODELS=('inception_v3','swin_b','mean_fusion','gated_fusion','const_alpha','logit_stack')
def sha256_bytes(b):return hashlib.sha256(b).hexdigest()
def validate(d,cohort='KermanyNorm'):
    if cohort!='KermanyNorm':raise ValueError('KermanyRaw/Norm mixing forbidden')
    if not {'id','label','inception_v3','swin_b','mean_fusion','gated_fusion'}.issubset(d.columns):raise ValueError('missing input columns')
    if d.id.isna().any() or d.id.duplicated().any():raise ValueError('duplicate/missing IDs')
    if d.label.isna().any() or not d.label.isin([0,1]).all():raise ValueError('invalid labels')
    for col in ('inception_v3','swin_b','mean_fusion','gated_fusion'):
        p=np.asarray(d[col],float)
        if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('invalid probabilities')
    return True

def six_scores(d,lock):
    from stage07_calibrators import logit
    from scipy.special import expit
    validate(d)
    pi,ps=d.inception_v3.to_numpy(),d.swin_b.to_numpy(); alpha=lock['const_alpha_inception'];s=lock['logit_stack']
    raw={'inception_v3':pi,'swin_b':ps,'mean_fusion':d.mean_fusion.to_numpy(),
         'gated_fusion':d.gated_fusion.to_numpy(),
         'const_alpha':alpha*pi+(1-alpha)*ps,
         'logit_stack':expit(s['intercept']+s['coef_logit_inception']*logit(pi)+s['coef_logit_swin']*logit(ps))}
    return {m:expit(logit(raw[m])/float(lock['models'][m]['temperature'])) for m in MODELS}
