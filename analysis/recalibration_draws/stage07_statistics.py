"""Empirical conditional-on-cohort draw summaries, not population CI."""
import numpy as np
def describe(v):
    a=np.asarray(v,float);a=a[np.isfinite(a)]
    if len(a)==0:return {'valid_draws':0,'mean':None,'median':None,'sd':None,'p2_5':None,'p25':None,'p75':None,'p97_5':None,'min':None,'max':None}
    p=np.percentile(a,[2.5,25,75,97.5]);return {'valid_draws':len(a),'mean':float(a.mean()),'median':float(np.median(a)), 'sd':float(a.std(ddof=1)) if len(a)>1 else None, 'p2_5':float(p[0]),'p25':float(p[1]),'p75':float(p[2]),'p97_5':float(p[3]),'min':float(a.min()),'max':float(a.max())}

def paired(a,b):
    if set(a)!=set(b):raise ValueError('unpaired draw IDs')
    return {k:a[k]-b[k] for k in sorted(a)}
