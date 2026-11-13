"""Deterministic per-cohort per-strategy per-budget draw streams."""
import hashlib
import numpy as np

def draw_seed(base,cohort,strategy,n,draw_id):
    b=f'{int(base)}|{cohort}|{strategy}|{int(n)}|{int(draw_id)}'.encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8],'big')

def sample(y,n,strategy,seed,prev=None):
    y=np.asarray(y,dtype=int); n=int(n)
    if n<0 or n>=len(y):raise ValueError('invalid adaptation budget')
    rng=np.random.default_rng(int(seed)); idx=np.arange(len(y))
    if n==0:return np.array([],dtype=int),idx
    if strategy=='stratified':
        if prev is None:raise ValueError('retrospective prevalence must be declared')
        np1=int(round(n*float(prev))); n0=n-np1
        pp=idx[y==1];nn=idx[y==0]
        if np1>len(pp) or n0>len(nn):raise ValueError('insufficient class counts')
        a=np.r_[rng.choice(pp,np1,replace=False),rng.choice(nn,n0,replace=False)]
    elif strategy=='unstratified':
        a=rng.choice(idx,n,replace=False)
    else:raise ValueError('unknown strategy')
    ev=np.setdiff1d(idx,a,assume_unique=True)
    if len(a)!=n or len(set(a))!=n or np.intersect1d(a,ev).size:raise RuntimeError('sampling leakage')
    return a,ev
