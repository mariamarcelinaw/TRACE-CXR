"""Historical first-maximum balanced-accuracy threshold rule."""
import numpy as np

def fast_thr(q,y):
    q=np.asarray(q,float);y=np.asarray(y,int)
    if not np.isin(y,[0,1]).all() or len(np.unique(y))<2:raise ValueError('single-class threshold fit')
    c=np.unique(np.r_[0.0,q,1.0]);P=(y==1).sum();N=(y==0).sum()
    scores=[(((q>=t)&(y==1)).sum()/P + ((q<t)&(y==0)).sum()/N)/2 for t in c]
    return float(c[int(np.argmax(scores))])
