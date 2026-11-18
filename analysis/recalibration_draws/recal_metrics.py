"""Evaluation metrics; never uses adaptation labels to fit a model."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def ece15(y, p, bins=15):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    edges=np.linspace(0,1,bins+1);out=0.0
    for i in range(bins):
        mask=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        if mask.any():out+=float(mask.mean()*abs(y[mask].mean()-p[mask].mean()))
    return out

def metrics(y,p,t):
    y=np.asarray(y,dtype=int); p=np.asarray(p,float)
    if len(y)==0 or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('invalid evaluation arrays')
    pred=p>=float(t); pos=y==1; neg=~pos
    se=float(np.mean(pred[pos])) if pos.any() else np.nan
    sp=float(np.mean(~pred[neg])) if neg.any() else np.nan
    ba=float((se+sp)/2) if pos.any() and neg.any() else np.nan
    ece=ece15(y,p)
    brier=float(np.mean((p-y)**2)); q=np.clip(p,1e-7,1-1e-7)
    nll=float(-np.mean(y*np.log(q)+(1-y)*np.log(1-q)))
    tp=np.sum(pred&pos);fp=np.sum(pred&neg);fn=np.sum((~pred)&pos)
    f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn>0 else 0.0
    return {'BA':ba,'Se':se,'Sp':sp,'ECE15':ece,'Brier':brier,'NLL':nll,'F1':f1,
            'ROC_AUC':float(roc_auc_score(y,p)) if pos.any() and neg.any() else np.nan,
            'AP':float(average_precision_score(y,p)) if pos.any() else np.nan,
            'predicted_positive_fraction':float(pred.mean())}
