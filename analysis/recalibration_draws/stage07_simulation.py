"""Optional future retrospective extension. All draws and fitting failures retained."""
import hashlib
import numpy as np
from stage07_sampling import sample, draw_seed
from stage07_metrics import metrics
from stage07_thresholds import fast_thr
from stage07_calibrators import intercept_only,full_logistic,predict
from stage07_failure_tracking import status_for_error

def run_one(y,q,t,n,strategy,seed,cohort='KermanyNorm',draw_id=0):
    y=np.asarray(y,int);q=np.asarray(q,float)
    adapt,ev=sample(y,n,strategy,seed,prev=float(y.mean()) if strategy=='stratified' else None)
    if len(ev)==0:raise ValueError('empty evaluation')
    adapt_hash=hashlib.sha256(','.join(map(str,sorted(adapt))).encode()).hexdigest()
    eval_hash=hashlib.sha256(','.join(map(str,sorted(ev))).encode()).hexdigest()
    base={'n':n,'strategy':strategy,'draw_id':draw_id,'draw_seed':seed,'n_pos_adapt':int(y[adapt].sum()),'n_neg_adapt':int(n-y[adapt].sum()),'adapt_id_hash':adapt_hash,'eval_id_hash':eval_hash,'eval_size':len(ev)}
    frozen=metrics(y[ev],q[ev],t)
    rows=[{**base,'method':'frozen','status':'SUCCESS',**frozen,'intercept':None,'slope':None,'threshold':t}]
    if n==0:return rows
    for method in ('threshold','intercept_only','full_logistic'):
        try:
            if method=='threshold':
                p=q[ev];tau=fast_thr(q[adapt],y[adapt]);pars={'intercept':None,'slope':None}
            elif method=='intercept_only':
                pars=intercept_only(q[adapt],y[adapt]);p=predict(q[ev],pars);tau=t
            else:
                pars=full_logistic(q[adapt],y[adapt]);p=predict(q[ev],pars);tau=float(y[adapt].mean())
            row={**base,'method':method,'status':'SUCCESS',**metrics(y[ev],p,tau), 'intercept':pars['intercept'],'slope':pars['slope'],'threshold':tau}
            if pars['slope'] is not None and pars['slope']<0:row['diagnostic']='NEGATIVE_SLOPE'
        except Exception as e:
            row={**base,'method':method,'status':status_for_error(e),'error_message':str(e),'BA':None,'Se':None,'Sp':None,'ECE15':None,'Brier':None,'NLL':None,'intercept':None,'slope':None,'threshold':None}
        rows.append(row)
    return rows
