"""Logistic recalibrators, with explicit method divergence from historical fit."""
import warnings
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

def logit(p,eps=1e-6):
    p=np.clip(np.asarray(p,float),eps,1-eps)
    return np.log(p/(1-p))

def intercept_only(q,y):
    z=logit(q); y=np.asarray(y,int)
    if len(np.unique(y))<2:raise ValueError('SINGLE_CLASS_ADAPTATION')
    def loss(a):
        u=expit(z+a);u=np.clip(u,1e-7,1-1e-7)
        return float(-np.mean(y*np.log(u)+(1-y)*np.log(1-u)))
    r=minimize_scalar(loss,bounds=(-20,20),method='bounded',options={'xatol':1e-9})
    if not r.success:raise RuntimeError('NONCONVERGENCE')
    return {'intercept':float(r.x),'slope':1.0,'solver':'scipy-bounded','regularization':'none','converged':True,'n_iter':int(r.nfev),'method':'intercept_only'}

def full_logistic(q,y,C=1.0):
    y=np.asarray(y,int)
    if len(np.unique(y))<2:raise ValueError('SINGLE_CLASS_ADAPTATION')
    if not np.isfinite(C) or C<=0:raise ValueError('invalid C')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always',ConvergenceWarning)
        fit=LogisticRegression(C=C,solver='lbfgs',max_iter=1000,tol=1e-4,penalty='l2',fit_intercept=True).fit(logit(q).reshape(-1,1),y)
    converged=not any(issubclass(x.category,ConvergenceWarning) for x in w)
    if not converged:raise RuntimeError('NONCONVERGENCE')
    return {'intercept':float(fit.intercept_[0]),'slope':float(fit.coef_[0,0]),'solver':'lbfgs','C':float(C),'max_iter':1000,'tol':1e-4,'converged':True,'n_iter':int(fit.n_iter_[0]),'method':'full_logistic'}

def positive_slope_logistic(q,y):
    y=np.asarray(y,int)
    if len(np.unique(y))<2:raise ValueError('SINGLE_CLASS_ADAPTATION')
    z=logit(q)
    def fun(ab):
        a,b=ab;u=expit(a+b*z);u=np.clip(u,1e-7,1-1e-7)
        return float(-np.mean(y*np.log(u)+(1-y)*np.log(1-u)))
    fit=minimize(fun,x0=[0,1],method='L-BFGS-B',bounds=[(-20,20),(1e-5,20)],options={'maxiter':1000,'ftol':1e-10})
    if not fit.success:raise RuntimeError('NONCONVERGENCE')
    return {'intercept':float(fit.x[0]),'slope':float(fit.x[1]),'solver':'scipy-L-BFGS-B','regularization':'none','converged':True,'n_iter':int(fit.nit),'method':'positive_slope_sensitivity'}

def predict(q,fit):return expit(float(fit['intercept'])+float(fit['slope'])*logit(q))
