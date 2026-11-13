"""Gate training-duration check (dev OOF only, no test data).
Usage: python gate_convergence_check.py <folder with TRACE_CXR_OOF_FUSION.csv and TRACE_CXR_RSNA_SPLIT_MANIFEST.csv>
Cross-fits over the five OOF folds: 60 AdamW steps (lr 1e-3, wd 1e-4) vs. L-BFGS to convergence vs. constant alpha.
NumPy reimplementation of the 5-16-1 gate; the 60-step setting reproduces the archived gate (held-out AUC 0.8894, alpha 0.395-0.503)."""
import sys,csv,numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
E=sys.argv[1]
fold={r['id']:r['fold'] for r in csv.DictReader(open(f"{E}/TRACE_CXR_RSNA_SPLIT_MANIFEST.csv")) if r['split']=='development'}
R=list(csv.DictReader(open(f"{E}/TRACE_CXR_OOF_FUSION.csv")))
y=np.array([int(r['label']) for r in R],float); pI=np.array([float(r['inception_v3']) for r in R]); pS=np.array([float(r['swin_b']) for r in R])
F=np.array([int(float(fold[r['id']])) for r in R])
H=lambda p:-(p*np.log(np.clip(p,1e-12,1))+(1-p)*np.log(np.clip(1-p,1e-12,1)))
X=np.column_stack([pI,pS,H(pI),H(pS),np.abs(pI-pS)])
def unpack(w): W1=w[:80].reshape(16,5);b1=w[80:96];w2=w[96:112];b2=w[112];return W1,b1,w2,b2
def forward(w,X,pI,pS):
    W1,b1,w2,b2=unpack(w); h=X@W1.T+b1; a=np.maximum(h,0); z=a@w2+b2; al=1/(1+np.exp(-z)); p=al*pI+(1-al)*pS
    return h,a,al,np.clip(p,1e-7,1-1e-7)
def loss_grad(w,X,pI,pS,y,wd=1e-4):
    h,a,al,p=forward(w,X,pI,pS); N=len(y)
    L=-np.mean(y*np.log(p)+(1-y)*np.log(1-p))+0.5*wd*np.sum(w**2)
    dp=(p-y)/(p*(1-p))/N; dz=dp*(pI-pS)*al*(1-al)
    W1,b1,w2,b2=unpack(w); gw2=a.T@dz; gb2=dz.sum(); da=np.outer(dz,w2)*(h>0); gW1=da.T@X; gb1=da.sum(0)
    g=np.concatenate([gW1.ravel(),gb1,gw2,[gb2]])+wd*w; return L,g
def init(seed):
    r=np.random.default_rng(seed); k1=1/np.sqrt(5); k2=1/4
    return np.concatenate([r.uniform(-k1,k1,80),r.uniform(-k1,k1,16),r.uniform(-k2,k2,16),r.uniform(-k2,k2,1)])
def adam(w,args,steps=60,lr=1e-3):
    m=np.zeros_like(w);v=np.zeros_like(w)
    for t in range(1,steps+1):
        _,g=loss_grad(w,*args); m=.9*m+.1*g; v=.999*v+.001*g*g
        w=w-lr*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
    return w
res={k:np.zeros(len(y)) for k in ['g60','gconv','const']}; alph={'g60':[],'gconv':[]}
for k in range(5):
    tr=F!=k; te=F==k; args=(X[tr],pI[tr],pS[tr],y[tr])
    w0=init(20260908+k)
    w60=adam(w0.copy(),args); wc=minimize(loss_grad,w0.copy(),args=args,jac=True,method='L-BFGS-B',options={'maxiter':5000}).x
    for name,w in [('g60',w60),('gconv',wc)]:
        _,_,al,p=forward(w,X[te],pI[te],pS[te]); res[name][te]=p; alph[name].append(al)
    grid=np.linspace(0,1,1001); bce=[-np.mean(y[tr]*np.log(np.clip(a*pI[tr]+(1-a)*pS[tr],1e-7,1))+(1-y[tr])*np.log(np.clip(1-(a*pI[tr]+(1-a)*pS[tr]),1e-7,1))) for a in grid]
    a=grid[int(np.argmin(bce))]; res['const'][te]=a*pI[te]+(1-a)*pS[te]
for n,p in res.items():
    bce=-np.mean(y*np.log(np.clip(p,1e-7,1))+(1-y)*np.log(np.clip(1-p,1e-7,1)))
    s=f"{n}: held-out OOF AUC {roc_auc_score(y,p):.4f} BCE {bce:.4f}"
    if n in alph: a=np.concatenate(alph[n]); s+=f" alpha range {a.min():.3f}-{a.max():.3f} sd {a.std():.3f}"
    print(s)
print('swin',round(roc_auc_score(y,pS),4),'mean',round(roc_auc_score(y,(pI+pS)/2),4))
