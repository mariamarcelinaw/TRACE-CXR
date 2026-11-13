"""Replay of the primary local-recalibration run (source of Table IV) with convergence and slope logging.
usage: python replay_primary_recalibration.py FOLDER OUT_DRAWS_CSV
FOLDER holds TRACE_CXR_CONTROL_LOCK.json and TRACE_CXR_KERMANY_NORM_PREDICTIONS.csv (private, not distributed);
if TRACE_CXR_EXT_KermanyNorm_FEWSHOT.csv is also present, the replay is compared with it."""
import os
import json, sys, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
out = sys.argv[1]; out_csv = sys.argv[2]
lock = json.load(open(out + "/TRACE_CXR_CONTROL_LOCK.json"))
d = pd.read_csv(out + "/TRACE_CXR_KERMANY_NORM_PREDICTIONS.csv")
CANDS = ["inception_v3", "swin_b", "mean_fusion", "gated_fusion", "const_alpha", "logit_stack"]
def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps); return np.log(p / (1 - p))
def temp(p, t): return 1 / (1 + np.exp(-logit(p) / max(float(t), 1e-6)))
def ece15(y, p, bins=15):
    e = np.linspace(0, 1, bins + 1); v = 0.0
    for i in range(bins):
        m = (p >= e[i]) & ((p < e[i + 1]) if i < bins - 1 else (p <= e[i + 1]))
        if m.any(): v += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(v)
def light(y, q, t):
    p = q >= t; se = (p & (y == 1)).sum() / (y == 1).sum(); sp = ((~p) & (y == 0)).sum() / (y == 0).sum()
    return (se + sp) / 2, se, sp, ece15(y, q)
def fast_thr(q, y):
    c = np.unique(np.r_[0.0, q, 1.0]); P = (y == 1).sum(); N = (y == 0).sum()
    ba = [(((q >= t) & (y == 1)).sum() / P + ((q < t) & (y == 0)).sum() / N) / 2 for t in c]
    return float(c[int(np.argmax(ba))])
pi, ps = d.inception_v3.values, d.swin_b.values; s = lock["logit_stack"]; a = lock["const_alpha_inception"]
raw = {"inception_v3": pi, "swin_b": ps, "mean_fusion": d.mean_fusion.values, "gated_fusion": d.gated_fusion.values,
       "const_alpha": a * pi + (1 - a) * ps,
       "logit_stack": 1 / (1 + np.exp(-(s["intercept"] + s["coef_logit_inception"] * logit(pi) + s["coef_logit_swin"] * logit(ps))))}
q = {m: temp(raw[m], lock["models"][m]["temperature"]) for m in CANDS}
tau = {m: lock["models"][m]["threshold"] for m in CANDS}
y = d.label.values.astype(int)
rng = np.random.default_rng(20260908); pos, neg, prev = np.where(y == 1)[0], np.where(y == 0)[0], y.mean()
rows, draws, nwarn, overlap = [], [], 0, 0
for m in CANDS:
    z = logit(q[m])
    for n in [0, 10, 20, 50, 100, 200]:
        res = {"none": [], "thr": [], "platt": []}
        for r in range(500 if n else 1):
            mask = np.zeros(len(y), bool)
            if n:
                npos = int(round(n * prev))
                mask[np.r_[rng.choice(pos, npos, replace=False), rng.choice(neg, n - npos, replace=False)]] = True
            ev = ~mask
            overlap += int((mask & ev).any()) + int(mask.sum() != n)
            res["none"].append(light(y[ev], q[m][ev], tau[m]))
            if n:
                res["thr"].append(light(y[ev], q[m][ev], fast_thr(q[m][mask], y[mask])))
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", ConvergenceWarning)
                    lr = LogisticRegression(C=1.0).fit(z[mask].reshape(-1, 1), y[mask])
                nwarn += sum(issubclass(x.category, ConvergenceWarning) for x in w)
                met = light(y[ev], lr.predict_proba(z[ev].reshape(-1, 1))[:, 1], y[mask].mean())
                res["platt"].append(met)
                draws.append(dict(model=m, n=n, draw=r, n_pos=int(y[mask].sum()), intercept=float(lr.intercept_[0]),
                                  slope=float(lr.coef_[0, 0]), n_iter=int(lr.n_iter_[0]), BA=met[0], Se=met[1], Sp=met[2], ECE15=met[3]))
        for k, v in res.items():
            if v:
                A = np.array(v)
                rows.append(dict(model=m, n=n, method=k, BA=A[:, 0].mean(), Se=A[:, 1].mean(), Sp=A[:, 2].mean(), ECE=A[:, 3].mean()))
mine = pd.DataFrame(rows); D = pd.DataFrame(draws)
refp = out + "/TRACE_CXR_EXT_KermanyNorm_FEWSHOT.csv"
if os.path.exists(refp):
    ref = pd.read_csv(refp)
    j = mine.merge(ref, on=["model", "n", "method"], suffixes=("", "_ref"))
    print("rows matched", len(j), "of", len(ref))
    for c in ["BA", "Se", "Sp", "ECE"]: print(c, "max abs diff", float((j[c] - j[c + "_ref"]).abs().max()))
print("convergence warnings", nwarn, "of", len(D), "fits; max n_iter", D.n_iter.max(), "; overlap/size violations", overlap)
print("non-positive slopes by model/n:\n", D[D.slope <= 0].groupby(["model", "n"]).size())
D.to_csv(out_csv, index=False)
