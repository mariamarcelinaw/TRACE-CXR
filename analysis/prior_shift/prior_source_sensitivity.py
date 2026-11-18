"""Sensitivity of the prior-shift correction to the definition of the source prior pi_s (paper Sec. IV-E).

usage: python prior_source_sensitivity.py FOLDER OUT.csv
FOLDER holds the private files TRACE_CXR_CONTROL_LOCK.json, TRACE_CXR_CALIBRATION_PREDICTIONS.csv and
TRACE_CXR_KERMANY_NORM_PREDICTIONS.csv (not distributed). No model is refitted: the frozen calibrated scores are
prior-corrected (Saerens et al., 2002) with two choices of pi_s,
  mean_score  : mean calibrated score on the RSNA calibration split (the paper's primary choice),
  observed    : observed prevalence of the RSNA calibration split,
and two choices of the target prior pi_t: the label-free EM estimate and the true Kermany prevalence
("true-prevalence reference"; named ECE_oracle in the historical scripts). Only aggregate values are written.
Requires numpy and pandas.
"""
import json
import sys

import numpy as np
import pandas as pd

CANDS = ["inception_v3", "swin_b", "mean_fusion", "gated_fusion", "const_alpha", "logit_stack"]


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def apply_temperature(p, t):
    return 1 / (1 + np.exp(-logit(p) / max(float(t), 1e-6)))


def ece15(y, p, bins=15):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    v = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            v += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(v)


def prior_adjust(q, pi_s, pi_t):  # identical to training/trace_cxr_shift_analysis.py
    q = np.clip(q, 1e-7, 1 - 1e-7)
    a = (pi_t / pi_s) * q
    b = ((1 - pi_t) / (1 - pi_s)) * (1 - q)
    return a / (a + b)


def em_prior(q, pi_s, iters=1000, tol=1e-8):  # identical to training/trace_cxr_shift_analysis.py
    pt = pi_s
    for _ in range(iters):
        new = prior_adjust(q, pi_s, pt).mean()
        if abs(new - pt) < tol:
            break
        pt = new
    return pt


def main(folder, out_csv):
    lock = json.load(open(f"{folder}/TRACE_CXR_CONTROL_LOCK.json"))
    cal = pd.read_csv(f"{folder}/TRACE_CXR_CALIBRATION_PREDICTIONS.csv")
    ker = pd.read_csv(f"{folder}/TRACE_CXR_KERMANY_NORM_PREDICTIONS.csv")

    def raw(d, m):
        pi, ps = d.inception_v3.values, d.swin_b.values
        if m == "const_alpha":
            a = lock["const_alpha_inception"]
            return a * pi + (1 - a) * ps
        if m == "logit_stack":
            s = lock["logit_stack"]
            return 1 / (1 + np.exp(-(s["intercept"] + s["coef_logit_inception"] * logit(pi)
                                     + s["coef_logit_swin"] * logit(ps))))
        return d[m].values

    calq = lambda d, m: apply_temperature(raw(d, m), lock["models"][m]["temperature"])
    yc, yk = cal.label.values.astype(int), ker.label.values.astype(int)
    rows = []
    for m in CANDS:
        qc, qk = calq(cal, m), calq(ker, m)
        for name, pi_s in [("mean_score", float(qc.mean())), ("observed", float(yc.mean()))]:
            pt = em_prior(qk, pi_s)
            rows.append(dict(model=m, pi_s_definition=name, pi_s=round(pi_s, 4), pi_t_EM=round(pt, 4),
                             ECE15_frozen=round(ece15(yk, qk), 4),
                             ECE15_EM=round(ece15(yk, prior_adjust(qk, pi_s, pt)), 4),
                             ECE15_true_prevalence_reference=round(ece15(yk, prior_adjust(qk, pi_s, yk.mean())), 4),
                             n_calibration=len(yc), n_kermany=len(yk)))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
