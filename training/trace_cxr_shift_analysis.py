#!/usr/bin/env python3
"""
TRACE-CXR external shift analysis (CPU only, uses saved predictions; no retraining).

1. Prior-shift correction without labels: EM procedure of Saerens et al. (2002)
   on the calibrated Kermany scores; oracle correction with the true prevalence.
2. Class-conditional shift diagnostics: sensitivity at the frozen RSNA threshold
   (prior-invariant under pure label shift) and two-sample KS distance between the
   RSNA-test and Kermany score distributions of each class.
3. Local recalibration with n labeled Kermany images (n = 10..200, stratified,
   500 repetitions): (a) re-select the BA threshold; (b) two-parameter logistic
   recalibration of the score logit + Bayes BA rule (predict positive if the
   recalibrated probability >= sample prevalence). Always evaluated on the
   remaining Kermany images.

Usage (program directory):  python trace_cxr_shift_analysis.py --out outputs/trace_cxr_inception_swinb_v2
Writes: <out>/TRACE_CXR_SHIFT_PRIOR.csv, <out>/TRACE_CXR_SHIFT_FEWSHOT.csv,
        <out>/paper/shift_numbers.tex, <out>/paper/table5_rows.tex
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression

from trace_cxr_controls import apply_temperature, ece15, logit, metrics

CANDS = ["inception_v3", "swin_b", "mean_fusion", "gated_fusion", "const_alpha", "logit_stack"]
NAME = {"inception_v3": "InceptionV3", "swin_b": "Swin-B", "mean_fusion": "Mean fusion",
        "gated_fusion": "Gated fusion", "const_alpha": "Constant-α", "logit_stack": "Logit stacking"}


def prior_adjust(q, pi_s, pi_t):
    q = np.clip(q, 1e-7, 1 - 1e-7)
    a = (pi_t / pi_s) * q
    b = ((1 - pi_t) / (1 - pi_s)) * (1 - q)
    return a / (a + b)


def em_prior(q, pi_s, iters=1000, tol=1e-8):
    pt = pi_s
    for _ in range(iters):
        new = prior_adjust(q, pi_s, pt).mean()
        if abs(new - pt) < tol:
            break
        pt = new
    return pt


def fast_thr(q, y):
    c = np.unique(np.r_[0.0, q, 1.0]); P = (y == 1).sum(); N = (y == 0).sum()
    ba = [(((q >= t) & (y == 1)).sum() / P + ((q < t) & (y == 0)).sum() / N) / 2 for t in c]
    return float(c[int(np.argmax(ba))])


def light(y, q, t):
    p = q >= t
    se = (p & (y == 1)).sum() / (y == 1).sum(); sp = ((~p) & (y == 0)).sum() / (y == 0).sum()
    return (se + sp) / 2, se, sp, ece15(y, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()
    out = Path(args.out)
    lock = json.loads((out / "TRACE_CXR_CONTROL_LOCK.json").read_text())
    cal = pd.read_csv(out / "TRACE_CXR_CALIBRATION_PREDICTIONS.csv")
    te = pd.read_csv(out / "TRACE_CXR_INTERNAL_PREDICTIONS.csv")
    ke = pd.read_csv(out / "TRACE_CXR_KERMANY_PREDICTIONS.csv")

    def raw(d, m):
        pi, ps = d.inception_v3.values, d.swin_b.values
        if m == "const_alpha":
            a = lock["const_alpha_inception"]; return a * pi + (1 - a) * ps
        if m == "logit_stack":
            s = lock["logit_stack"]
            return 1 / (1 + np.exp(-(s["intercept"] + s["coef_logit_inception"] * logit(pi) + s["coef_logit_swin"] * logit(ps))))
        return d[m].values

    calq = lambda d, m: apply_temperature(raw(d, m), lock["models"][m]["temperature"])
    yk, yt = ke.label.values.astype(int), te.label.values.astype(int)

    # ---------------- 1-2: prior correction and conditional-shift diagnostics
    prior_rows = []
    for m in CANDS:
        pi_s = float(calq(cal, m).mean())
        qk, qt = calq(ke, m), calq(te, m); tau = lock["models"][m]["threshold"]
        base = metrics(yk, qk, tau)
        pt = em_prior(qk, pi_s)
        adj = metrics(yk, prior_adjust(qk, pi_s, pt), float(prior_adjust(np.array([tau]), pi_s, pt)[0]))
        orc = ece15(yk, prior_adjust(qk, pi_s, yk.mean()))
        prior_rows.append(dict(model=NAME[m], pi_src=pi_s, pi_EM=pt, ECE_before=base["ece15"], ECE_EM=adj["ece15"],
                               ECE_oracle=orc, Se_RSNA=metrics(yt, qt, tau)["sensitivity"], Se_Ker=base["sensitivity"],
                               Se_Ker_EM=adj["sensitivity"], KS_pos=ks_2samp(qt[yt == 1], qk[yk == 1]).statistic,
                               KS_neg=ks_2samp(qt[yt == 0], qk[yk == 0]).statistic))
    ps = pd.DataFrame(prior_rows).set_index("model")
    ps.to_csv(out / "TRACE_CXR_SHIFT_PRIOR.csv")

    # ---------------- 3: local recalibration with n labeled Kermany images
    rng = np.random.default_rng(args.seed)
    pos, neg, prev = np.where(yk == 1)[0], np.where(yk == 0)[0], yk.mean()
    few = []
    for m in CANDS:
        q = calq(ke, m); z = logit(q); tau0 = lock["models"][m]["threshold"]
        for n in [0, 10, 20, 50, 100, 200]:
            res = {"none": [], "thr": [], "platt": []}
            for _ in range(args.reps if n else 1):
                mask = np.zeros(len(yk), bool)
                if n:
                    npos = int(round(n * prev))
                    mask[np.r_[rng.choice(pos, npos, replace=False), rng.choice(neg, n - npos, replace=False)]] = True
                ev = ~mask; ye = yk[ev]
                res["none"].append(light(ye, q[ev], tau0))
                if n:
                    res["thr"].append(light(ye, q[ev], fast_thr(q[mask], yk[mask])))
                    lr = LogisticRegression(C=1.0).fit(z[mask].reshape(-1, 1), yk[mask])
                    res["platt"].append(light(ye, lr.predict_proba(z[ev].reshape(-1, 1))[:, 1], yk[mask].mean()))
            for k, v in res.items():
                if not v: continue
                a = np.array(v)
                few.append(dict(model=NAME[m], n=n, method=k, BA=a[:, 0].mean(), BA_lo=np.percentile(a[:, 0], 2.5),
                                BA_hi=np.percentile(a[:, 0], 97.5), Se=a[:, 1].mean(), Sp=a[:, 2].mean(), ECE=a[:, 3].mean()))
    fs = pd.DataFrame(few)
    fs.to_csv(out / "TRACE_CXR_SHIFT_FEWSHOT.csv", index=False)

    # ---------------- paper macros and table rows
    f = lambda x: f"{x:.3f}"
    fw = lambda m, n, meth, c: fs[(fs.model == m) & (fs.n == n) & (fs.method == meth)].iloc[0][c]
    mac = {}
    for key, m in [("Gated", "Gated fusion"), ("Swin", "Swin-B"), ("Const", "Constant-α")]:
        r = ps.loc[m]
        mac.update({f"ShEM{key}": f(r.pi_EM), f"ShECEEM{key}": f(r.ECE_EM), f"ShECEOr{key}": f(r.ECE_oracle),
                    f"ShKSPos{key}": f(r.KS_pos), f"ShKSNeg{key}": f(r.KS_neg), f"ShSeRsna{key}": f(r.Se_RSNA),
                    f"ShSeKer{key}": f(r.Se_Ker)})
        for n, tag in [(20, "a"), (50, "b"), (100, "c")]:
            for col, c in [("BA", "BA"), ("Se", "Se"), ("ECE", "ECE"), ("BA_lo", "BALo"), ("BA_hi", "BAHi")]:
                mac[f"ShFew{key}{c}N{tag}"] = f(fw(m, n, "platt", col))
    mac["ShThrLoTen"] = f(fw("Gated fusion", 10, "thr", "BA_lo")); mac["ShThrHiTen"] = f(fw("Gated fusion", 10, "thr", "BA_hi"))
    for k, col in [("ShKSPos", "KS_pos"), ("ShKSNeg", "KS_neg"), ("ShEM", "pi_EM"), ("ShOr", "ECE_oracle")]:
        mac[k + "Min"] = f(ps[col].min()); mac[k + "Max"] = f(ps[col].max())
    (out / "paper").mkdir(exist_ok=True)
    with open(out / "paper" / "shift_numbers.tex", "w") as fh:
        fh.write("% AUTO-GENERATED by trace_cxr_shift_analysis.py\n")
        for k, v in sorted(mac.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    G, S = ps.loc["Gated fusion"], ps.loc["Swin-B"]
    rows = [["Frozen RSNA settings", f(fw("Gated fusion", 0, "none", "Se")), f(fw("Gated fusion", 0, "none", "BA")), f(G.ECE_before),
             f(fw("Swin-B", 0, "none", "Se")), f(fw("Swin-B", 0, "none", "BA")), f(S.ECE_before)],
            ["EM prior correction$^{a}$", "=", "=", f(G.ECE_EM), "=", "=", f(S.ECE_EM)],
            ["Oracle prior correction$^{b}$", "=", "=", f(G.ECE_oracle), "=", "=", f(S.ECE_oracle)]]
    for n in [20, 50, 100]:
        rows.append([f"Local recalibration, $n={n}$"] + [f(fw(m, n, "platt", c)) for m in ["Gated fusion", "Swin-B"] for c in ["Se", "BA", "ECE"]])
    with open(out / "paper" / "table5_rows.tex", "w") as fh:
        for i, r in enumerate(rows):
            if i == 3: fh.write("\\midrule\n")
            fh.write(" & ".join(r) + " \\\\\n")
    print(ps.round(3).to_string()); print("\nwrote shift CSVs, paper/shift_numbers.tex and paper/table5_rows.tex")


if __name__ == "__main__":
    main()
