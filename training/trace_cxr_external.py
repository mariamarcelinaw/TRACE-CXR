#!/usr/bin/env python3
"""
Evaluate the frozen TRACE-CXR candidates on an additional external cohort and run the
same analyses as for Kermany. Nothing is fitted on the external cohort except inside the
clearly separated local-recalibration experiment (evaluated on held-out images only).

Step 1 (GPU): frozen inference with trace_cxr.evaluate_locked_cohort
        -> <out>/TRACE_CXR_EXT_<NAME>_PREDICTIONS.csv
Step 2 (CPU): six candidates (incl. constant weight and logistic stacking from the control
        lock), paired bootstrap (gate vs mean / constant / Swin-B; patient-clustered if the
        index has 'patient_group'), EM prior correction, oracle bound, KS shift vs the RSNA
        locked test, and local recalibration with n labeled images.
        -> <out>/TRACE_CXR_EXT_<NAME>_METRICS6.csv, _PAIRED.csv, _SHIFT.csv, _FEWSHOT.csv

Usage (program directory):
    python trace_cxr_external.py --config configs/trace_cxr.yaml --index vindr_test_index.csv --name VinDr
    python trace_cxr_external.py --config ... --index ... --name VinDr --analysis-only   # reuse predictions
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from trace_cxr_controls import apply_temperature, ece15, logit, metrics          # noqa: E402
from trace_cxr_shift_analysis import em_prior, fast_thr, light, prior_adjust     # noqa: E402

CANDS = ["inception_v3", "swin_b", "mean_fusion", "gated_fusion", "const_alpha", "logit_stack"]
COMPARE = [("gated_fusion", "mean_fusion"), ("gated_fusion", "const_alpha"), ("gated_fusion", "swin_b")]


def candidate_scores(d, lock):
    pi, ps = d.inception_v3.values, d.swin_b.values
    s = lock["logit_stack"]; a = lock["const_alpha_inception"]
    raw = {"inception_v3": pi, "swin_b": ps, "mean_fusion": d.mean_fusion.values, "gated_fusion": d.gated_fusion.values,
           "const_alpha": a * pi + (1 - a) * ps,
           "logit_stack": 1 / (1 + np.exp(-(s["intercept"] + s["coef_logit_inception"] * logit(pi) + s["coef_logit_swin"] * logit(ps))))}
    return {m: apply_temperature(raw[m], lock["models"][m]["temperature"]) for m in CANDS}


def paired_bootstrap(y, qa, qb, ta, tb, groups, n_boot, seed):
    """Paired bootstrap of (a - b); resamples patients (clusters) when groups repeat."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(ug))]
    out = {"roc_auc": [], "balanced_accuracy": [], "brier": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        i = np.concatenate([members[g] for g in pick])
        yi = y[i]
        if yi.min() == yi.max():
            continue
        out["roc_auc"].append(roc_auc_score(yi, qa[i]) - roc_auc_score(yi, qb[i]))
        out["balanced_accuracy"].append(balanced_accuracy_score(yi, qa[i] >= ta) - balanced_accuracy_score(yi, qb[i] >= tb))
        out["brier"].append(brier_score_loss(yi, qa[i]) - brier_score_loss(yi, qb[i]))
    return out, len(ug) < len(y)


def analyze(out, name, pred_csv, index_csv=None, n_boot=2000, reps=500, seed=20260908):
    lock = json.loads((out / "TRACE_CXR_CONTROL_LOCK.json").read_text())
    d = pd.read_csv(pred_csv)
    if index_csv is not None and Path(index_csv).exists():
        idx = pd.read_csv(index_csv)
        if "patient_group" in idx.columns:
            d = d.merge(idx[["id", "patient_group"]], on="id", how="left")
    groups = d["patient_group"].fillna(d["id"]).astype(str).values if "patient_group" in d.columns else d["id"].astype(str).values
    y = d.label.values.astype(int)
    q = candidate_scores(d, lock)
    tau = {m: lock["models"][m]["threshold"] for m in CANDS}

    # 1. metrics of all six candidates
    rows = []
    for m in CANDS:
        r = metrics(y, q[m], tau[m]); r.update(model=m, cohort=name); rows.append(r)
    pd.DataFrame(rows).to_csv(out / f"TRACE_CXR_EXT_{name}_METRICS6.csv", index=False)

    # 2. paired differences (patient-clustered when possible)
    prow = []
    for a, b in COMPARE:
        boot, clustered = paired_bootstrap(y, q[a], q[b], tau[a], tau[b], groups, n_boot, seed)
        full = {"roc_auc": roc_auc_score(y, q[a]) - roc_auc_score(y, q[b]),
                "balanced_accuracy": balanced_accuracy_score(y, q[a] >= tau[a]) - balanced_accuracy_score(y, q[b] >= tau[b]),
                "brier": brier_score_loss(y, q[a]) - brier_score_loss(y, q[b])}
        for met, v in boot.items():
            lo, hi = np.quantile(v, [0.025, 0.975])
            prow.append(dict(cohort=name, comparison=f"{a}_minus_{b}", metric=met, delta=full[met], ci_low=lo,
                             ci_high=hi, n_boot=len(v), patient_clustered=clustered))
    pd.DataFrame(prow).to_csv(out / f"TRACE_CXR_EXT_{name}_PAIRED.csv", index=False)

    # 3. shift diagnostics (prior vs class-conditional), relative to the RSNA locked test
    cal = pd.read_csv(out / "TRACE_CXR_CALIBRATION_PREDICTIONS.csv"); te = pd.read_csv(out / "TRACE_CXR_INTERNAL_PREDICTIONS.csv")
    qc, qt, yt = candidate_scores(cal, lock), candidate_scores(te, lock), te.label.values.astype(int)
    srow = []
    for m in CANDS:
        pi_s = float(qc[m].mean()); pt = em_prior(q[m], pi_s)
        base = metrics(y, q[m], tau[m])
        srow.append(dict(model=m, cohort=name, prevalence=y.mean(), pi_EM=pt, ECE_before=base["ece15"],
                         ECE_EM=ece15(y, prior_adjust(q[m], pi_s, pt)), ECE_oracle=ece15(y, prior_adjust(q[m], pi_s, y.mean())),
                         Se_RSNA=metrics(yt, qt[m], tau[m])["sensitivity"], Se_ext=base["sensitivity"],
                         Sp_RSNA=metrics(yt, qt[m], tau[m])["specificity"], Sp_ext=base["specificity"],
                         KS_pos=ks_2samp(qt[m][yt == 1], q[m][y == 1]).statistic, KS_neg=ks_2samp(qt[m][yt == 0], q[m][y == 0]).statistic))
    pd.DataFrame(srow).to_csv(out / f"TRACE_CXR_EXT_{name}_SHIFT.csv", index=False)

    # 4. local recalibration with n labeled external images (evaluated on the rest)
    rng = np.random.default_rng(seed); pos, neg, prev = np.where(y == 1)[0], np.where(y == 0)[0], y.mean()
    frow = []
    for m in CANDS:
        z = logit(q[m])
        for n in [0, 10, 20, 50, 100, 200]:
            if n and (int(round(n * prev)) < 2 or n - int(round(n * prev)) < 2 or n >= len(y) // 2):
                continue
            res = {"none": [], "thr": [], "platt": []}
            for _ in range(reps if n else 1):
                mask = np.zeros(len(y), bool)
                if n:
                    npos = int(round(n * prev))
                    mask[np.r_[rng.choice(pos, npos, replace=False), rng.choice(neg, n - npos, replace=False)]] = True
                ev = ~mask
                res["none"].append(light(y[ev], q[m][ev], tau[m]))
                if n:
                    res["thr"].append(light(y[ev], q[m][ev], fast_thr(q[m][mask], y[mask])))
                    lr = LogisticRegression(C=1.0).fit(z[mask].reshape(-1, 1), y[mask])
                    res["platt"].append(light(y[ev], lr.predict_proba(z[ev].reshape(-1, 1))[:, 1], y[mask].mean()))
            for k, v in res.items():
                if v:
                    a = np.array(v)
                    frow.append(dict(model=m, cohort=name, n=n, method=k, BA=a[:, 0].mean(), BA_lo=np.percentile(a[:, 0], 2.5),
                                     BA_hi=np.percentile(a[:, 0], 97.5), Se=a[:, 1].mean(), Sp=a[:, 2].mean(), ECE=a[:, 3].mean()))
    pd.DataFrame(frow).to_csv(out / f"TRACE_CXR_EXT_{name}_FEWSHOT.csv", index=False)
    summary = pd.DataFrame(rows).set_index("model")[["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "ece15"]]
    print(f"\n{name}: n={len(y)}, prevalence={y.mean():.3f}\n" + summary.round(4).to_string())
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/trace_cxr.yaml")
    ap.add_argument("--index", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    name = "".join(ch for ch in a.name if ch.isalnum())
    if a.analysis_only:
        import yaml
        cfg = yaml.safe_load(open(ROOT / a.config)); out = ROOT / cfg["paths"]["output_dir"]
    else:
        import trace_cxr as T
        cfg = T.load_cfg(ROOT / a.config); T.set_seed(cfg["experiment"]["seed"]); out = T.get_out(cfg)
        lock = json.loads((out / "TRACE_CXR_DEPLOYMENT_LOCK.json").read_text())
        if lock.get("external_tuning") is not False:
            raise RuntimeError("Deployment lock does not prohibit external tuning.")
        df = pd.read_csv(a.index)[["id", "path", "label"]]
        T.evaluate_locked_cohort(df, f"{name}_external", cfg, out, f"TRACE_CXR_EXT_{name}_PREDICTIONS.csv",
                                 f"TRACE_CXR_EXT_{name}_METRICS.csv")
    analyze(out, name, out / f"TRACE_CXR_EXT_{name}_PREDICTIONS.csv", a.index)


if __name__ == "__main__":
    main()
