#!/usr/bin/env python3
"""Corrected development analysis (paper Sections IV-B and IV-C), development data only.

usage: python corrected_crossfit_check.py CORRECTED_OOF_CSV SPLIT_MANIFEST_CSV OUT_DIR

CORRECTED_OOF_CSV   private backbone OOF scores whose fold checkpoints were selected on an inner
                    validation subset of each fold's training portion (columns id, fold,
                    p_inception_v3, p_swin_b; 18,678 rows; SHA-256 fdb731e7...cf4b74a).
SPLIT_MANIFEST_CSV  private TRACE_CXR_RSNA_SPLIT_MANIFEST.csv (needs id or patientId, split, fold, label).
OUT_DIR             new directory; receives aggregate CSV files only (no image identifiers).

Steps: (1) cross-fit constant weight, logistic stacking and the 5-16-1 gate over the five
development folds (each rule is fitted on four folds and scored on the fifth); (2) held-out
ROC-AUC, AP, Brier and NLL for all six candidates; (3) gate-minus-comparator paired bootstrap
(500 image-level resamples, seed 20260909 + comparator index, as in the original run);
(4) gate-initialization stability: the same cross-fit with five gate seeds.
No RSNA locked-test or Kermany data are read. Requires numpy, pandas, scipy, scikit-learn, torch.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
import torch

EPS = 1e-7
SEEDS = [20260908, 20260909, 20260910, 20260911, 20260912]
CANDIDATES = ["inception_v3", "swin_b", "mean_fusion", "const_alpha", "gated_fusion", "logit_stack"]
OOF_SHA256 = "fdb731e7df12bf0a32bb6c393da3b75d724fe0971871868c163e11d00cf4b74a"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -p * np.log(p) - (1 - p) * np.log1p(-p)


def features(pi, ps):
    return np.column_stack((pi, ps, entropy(pi), entropy(ps), np.abs(pi - ps))).astype("float32")


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def bce(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def fit_const_alpha(y, pi, ps):
    res = minimize_scalar(lambda a: bce(y, a * pi + (1 - a) * ps), bounds=(0, 1),
                          method="bounded", options={"xatol": 1e-10})
    return min((bce(y, a * pi + (1 - a) * ps), float(a)) for a in (0.0, 1.0, float(res.x)))[1]


def train_gate(y, pi, ps, seed):
    """60 full-batch AdamW steps (lr 1e-3, weight decay 1e-4) on BCE, as for the reported gate."""
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    net = torch.nn.Sequential(torch.nn.Linear(5, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1), torch.nn.Sigmoid())
    x = torch.from_numpy(features(pi, ps))
    ti = torch.as_tensor(pi, dtype=torch.float32).reshape(-1, 1)
    ts = torch.as_tensor(ps, dtype=torch.float32).reshape(-1, 1)
    ty = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        alpha = net(x)
        loss = torch.nn.functional.binary_cross_entropy((alpha * ti + (1 - alpha) * ts).clamp(1e-6, 1 - 1e-6), ty)
        loss.backward()
        opt.step()
    return net


def gate_alpha(net, pi, ps):
    with torch.no_grad():
        return net(torch.from_numpy(features(pi, ps))).flatten().numpy().astype(float)


def crossfit(y, pi, ps, folds, gate_seed):
    n = len(y)
    pred = {"inception_v3": pi.copy(), "swin_b": ps.copy(), "mean_fusion": (pi + ps) / 2,
            "const_alpha": np.full(n, np.nan), "gated_fusion": np.full(n, np.nan), "logit_stack": np.full(n, np.nan)}
    alpha_g = np.full(n, np.nan)
    audit = []
    for f in range(5):
        tr, te = folds != f, folds == f
        a = fit_const_alpha(y[tr], pi[tr], ps[tr])
        pred["const_alpha"][te] = a * pi[te] + (1 - a) * ps[te]
        st = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(
            np.column_stack((logit(pi[tr]), logit(ps[tr]))), y[tr])
        pred["logit_stack"][te] = st.predict_proba(np.column_stack((logit(pi[te]), logit(ps[te]))))[:, 1]
        g = gate_alpha(train_gate(y[tr], pi[tr], ps[tr], gate_seed), pi[te], ps[te])
        alpha_g[te] = g
        pred["gated_fusion"][te] = g * pi[te] + (1 - g) * ps[te]
        audit.append({"heldout_fold": f, "training_n": int(tr.sum()), "heldout_n": int(te.sum()),
                      "training_positive": int(y[tr].sum()), "heldout_positive": int(y[te].sum()),
                      "const_alpha_inception": a, "gate_alpha_mean_heldout": float(g.mean()),
                      "gate_alpha_sd_heldout": float(g.std())})
    return pred, alpha_g, audit


def metrics(y, p):
    return {"roc_auc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p),
            "brier": float(np.mean((p - y) ** 2)), "nll": log_loss(y, np.clip(p, EPS, 1 - EPS))}


def paired_bootstrap(y, gate, other, B, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(B):
        ii = rng.integers(0, n, size=n)
        if np.unique(y[ii]).size < 2:
            continue
        out.append((roc_auc_score(y[ii], gate[ii]) - roc_auc_score(y[ii], other[ii]),
                    np.mean((gate[ii] - y[ii]) ** 2) - np.mean((other[ii] - y[ii]) ** 2)))
    return np.asarray(out)


def main():
    oof_path, split_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    if out.exists():
        sys.exit("OUT_DIR must not exist")
    digest = sha256(oof_path)
    if digest != OOF_SHA256:
        print(f"WARNING: corrected OOF SHA-256 {digest} differs from the archived file")
    oof = pd.read_csv(oof_path, dtype={"id": str})
    man = pd.read_csv(split_path).rename(columns={"patientId": "id"})
    man["id"] = man["id"].astype(str)
    dev = man.loc[man.split == "development", ["id", "fold", "label"]]
    d = oof.merge(dev, on="id", validate="one_to_one", suffixes=("", "_manifest"))
    assert len(d) == len(oof) == len(dev) == 18678, "expected 18,678 development rows"
    assert (d.fold.astype(int) == d.fold_manifest.astype(float).astype(int)).all(), "fold mismatch"
    d = d.sort_values("id").reset_index(drop=True)
    y = d.label.to_numpy(int)
    pi, ps = d.p_inception_v3.to_numpy(float), d.p_swin_b.to_numpy(float)
    folds = d.fold.to_numpy(int)
    out.mkdir(parents=True)

    pred, _, audit = crossfit(y, pi, ps, folds, SEEDS[0])
    pd.DataFrame([{"candidate": k, **metrics(y, v)} for k, v in pred.items()]).to_csv(
        out / "corrected_development_metrics.csv", index=False, float_format="%.6f")
    pd.DataFrame(audit).to_csv(out / "corrected_development_fold_audit.csv", index=False, float_format="%.6f")
    rows = []
    for i, c in enumerate(["swin_b", "mean_fusion", "const_alpha", "logit_stack"]):
        arr = paired_bootstrap(y, pred["gated_fusion"], pred[c], 500, 20260909 + i)
        full = metrics(y, pred["gated_fusion"]), metrics(y, pred[c])
        for j, (name, key) in enumerate((("roc_auc", "roc_auc"), ("brier", "brier"))):
            rows.append({"comparison": f"gated_fusion_minus_{c}", "metric": name,
                         "delta": full[0][key] - full[1][key],
                         "ci_low": np.quantile(arr[:, j], .025), "ci_high": np.quantile(arr[:, j], .975),
                         "valid_resamples": len(arr), "resampling_unit": "image (RSNA patientId)"})
    pd.DataFrame(rows).to_csv(out / "corrected_development_paired_bootstrap.csv", index=False, float_format="%.6f")

    stab = []
    for s in SEEDS:
        p_s, a_s, _ = crossfit(y, pi, ps, folds, s)
        m = metrics(y, p_s["gated_fusion"])
        stab.append({"gate_seed": s, "heldout_roc_auc": m["roc_auc"], "heldout_brier": m["brier"],
                     "alpha_mean": a_s.mean(), "alpha_sd": a_s.std(), "alpha_min": a_s.min(), "alpha_max": a_s.max(),
                     "delta_auc_vs_const_alpha": m["roc_auc"] - roc_auc_score(y, p_s["const_alpha"]),
                     "delta_auc_vs_swin_b": m["roc_auc"] - roc_auc_score(y, ps)})
    pd.DataFrame(stab).to_csv(out / "gate_initialization_stability.csv", index=False, float_format="%.6f")
    print("written:", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
