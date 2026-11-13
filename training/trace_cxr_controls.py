#!/usr/bin/env python3
"""
TRACE-CXR fusion controls (CPU only, no torch).

Adds two score-level controls that are fitted on exactly the same development
OOF scores as the reliability gate, then calibrated/thresholded on exactly the
same RSNA calibration split as the other candidates:

  const_alpha : p = a*p_I + (1-a)*p_S, a single scalar fitted by OOF BCE
                (isolates "learned weighting" from "per-image adaptivity")
  logit_stack : logistic regression on [logit p_I, logit p_S] + intercept
                (classic stacking baseline, Wolpert 1992)

Kermany is never used for fitting anything.

Inputs  (in --out): TRACE_CXR_OOF_FUSION.csv, TRACE_CXR_CALIBRATION_PREDICTIONS.csv,
                    TRACE_CXR_INTERNAL_PREDICTIONS.csv, TRACE_CXR_KERMANY_PREDICTIONS.csv,
                    TRACE_CXR_DEPLOYMENT_LOCK.json, TRACE_CXR_RSNA_SPLIT_MANIFEST.csv
                    (+ optional XAI summaries for paper macros)
Outputs (in --out): TRACE_CXR_CONTROL_LOCK.json
                    TRACE_CXR_CONTROL_METRICS.csv
                    TRACE_CXR_CONTROL_PAIRED_STATISTICS.csv
                    TRACE_CXR_BRANCH_CONSISTENCY.csv
                    paper/paper_numbers.tex, paper/table2_rows.tex,
                    paper/table3_rows.tex, paper/table4_rows.tex

Usage:  python trace_cxr_controls.py --out outputs [--n-boot 2000] [--seed 20260908]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score)

# ----------------------------------------------------------------------------
# Functions copied verbatim in behaviour from trace_cxr.py so that the four
# original candidates reproduce TRACE_CXR_DEPLOYMENT_LOCK.json exactly.
# ----------------------------------------------------------------------------

def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_temperature(p, y):
    z = logit(p)
    y = np.asarray(y, float)

    def objective(log_t):
        q = 1 / (1 + np.exp(-z / np.exp(log_t)))
        q = np.clip(q, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    return float(np.exp(minimize_scalar(objective, bounds=(-3, 3), method="bounded").x))


def apply_temperature(p, t):
    return 1 / (1 + np.exp(-logit(p) / max(float(t), 1e-6)))


def choose_threshold(p, y):
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    best_t, best = 0.5, -1.0
    for t in np.unique(np.r_[0.0, p, 1.0]):
        s = balanced_accuracy_score(y, (p >= t).astype(int))
        if s > best:
            best, best_t = float(s), float(t)
    return best_t, best


def ece15(y, p, bins=15):
    y = np.asarray(y)
    p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    v = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            v += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(v)


def metrics(y, q, tau):
    y = np.asarray(y, int)
    q = np.asarray(q, float)
    pred = (q >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y), "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, q)),
        "auprc": float(average_precision_score(y, q)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, q)),
        "ece15": ece15(y, q),
        "mean_calibrated_probability": float(q.mean()),
        "calibration_in_the_large": float(q.mean() - y.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(tau),
    }


# ----------------------------------------------------------------------------

CANDIDATES = ["inception_v3", "swin_b", "mean_fusion", "gated_fusion", "const_alpha", "logit_stack"]
LABEL = {"inception_v3": "InceptionV3", "swin_b": "Swin-B", "mean_fusion": "Mean fusion",
         "gated_fusion": "Gated fusion", "const_alpha": r"Constant-$\alpha$", "logit_stack": "Logit stacking"}
MACRO = {"inception_v3": "Inc", "swin_b": "Swin", "mean_fusion": "Mean", "gated_fusion": "Gated",
         "const_alpha": "Const", "logit_stack": "Stack"}
COHORTS = [("RSNA_locked_test", "Rsna", "TRACE_CXR_INTERNAL_PREDICTIONS.csv"),
           ("Kermany_external", "Ker", "TRACE_CXR_KERMANY_PREDICTIONS.csv")]
COMPARISONS = [("gated_fusion", "mean_fusion"), ("gated_fusion", "const_alpha"),
               ("gated_fusion", "logit_stack")]
METRIC_MACRO = {"roc_auc": "AUC", "auprc": "AP", "balanced_accuracy": "BA", "sensitivity": "Se",
                "specificity": "Sp", "f1": "F", "brier": "Brier", "ece15": "ECE",
                "calibration_in_the_large": "CITL", "mean_calibrated_probability": "MeanQ"}


class ControlModels:
    """Fitted on development OOF scores only."""

    def __init__(self, oof: pd.DataFrame):
        y = oof["label"].to_numpy()
        pi = oof["inception_v3"].to_numpy()
        ps = oof["swin_b"].to_numpy()

        def bce(a):
            p = np.clip(a * pi + (1 - a) * ps, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        self.alpha = float(minimize_scalar(bce, bounds=(0, 1), method="bounded").x)
        self.stack = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(
            np.c_[logit(pi), logit(ps)], y)

    def raw_scores(self, df: pd.DataFrame) -> dict:
        pi, ps = df["inception_v3"].to_numpy(), df["swin_b"].to_numpy()
        return {
            "inception_v3": pi, "swin_b": ps,
            "mean_fusion": df["mean_fusion"].to_numpy(),
            "gated_fusion": df["gated_fusion"].to_numpy(),
            "const_alpha": self.alpha * pi + (1 - self.alpha) * ps,
            "logit_stack": self.stack.predict_proba(np.c_[logit(pi), logit(ps)])[:, 1],
        }


def fmt(x, nd=4, sign=False):
    if abs(x) < 0.5 * 10 ** (-nd):
        x, sign = 0.0, False  # avoid "-0.0000"/"+0.0000"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs", help="TRACE-CXR output directory")
    ap.add_argument("--paper-dir", default=None, help="where to write .tex files (default: <out>/paper)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    out = Path(args.out)
    paper = Path(args.paper_dir) if args.paper_dir else out / "paper"
    paper.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(out / "TRACE_CXR_OOF_FUSION.csv")
    cal = pd.read_csv(out / "TRACE_CXR_CALIBRATION_PREDICTIONS.csv")
    lock = json.loads((out / "TRACE_CXR_DEPLOYMENT_LOCK.json").read_text())
    manifest = pd.read_csv(out / "TRACE_CXR_RSNA_SPLIT_MANIFEST.csv")

    # ---- integrity: no overlap, OOF == development --------------------------
    dev_ids = set(manifest.loc[manifest.split == "development", "id"])
    assert set(oof.id) == dev_ids, "OOF ids do not match development ids"
    cohorts = {name: pd.read_csv(out / f) for name, _, f in COHORTS}
    for a, b in [(set(oof.id), set(cal.id)), (set(oof.id), set(cohorts["RSNA_locked_test"].id)),
                 (set(cal.id), set(cohorts["RSNA_locked_test"].id))]:
        assert not (a & b), "RSNA partitions overlap"

    # ---- fit controls on OOF, T and tau on RSNA calibration -----------------
    ctrl = ControlModels(oof)
    cal_scores = ctrl.raw_scores(cal)
    params = {}
    for m in CANDIDATES:
        t = fit_temperature(cal_scores[m], cal["label"])
        tau, cal_ba = choose_threshold(apply_temperature(cal_scores[m], t), cal["label"])
        params[m] = {"temperature": t, "threshold": tau, "calibration_balanced_accuracy": cal_ba}
        if m in lock["models"]:
            dt = abs(t - lock["models"][m]["temperature"])
            dtau = abs(tau - lock["models"][m]["threshold"])
            status = "OK" if (dt < 1e-5 and dtau < 1e-6) else "MISMATCH"
            print(f"[lock check] {m:13s} T diff={dt:.2e} tau diff={dtau:.2e} -> {status}")
            if status != "OK":
                raise SystemExit("Refitted T/tau differ from deployment lock; inputs are inconsistent.")

    control_lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fitted_on": {"fusion_weights": "RSNA development OOF only",
                      "temperature_threshold": "RSNA calibration only"},
        "external_tuning": False,
        "const_alpha_inception": ctrl.alpha,
        "logit_stack": {"coef_logit_inception": float(ctrl.stack.coef_[0, 0]),
                        "coef_logit_swin": float(ctrl.stack.coef_[0, 1]),
                        "intercept": float(ctrl.stack.intercept_[0])},
        "models": params,
        "bootstrap": {"n_boot": args.n_boot, "seed": args.seed, "unit": "technical image",
                      "interval": "percentile 95%", "delta": "full-sample point estimate"},
        "input_sha256": {f: sha256(out / f) for f in
                         ["TRACE_CXR_OOF_FUSION.csv", "TRACE_CXR_CALIBRATION_PREDICTIONS.csv",
                          "TRACE_CXR_INTERNAL_PREDICTIONS.csv", "TRACE_CXR_KERMANY_PREDICTIONS.csv",
                          "TRACE_CXR_DEPLOYMENT_LOCK.json"]},
    }
    (out / "TRACE_CXR_CONTROL_LOCK.json").write_text(json.dumps(control_lock, indent=2))

    # ---- evaluate -----------------------------------------------------------
    metric_rows, stat_rows, macros = [], [], {}
    for cohort, ck, _ in COHORTS:
        df = cohorts[cohort]
        y = df["label"].to_numpy().astype(int)
        raw = ctrl.raw_scores(df)
        q = {m: apply_temperature(raw[m], params[m]["temperature"]) for m in CANDIDATES}
        pred = {m: (q[m] >= params[m]["threshold"]).astype(int) for m in CANDIDATES}
        full = {}
        for m in CANDIDATES:
            r = metrics(y, q[m], params[m]["threshold"])
            r["raw_ece15"] = ece15(y, raw[m])
            r.update(model=m, cohort=cohort, temperature=params[m]["temperature"])
            metric_rows.append(r)
            full[m] = r
            for k, mk in METRIC_MACRO.items():
                macros[f"{ck}{MACRO[m]}{mk}"] = fmt(r[k], 4 if k not in ("calibration_in_the_large", "mean_calibrated_probability") else 3)
            macros[f"{ck}{MACRO[m]}RawECE"] = fmt(r["raw_ece15"])
            for k in ("tn", "fp", "fn", "tp"):
                macros[f"{ck}{MACRO[m]}{k.upper()}"] = f"{r[k]:,}"
        macros[f"{ck}Prev"] = fmt(float(y.mean()), 3)
        macros[f"{ck}N"] = f"{len(y):,}"
        macros[f"{ck}Pos"] = f"{int(y.sum()):,}"
        if "gate_alpha_inception" in df:
            macros[f"{ck}AlphaMin"] = fmt(df.gate_alpha_inception.min(), 3)
            macros[f"{ck}AlphaMax"] = fmt(df.gate_alpha_inception.max(), 3)
            macros[f"{ck}AlphaSD"] = fmt(df.gate_alpha_inception.std(), 3)

        # paired image-level bootstrap: identical index matrix for every metric and comparison
        rng = np.random.default_rng(args.seed)
        idx = [rng.integers(0, len(y), len(y)) for _ in range(args.n_boot)]
        needed = sorted({m for pair in COMPARISONS for m in pair})
        boot = {m: {"roc_auc": [], "balanced_accuracy": [], "brier": []} for m in needed}
        for i in idx:
            yi = y[i]
            for m in needed:
                boot[m]["roc_auc"].append(roc_auc_score(yi, q[m][i]))
                boot[m]["balanced_accuracy"].append(balanced_accuracy_score(yi, pred[m][i]))
                boot[m]["brier"].append(brier_score_loss(yi, q[m][i]))
        for a, b in COMPARISONS:
            for met in ("roc_auc", "balanced_accuracy", "brier"):
                d = np.asarray(boot[a][met]) - np.asarray(boot[b][met])
                delta = full[a][met] - full[b][met]
                lo, hi = np.quantile(d, [0.025, 0.975])
                stat_rows.append(dict(cohort=cohort, comparison=f"{a}_minus_{b}", metric=met,
                                      delta=delta, ci_low=lo, ci_high=hi, n_boot=args.n_boot,
                                      seed=args.seed))
                key = f"D{ck}{MACRO[a]}{MACRO[b]}{METRIC_MACRO[met]}"
                macros[key] = fmt(delta, 4, sign=True)
                macros["CI" + key[1:]] = f"[{fmt(lo, 4, True)}, {fmt(hi, 4, True)}]"
            ca, cb = pred[a] == y, pred[b] == y
            n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
            stat_rows.append(dict(cohort=cohort, comparison=f"{a}_minus_{b}", metric="mcnemar_exact",
                                  a_correct_b_wrong=n01, a_wrong_b_correct=n10, p_value=p))

    pd.DataFrame(metric_rows).to_csv(out / "TRACE_CXR_CONTROL_METRICS.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(out / "TRACE_CXR_CONTROL_PAIRED_STATISTICS.csv", index=False)

    # ---- branch consistency: OOF fold models vs final model ------------------
    oofm = oof.merge(manifest[["id", "fold"]], on="id")
    cons = []
    for m in ("inception_v3", "swin_b"):
        fold_auc = [roc_auc_score(g.label, g[m]) for _, g in oofm.groupby("fold")]
        cal_auc = roc_auc_score(cal.label, cal[m])
        test_auc = roc_auc_score(cohorts["RSNA_locked_test"].label, cohorts["RSNA_locked_test"][m])
        flag = cal_auc < min(fold_auc) - 0.02
        cons.append(dict(model=m, oof_fold_auc_min=min(fold_auc), oof_fold_auc_max=max(fold_auc),
                         oof_pooled_auc=roc_auc_score(oof.label, oof[m]),
                         final_calibration_auc=cal_auc, final_test_auc=test_auc,
                         final_inconsistent_with_oof=bool(flag)))
        mk = MACRO[m]
        macros[f"Oof{mk}Min"] = fmt(min(fold_auc))
        macros[f"Oof{mk}Max"] = fmt(max(fold_auc))
        macros[f"Cal{mk}AUC"] = fmt(cal_auc)
    pd.DataFrame(cons).to_csv(out / "TRACE_CXR_BRANCH_CONSISTENCY.csv", index=False)

    # ---- other paper macros --------------------------------------------------
    macros["ConstAlpha"] = fmt(ctrl.alpha, 3)
    macros["StackCoefI"] = fmt(float(ctrl.stack.coef_[0, 0]), 3)
    macros["StackCoefS"] = fmt(float(ctrl.stack.coef_[0, 1]), 3)
    macros["NBoot"] = f"{args.n_boot:,}"
    for m in CANDIDATES:
        macros[f"T{MACRO[m]}"] = fmt(params[m]["temperature"])
        macros[f"Tau{MACRO[m]}"] = fmt(params[m]["threshold"])
    split = manifest.groupby("split")["label"].agg(["count", "sum"])
    for s, key in [("development", "Dev"), ("calibration", "Cal"), ("locked_test", "Test")]:
        macros[f"Split{key}N"] = f"{int(split.loc[s, 'count']):,}"
        macros[f"Split{key}Pos"] = f"{int(split.loc[s, 'sum']):,}"
        macros[f"Split{key}Neg"] = f"{int(split.loc[s, 'count'] - split.loc[s, 'sum']):,}"
    macros["SplitAllN"] = f"{len(manifest):,}"
    kd = cohorts["Kermany_external"]
    macros["KerNeg"] = f"{int((kd.label == 0).sum()):,}"

    vf = out / "TRACE_CXR_TRANSFORM_INPUT_VERIFICATION.json"
    if vf.exists():
        v = json.loads(vf.read_text())
        macros["FixRecordedHoldoutAUC"] = fmt(v["recorded_holdout_best_val_auc"])
        macros["FixHoldoutAUCFalse"] = fmt(v["transform_input_false"]["holdout_auc"])
        macros["FixHoldoutAUCTrue"] = fmt(v["transform_input_true"]["holdout_auc"])
    else:
        macros["FixRecordedHoldoutAUC"] = "0.8819"
        macros["FixHoldoutAUCFalse"] = r"\VERIFY{n/a}"
        macros["FixHoldoutAUCTrue"] = r"\VERIFY{n/a}"

    fs = out / "TRACE_CXR_XAI_FAITHFULNESS_SUMMARY.csv"
    if fs.exists():
        f = pd.read_csv(fs)
        f = f[f.metric == "aopc_gain_over_random"]
        for _, r in f.iterrows():
            k = f"Aopc{'Rsna' if r.cohort == 'RSNA' else 'Ker'}{MACRO[r.branch]}"
            macros[k] = fmt(r["mean"])
            macros[k + "CI"] = f"{fmt(r.ci_low)} to {fmt(r.ci_high)}"
            macros["AopcN"] = str(int(r["n"]))
    ls = out / "TRACE_CXR_XAI_LOCALIZATION_SUMMARY.csv"
    if ls.exists():
        l = pd.read_csv(ls)
        for _, r in l[l.metric.isin(["pointing_hit", "energy_in_box"])].iterrows():
            k = f"Loc{MACRO[r.branch]}{'Hit' if r.metric == 'pointing_hit' else 'Energy'}"
            macros[k] = fmt(r["mean"], 3)
            macros["LocN"] = str(int(r["n"]))

    with open(paper / "paper_numbers.tex", "w") as fh:
        fh.write("% AUTO-GENERATED by trace_cxr_controls.py -- do not edit by hand\n")
        fh.write(f"% generated {control_lock['created_utc']}\n")
        for k in sorted(macros):
            fh.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")

    # ---- LaTeX table rows ----------------------------------------------------
    mdf = pd.DataFrame(metric_rows)
    with open(paper / "table3_rows.tex", "w") as fh:
        for cohort, ck, _ in COHORTS:
            name = "RSNA" if ck == "Rsna" else "Kermany"
            sub = mdf[mdf.cohort == cohort].set_index("model")
            best_auc = sub.roc_auc.max()
            for j, m in enumerate(CANDIDATES):
                r = sub.loc[m]
                cells = [fmt(r[k]) for k in ("roc_auc", "auprc", "balanced_accuracy", "sensitivity",
                                              "specificity", "f1", "brier", "ece15")]
                if r.roc_auc == best_auc:
                    cells[0] = r"\textbf{" + cells[0] + "}"
                lead = rf"\multirow{{6}}{{*}}{{{name}}}" if j == 0 else ""
                sep = r"\cmidrule(l){2-10}" if j == 3 else ""
                fh.write(f"{sep}{lead} & {LABEL[m]} & " + " & ".join(cells) + r" \\" + "\n")
            if ck == "Rsna":
                fh.write(r"\midrule" + "\n")

    with open(paper / "table2_rows.tex", "w") as fh:
        for m in CANDIDATES:
            fh.write(f"{LABEL[m]} & {fmt(params[m]['temperature'])} & {fmt(params[m]['threshold'])} "
                     f"& {fmt(params[m]['calibration_balanced_accuracy'])} " + r"\\" + "\n")

    sdf = pd.DataFrame(stat_rows)
    mlabel = {"roc_auc": "AUC", "balanced_accuracy": "BA", "brier": "Brier"}
    blabel = {"mean_fusion": "Mean", "const_alpha": r"Const-$\alpha$", "logit_stack": "Stack"}
    with open(paper / "table4_rows.tex", "w") as fh:
        for k, (a, b) in enumerate(COMPARISONS):
            for met in ("roc_auc", "balanced_accuracy", "brier"):
                cells = []
                for cohort, _, _ in COHORTS:
                    r = sdf[(sdf.cohort == cohort) & (sdf.comparison == f"{a}_minus_{b}") &
                            (sdf.metric == met)].iloc[0]
                    cells.append(f"{fmt(r.delta, 4, True)} [{fmt(r.ci_low, 4, True)}, {fmt(r.ci_high, 4, True)}]")
                lead = rf"\multirow{{3}}{{*}}{{G$-${blabel[b]}}}" if met == "roc_auc" else ""
                fh.write(f"{lead} & {mlabel[met]} & " + " & ".join(cells) + r" \\" + "\n")
            if k < len(COMPARISONS) - 1:
                fh.write(r"\midrule" + "\n")

    # ---- console summary ------------------------------------------------------
    print(f"\nconstant alpha (OOF BCE) = {ctrl.alpha:.4f}")
    print(mdf[["cohort", "model", "roc_auc", "balanced_accuracy", "sensitivity", "specificity",
               "brier", "ece15", "calibration_in_the_large"]].round(4).to_string(index=False))
    print()
    print(sdf[sdf.metric != "mcnemar_exact"][["cohort", "comparison", "metric", "delta", "ci_low",
                                               "ci_high"]].round(5).to_string(index=False))
    print()
    print(pd.DataFrame(cons).round(4).to_string(index=False))
    print(f"\nwrote tables/macros to {paper}")


if __name__ == "__main__":
    main()
