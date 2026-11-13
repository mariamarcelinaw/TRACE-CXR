#!/usr/bin/env python3
"""
Post-hoc sensitivity analysis: is the near-constant gate an artefact of the
locked training length?

The locked gate is trained full-batch for 60 epochs, i.e. only 60 AdamW steps.
This script retrains the SAME ReliabilityGate (trace_cxr.py) on the SAME OOF
scores for longer, then applies the SAME protocol as every other candidate
(temperature + BA threshold on RSNA calibration, frozen for RSNA test and
Kermany). Kermany is never used for fitting. CPU only; no images needed.

The locked gate and all main results are NOT modified. Results are reported
as a labelled post-hoc sensitivity analysis.

Usage (program directory):  python gate_sensitivity.py
Outputs: <out>/TRACE_CXR_GATE_SENSITIVITY.csv, <out>/paper/gate_sensitivity.tex
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import trace_cxr as T  # noqa: E402
from trace_cxr_controls import (apply_temperature, choose_threshold,  # noqa: E402
                                fit_temperature, metrics)

SEEDS = [0, 1, 2]
SETTINGS = [(1e-3, 60), (1e-3, 20000), (1e-2, 20000)]   # (lr, full-batch steps); first = locked


def train(pi, ps, y, lr, steps, seed, cfg):
    torch.manual_seed(seed)
    gate = T.ReliabilityGate(cfg["fusion"]["hidden_dim"])
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=cfg["fusion"]["weight_decay"])
    x = torch.tensor(T.gate_features(pi, ps), dtype=torch.float32)
    tpi, tps = torch.tensor(pi, dtype=torch.float32), torch.tensor(ps, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    loss_fn = nn.BCELoss()
    for _ in range(steps):
        opt.zero_grad()
        a = gate(x)
        loss = loss_fn(torch.clamp(a * tpi + (1 - a) * tps, 1e-6, 1 - 1e-6), ty)
        loss.backward()
        opt.step()
    return gate, float(loss.detach())


def alpha(gate, df):
    with torch.no_grad():
        x = torch.tensor(T.gate_features(df.inception_v3.values, df.swin_b.values), dtype=torch.float32)
        return gate(x).numpy().astype(float)


def main():
    cfg = T.load_cfg(ROOT / "configs/trace_cxr.yaml")
    out = T.get_out(cfg)
    oof = pd.read_csv(out / "TRACE_CXR_OOF_FUSION.csv")
    cal = pd.read_csv(out / "TRACE_CXR_CALIBRATION_PREDICTIONS.csv")
    cohorts = {"RSNA": pd.read_csv(out / "TRACE_CXR_INTERNAL_PREDICTIONS.csv"),
               "Kermany": pd.read_csv(out / "TRACE_CXR_KERMANY_PREDICTIONS.csv")}
    pi, ps, y = oof.inception_v3.values, oof.swin_b.values, oof.label.values.astype(float)

    rows = []
    for lr, steps in SETTINGS:
        for seed in SEEDS:
            gate, oof_bce = train(pi, ps, y, lr, steps, seed, cfg)
            mix = lambda d: (lambda a: a * d.inception_v3.values + (1 - a) * d.swin_b.values)(alpha(gate, d))
            t = fit_temperature(mix(cal), cal.label)
            tau, _ = choose_threshold(apply_temperature(mix(cal), t), cal.label)
            for name, d in cohorts.items():
                m = metrics(d.label, apply_temperature(mix(d), t), tau)
                a = alpha(gate, d)
                rows.append(dict(lr=lr, steps=steps, seed=seed, cohort=name, oof_bce=oof_bce,
                                 roc_auc=m["roc_auc"], balanced_accuracy=m["balanced_accuracy"],
                                 brier=m["brier"], ece15=m["ece15"], alpha_min=a.min(),
                                 alpha_max=a.max(), alpha_sd=a.std(), temperature=t, threshold=tau))
                print(f"lr={lr:g} steps={steps:5d} seed={seed} {name:7s} OOF-BCE={oof_bce:.5f} "
                      f"AUC={m['roc_auc']:.4f} BA={m['balanced_accuracy']:.4f} ECE={m['ece15']:.4f} "
                      f"alpha {a.min():.3f}-{a.max():.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "TRACE_CXR_GATE_SENSITIVITY.csv", index=False)

    # constant-alpha reference from the control lock
    ctrl = pd.read_csv(out / "TRACE_CXR_CONTROL_METRICS.csv")
    const = lambda c: ctrl[(ctrl.model == "const_alpha") & (ctrl.cohort.str.startswith(c))].iloc[0]
    a0 = pd.read_json(out / "TRACE_CXR_CONTROL_LOCK.json", typ="series")["const_alpha_inception"]
    const_bce = -np.mean(y * np.log(np.clip(a0 * pi + (1 - a0) * ps, 1e-6, 1 - 1e-6))
                         + (1 - y) * np.log(1 - np.clip(a0 * pi + (1 - a0) * ps, 1e-6, 1 - 1e-6)))

    long = df[df.steps == 20000]
    f = lambda x: f"{x:.4f}"
    mac = {
        "SensSteps": "20{,}000", "SensSeeds": str(len(SEEDS)),
        "SensOofMin": f(long.oof_bce.min()), "SensOofMax": f(long.oof_bce.max()),
        "SensConstOof": f(const_bce),
        "SensRsnaAUCMin": f(long[long.cohort == "RSNA"].roc_auc.min()),
        "SensRsnaAUCMax": f(long[long.cohort == "RSNA"].roc_auc.max()),
        "SensKerAUCMin": f(long[long.cohort == "Kermany"].roc_auc.min()),
        "SensKerAUCMax": f(long[long.cohort == "Kermany"].roc_auc.max()),
        "SensKerECEMin": f(long[long.cohort == "Kermany"].ece15.min()),
        "SensKerECEMax": f(long[long.cohort == "Kermany"].ece15.max()),
        "SensAlphaSDMin": f"{long[long.cohort == 'RSNA'].alpha_sd.min():.2f}",
        "SensAlphaSDMax": f"{long[long.cohort == 'RSNA'].alpha_sd.max():.2f}",
        "SensConstRsnaAUC": f(const("RSNA").roc_auc), "SensConstKerAUC": f(const("Kermany").roc_auc),
    }
    (out / "paper").mkdir(exist_ok=True)
    with open(out / "paper" / "gate_sensitivity.tex", "w") as fh:
        fh.write("% AUTO-GENERATED by gate_sensitivity.py (post-hoc sensitivity analysis)\n")
        for k, v in mac.items():
            fh.write(f"\\renewcommand{{\\{k}}}{{{v}}}\n")
    print("\nwrote", out / "TRACE_CXR_GATE_SENSITIVITY.csv", "and paper/gate_sensitivity.tex")


if __name__ == "__main__":
    main()
