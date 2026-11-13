#!/usr/bin/env python3
"""
Kermany preprocessing-consistency check.

Finding: trace_cxr.read_cxr applies percentile contrast stretching (0.5-99.5 %, then 0-255)
to DICOM images (all RSNA training/calibration/test images) but NOT to raster images
(Kermany JPEGs, which are only converted to RGB). This script re-runs the frozen Kermany
evaluation with the SAME normalization applied to the Kermany images and compares results.
Models, temperatures, thresholds and the gate stay frozen; no Kermany label is used to fit anything
except inside the separated local-recalibration experiment.

Steps
  0. write the protocol (rationale + reporting rule) BEFORE inference;
  1. pixel check: how much the normalization changes Kermany images;
  2. (optional, default on) re-infer Kermany WITHOUT normalization and verify it reproduces the
     archived predictions, so any difference is due to the normalization alone;
  3. frozen inference WITH normalization -> TRACE_CXR_KERMANY_NORM_PREDICTIONS.csv;
  4. full Kermany analysis on the new predictions (six candidates, patient-clustered paired
     bootstrap, EM prior correction, KS shift, local recalibration);
  5. before/after comparison -> TRACE_CXR_KERMANY_NORM_COMPARISON.csv.
The archived Kermany outputs are never overwritten.

Usage (program directory, GPU):   python kermany_consistency_check.py
Needs in the same folder: trace_cxr.py (with the transform_input fix), trace_cxr_kermany.py,
trace_cxr_controls.py, trace_cxr_shift_analysis.py, trace_cxr_external.py
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def kermany_patient(s):
    m = re.match(r"(person\d+)", s) or re.match(r"((?:NORMAL2-)?IM-\d+)", s)
    return m.group(1) if m else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/trace_cxr.yaml")
    ap.add_argument("--no-verify", action="store_true", help="skip re-inference without normalization")
    a = ap.parse_args()

    import trace_cxr as T
    from trace_cxr_kermany import kermany_index
    from trace_cxr_external import analyze

    cfg = T.load_cfg(ROOT / a.config); T.set_seed(cfg["experiment"]["seed"]); out = T.get_out(cfg)
    lock = json.loads((out / "TRACE_CXR_DEPLOYMENT_LOCK.json").read_text())
    if lock.get("external_tuning") is not False:
        raise RuntimeError("Deployment lock does not prohibit external tuning.")
    df = kermany_index(cfg["paths"]["kermany_root"])
    df["patient_group"] = df["id"].astype(str).map(kermany_patient)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ---- 0. protocol before inference
    protocol = {
        "created_utc": stamp,
        "issue": "read_cxr normalizes DICOM intensities (percentile 0.5-99.5 stretch) but not raster images; "
                 "all RSNA images are DICOM, all Kermany images are JPEG.",
        "change": "apply the identical normalization (trace_cxr._to_uint8) to Kermany images at inference",
        "unchanged": ["backbone weights", "gate", "temperatures", "thresholds", "RSNA results"],
        "kermany_labels_used_for_fitting": False,
        "decision_rule": "Adopted on the principle of consistent preprocessing, decided before seeing the new "
                         "results; both versions are reported whatever the direction of the change.",
        "n_images": int(len(df)), "n_patients": int(df["patient_group"].nunique()),
    }
    (out / f"TRACE_CXR_KERMANY_NORM_PROTOCOL_{stamp}.json").write_text(json.dumps(protocol, indent=2))
    print(json.dumps(protocol, indent=2))

    # ---- 1. pixel check
    orig_read = T.read_cxr

    def read_norm(path):
        path = str(path)
        if path.lower().endswith(".dcm"):
            return orig_read(path)
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        return Image.fromarray(T._to_uint8(arr, None)).convert("RGB")

    sample = df.sample(min(50, len(df)), random_state=0)["path"]
    diffs, frac = [], []
    for p in sample:
        x0 = np.asarray(orig_read(p).convert("L"), dtype=np.float32); x1 = np.asarray(read_norm(p).convert("L"), dtype=np.float32)
        diffs.append(np.abs(x1 - x0).mean()); frac.append((np.abs(x1 - x0) > 2).mean())
    print(f"\n[pixel check] mean |normalized - raw| = {np.mean(diffs):.2f} gray levels; "
          f"pixels changed by >2 levels: {np.mean(frac):.1%}")

    base = df[["id", "path", "label"]]
    # ---- 2. verify that current code reproduces the archived (unnormalized) predictions
    if not a.no_verify:
        T.read_cxr = orig_read
        T.evaluate_locked_cohort(base, "Kermany_external_recheck", cfg, out,
                                 "TRACE_CXR_KERMANY_RECHECK_PREDICTIONS.csv", "TRACE_CXR_KERMANY_RECHECK_METRICS.csv")
        old = pd.read_csv(out / "TRACE_CXR_KERMANY_PREDICTIONS.csv").set_index("id")
        new = pd.read_csv(out / "TRACE_CXR_KERMANY_RECHECK_PREDICTIONS.csv").set_index("id").loc[old.index]
        dmax = float(max((old[c] - new[c]).abs().max() for c in ["inception_v3", "swin_b", "gated_fusion"]))
        print(f"[verify] max |recheck - archived| = {dmax:.2e} ({'OK' if dmax < 1e-4 else 'CHECK: differs'})")
        protocol["recheck_max_abs_diff"] = dmax

    # ---- 3. frozen inference with RSNA-consistent normalization
    T.read_cxr = read_norm
    T.evaluate_locked_cohort(base, "Kermany_external_norm", cfg, out,
                             "TRACE_CXR_KERMANY_NORM_PREDICTIONS.csv", "TRACE_CXR_KERMANY_NORM_METRICS.csv")
    T.read_cxr = orig_read
    idx = out / "TRACE_CXR_KERMANY_NORM_INDEX.csv"; df[["id", "label", "patient_group"]].to_csv(idx, index=False)
    # archived predictions analysed the same way (patient-clustered), for a like-for-like comparison
    analyze(out, "KermanyRaw", out / "TRACE_CXR_KERMANY_PREDICTIONS.csv", idx)
    analyze(out, "KermanyNorm", out / "TRACE_CXR_KERMANY_NORM_PREDICTIONS.csv", idx)

    # ---- 5. comparison
    comp = compare(out)
    comp.to_csv(out / "TRACE_CXR_KERMANY_NORM_COMPARISON.csv", index=False)
    protocol["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (out / f"TRACE_CXR_KERMANY_NORM_PROTOCOL_{stamp}.json").write_text(json.dumps(protocol, indent=2))


def compare(out):
    rows = []
    for tag in ["KermanyRaw", "KermanyNorm"]:
        m = pd.read_csv(out / f"TRACE_CXR_EXT_{tag}_METRICS6.csv").set_index("model")
        s = pd.read_csv(out / f"TRACE_CXR_EXT_{tag}_SHIFT.csv").set_index("model")
        f = pd.read_csv(out / f"TRACE_CXR_EXT_{tag}_FEWSHOT.csv")
        for k in m.index:
            fr = f[(f.model == k) & (f.n == 20) & (f.method == "platt")]
            rows.append(dict(version="raw (archived)" if tag == "KermanyRaw" else "RSNA-consistent normalization",
                             model=k, roc_auc=m.loc[k, "roc_auc"], balanced_accuracy=m.loc[k, "balanced_accuracy"],
                             sensitivity=m.loc[k, "sensitivity"], specificity=m.loc[k, "specificity"], ece15=m.loc[k, "ece15"],
                             mean_pred=m.loc[k, "mean_calibrated_probability"], pi_EM=s.loc[k, "pi_EM"],
                             ECE_oracle=s.loc[k, "ECE_oracle"], KS_pos=s.loc[k, "KS_pos"], KS_neg=s.loc[k, "KS_neg"],
                             Se_n20=fr.Se.iloc[0] if len(fr) else np.nan, ECE_n20=fr.ECE.iloc[0] if len(fr) else np.nan))
    c = pd.DataFrame(rows)
    show = c[c.model.isin(["swin_b", "gated_fusion", "const_alpha", "inception_v3"])]
    pd.set_option("display.width", 250)
    print("\n================ BEFORE vs AFTER (frozen RSNA settings) ================")
    print(show.round(3).to_string(index=False))
    g = c.set_index(["version", "model"])
    for k in ["gated_fusion", "swin_b"]:
        b, n = g.loc[("raw (archived)", k)], g.loc[("RSNA-consistent normalization", k)]
        print(f"\n{k}: sensitivity {b.sensitivity:.3f} -> {n.sensitivity:.3f} | AUC {b.roc_auc:.4f} -> {n.roc_auc:.4f} | "
              f"ECE {b.ece15:.3f} -> {n.ece15:.3f} | KS(pneumonia) {b.KS_pos:.3f} -> {n.KS_pos:.3f}")
    return c


if __name__ == "__main__":
    main()
