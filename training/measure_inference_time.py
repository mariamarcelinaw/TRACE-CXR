#!/usr/bin/env python3
"""
Measure per-image inference time and parameter counts of TRACE-CXR.

Stages timed per image (batch size 1, after warm-up):
  1. read + preprocess (DICOM read, normalization, resize, ImageNet norm.)  [CPU]
  2. InceptionV3 forward                                                    [GPU]
  3. Swin-B forward                                                         [GPU]
  4. fusion: features + gate + mixture + temperature + threshold            [CPU]
Uses the final backbones, the locked gate and the locked T/tau from the corrected run.

Usage (program directory, GPU):
    python measure_inference_time.py            # 200 RSNA locked-test images
    python measure_inference_time.py --n 500
Writes <out>/TRACE_CXR_INFERENCE_TIME.csv and <out>/paper/inference_time.tex
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import trace_cxr as T  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_params(model, exclude_prefix=None):
    return sum(p.numel() for n, p in model.named_parameters()
               if exclude_prefix is None or not n.startswith(exclude_prefix))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/trace_cxr.yaml")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    cfg = T.load_cfg(ROOT / args.config)
    T.set_seed(cfg["experiment"]["seed"])
    out = T.get_out(cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inc = T.load_final_model("inception_v3", cfg, out).to(dev).eval()
    swin = T.load_final_model("swin_b", cfg, out).to(dev).eval()
    assert getattr(inc, "transform_input", True), "transform_input fix missing: run patch_transform_input.py"
    gate = T.load_reliability_gate(cfg, out).eval()
    lock = json.loads((out / "TRACE_CXR_DEPLOYMENT_LOCK.json").read_text())["models"]["gated_fusion"]
    t_g, tau_g = float(lock["temperature"]), float(lock["threshold"])
    tf = T.build_transforms(cfg["experiment"]["image_size"], "inception_v3", train=False)

    man = pd.read_csv(out / "TRACE_CXR_RSNA_SPLIT_MANIFEST.csv")
    paths = man.loc[man.split == "locked_test", "path"].sample(
        n=min(args.n + args.warmup, (man.split == "locked_test").sum()),
        random_state=cfg["experiment"]["seed"]).tolist()

    rows = []
    with torch.no_grad():
        for k, p in enumerate(paths):
            t0 = time.perf_counter()
            x = tf(T.read_cxr(p)).unsqueeze(0)
            t1 = time.perf_counter()
            xg = x.to(dev); sync()
            t2 = time.perf_counter()
            zi, _ = T.forward_logits(inc, xg, "inception_v3"); sync()
            t3 = time.perf_counter()
            zs, _ = T.forward_logits(swin, xg, "swin_b"); sync()
            t4 = time.perf_counter()
            pi = torch.sigmoid(zi).float().cpu().numpy(); ps = torch.sigmoid(zs).float().cpu().numpy()
            feats = torch.tensor(T.gate_features(pi, ps), dtype=torch.float32)
            a = gate(feats).numpy()
            pg = np.clip(a * pi + (1 - a) * ps, 1e-6, 1 - 1e-6)
            q = 1 / (1 + np.exp(-np.log(pg / (1 - pg)) / t_g))
            _ = q >= tau_g
            t5 = time.perf_counter()
            if k >= args.warmup:
                rows.append(dict(read_preprocess=(t1 - t0) * 1e3, transfer=(t2 - t1) * 1e3,
                                 inception=(t3 - t2) * 1e3, swin=(t4 - t3) * 1e3,
                                 fusion_calibration=(t5 - t4) * 1e3, total=(t5 - t0) * 1e3))
    df = pd.DataFrame(rows)
    df.to_csv(out / "TRACE_CXR_INFERENCE_TIME.csv", index=False)

    params = {"inception_v3": count_params(inc, exclude_prefix="AuxLogits"),
              "inception_v3_with_aux": count_params(inc), "swin_b": count_params(swin),
              "gate": count_params(gate)}
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    summ = df.agg(["mean", "std"]).T
    print(f"device: {gpu} | images timed: {len(df)} | batch size 1")
    print(summ.round(3).to_string())
    print("parameters:", {k: f"{v:,}" for k, v in params.items()})

    m = lambda c: f"{summ.loc[c, 'mean']:.2f}"
    s = lambda c: f"{summ.loc[c, 'std']:.2f}"
    mac = {"TimePre": m("read_preprocess"), "TimePreSD": s("read_preprocess"),
           "TimeInc": m("inception"), "TimeIncSD": s("inception"), "TimeSwin": m("swin"),
           "TimeSwinSD": s("swin"), "TimeFuse": f"{summ.loc['fusion_calibration', 'mean']:.3f}",
           "TimeFuseSD": f"{summ.loc['fusion_calibration', 'std']:.3f}", "TimeTotal": m("total"),
           "TimeTotalSD": s("total"), "TimeN": str(len(df)), "TimeGPU": gpu,
           "ParamInc": f"{params['inception_v3'] / 1e6:.1f}M", "ParamSwin": f"{params['swin_b'] / 1e6:.1f}M",
           "ParamGate": f"{params['gate']}"}
    (out / "paper").mkdir(exist_ok=True)
    with open(out / "paper" / "inference_time.tex", "w") as fh:
        fh.write("% AUTO-GENERATED by measure_inference_time.py\n")
        for k, v in mac.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("\nwrote", out / "TRACE_CXR_INFERENCE_TIME.csv", "and", out / "paper" / "inference_time.tex")


if __name__ == "__main__":
    main()
