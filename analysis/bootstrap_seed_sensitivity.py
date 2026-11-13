"""Monte Carlo sensitivity of the RSNA gate-minus-Swin-B balanced-accuracy bootstrap interval (Table III).
usage: python bootstrap_seed_sensitivity.py FOLDER HISTORICAL_JSON
FOLDER holds the private TRACE_CXR_INTERNAL_PREDICTIONS.csv; HISTORICAL_JSON is configs/historical.json.
Uses the same resampling as training/trace_cxr_controls.py (np.random.default_rng(seed); one
rng.integers(0, n, n) index vector per resample; percentile 95% interval). Frozen thresholds are not refitted."""
import json, sys
import numpy as np, pandas as pd
folder, hist = sys.argv[1], sys.argv[2]
lock = json.load(open(hist))["control_lock"]["models"]
d = pd.read_csv(f"{folder}/TRACE_CXR_INTERNAL_PREDICTIONS.csv")
y = d.label.values.astype(int); n = len(y); pos = (y == 1).astype(float)
cg = ((d.gated_fusion_calibrated.values >= lock["gated_fusion"]["threshold"]).astype(int) == y).astype(float)
cs = ((d.swin_b_calibrated.values >= lock["swin_b"]["threshold"]).astype(int) == y).astype(float)
diff = cg - cs
def ba_delta(w):
    return 0.5 * ((w * pos * diff).sum() / (w * pos).sum() + (w * (1 - pos) * diff).sum() / (w * (1 - pos)).sum())
def interval(seed, B):
    rng = np.random.default_rng(seed)
    v = np.array([ba_delta(np.bincount(rng.integers(0, n, n), minlength=n).astype(float)) for _ in range(B)])
    return np.percentile(v, 2.5), np.percentile(v, 97.5), v
print(f"full-sample delta BA {ba_delta(np.ones(n)):+.5f}")
lo, hi, _ = interval(20260908, 2000)
print(f"archived seed 20260908, 2,000 resamples: [{lo:+.5f}, {hi:+.5f}]")
lows = np.array([interval(s, 2000)[0] for s in range(1, 201)])
print(f"seeds 1-200, 2,000 resamples each: lower bound median {np.median(lows):+.5f}, "
      f"range [{lows.min():+.5f}, {lows.max():+.5f}], share <= 0: {(lows <= 0).mean():.3f}")
lo, hi, v = interval(20260908, 20000)
print(f"seed 20260908, 20,000 resamples: [{lo:+.5f}, {hi:+.5f}]; share of resamples <= 0: {(v <= 0).mean():.4f}")
