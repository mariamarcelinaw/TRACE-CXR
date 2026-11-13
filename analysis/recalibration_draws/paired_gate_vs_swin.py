"""Paired gate-minus-Swin-B comparison of local logistic recalibration on identical draws.
usage: python paired_gate_vs_swin.py REPLICATE_DRAW_CSV OUT_CSV
REPLICATE_DRAW_CSV is the draw-level output of stage07_run.py (STAGE07_POSTHOC_DRAW_LEVEL_RESULTS.csv; private).
In the replicate, the per-draw seed depends on cohort, strategy, n and draw index but not on the model, so all
models are recalibrated on the same labeled subset and evaluated on the same remaining images (checked below via
adapt_index_sha256). The primary run behind Table IV draws its subsets from one random stream in model order, so
its rows are not paired across models; use this replicate for any between-model comparison."""
import sys
import pandas as pd
d = pd.read_csv(sys.argv[1], low_memory=False)
d = d[(d.method == "full_logistic") & (d.status == "SUCCESS")]
key = ["strategy", "n", "draw_id"]
g = d[d.model == "gated_fusion"].set_index(key)
s = d[d.model == "swin_b"].set_index(key)
j = g.join(s, lsuffix="_g", rsuffix="_s", how="inner")
same = (j.adapt_index_sha256_g == j.adapt_index_sha256_s) & (j.eval_index_sha256_g == j.eval_index_sha256_s)
if not same.all():
    raise SystemExit(f"draws are not paired: {int((~same).sum())} mismatching subsets")
rows = []
for (strategy, n), x in j.groupby(level=[0, 1]):
    r = dict(strategy=strategy, n=n, paired_draws=len(x))
    for m in ["ECE15", "BA", "Se", "Sp"]:
        dd = x[m + "_g"] - x[m + "_s"]
        r[f"{m}_gate_minus_swin_mean"] = dd.mean()
        r[f"{m}_gate_minus_swin_p2_5"] = dd.quantile(0.025)
        r[f"{m}_gate_minus_swin_p97_5"] = dd.quantile(0.975)
        r[f"{m}_share_gate_better"] = ((dd < 0) if m == "ECE15" else (dd > 0)).mean()
    rows.append(r)
pd.DataFrame(rows).round(4).to_csv(sys.argv[2], index=False)
print(f"paired draws {len(j)}; identical adaptation and evaluation subsets: {bool(same.all())}")
