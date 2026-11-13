# Local-recalibration draw analyses

Code for the retrospective local-recalibration analyses of the paper (Section III-F, Table IV). It needs the private per-image KermanyNorm scores (`TRACE_CXR_KERMANY_NORM_PREDICTIONS.csv`, SHA-256 `6deaf0f5…0c271`) and the control lock (`TRACE_CXR_CONTROL_LOCK.json`, SHA-256 `a34b7bbb…81b7`); neither is distributed. Requires numpy, pandas, scipy, scikit-learn and matplotlib.

| Script | What it does |
|---|---|
| `replay_primary_recalibration.py FOLDER OUT.csv` | Re-runs the **primary** run behind Table IV (seed 20260908, 500 stratified draws per budget, `LogisticRegression(C=1.0)` = L2, lbfgs, `max_iter=100`, `tol=1e-4`, unpenalized intercept, unconstrained slope; recalibrated threshold = labeled-subset prevalence). Logs convergence warnings, fitted intercept/slope and an adaptation/evaluation overlap check for every draw. On the archived inputs it reproduces all 96 Table IV source rows exactly, with 0 convergence warnings in 15,000 fits (at most 12 lbfgs iterations) and 0 overlaps. |
| `stage07_run.py --mode historical-exploratory --predictions … --lock … --output-dir NEW --acknowledge-posthoc` | The **replicate** analysis (500 stratified and 500 unstratified draws per budget; threshold-only, intercept-only and intercept+slope methods; `max_iter=1000`, `tol=1e-4`; per-draw seeds derived from 20260908 by SHA-256). Every draw asserts that adaptation and evaluation images are disjoint (`stage07_sampling.py`); single-class draws are kept as recorded failures. A rerun reproduced the archived 120,000 draw-level rows (identical seeds, subsets and metrics). |
| `paired_gate_vs_swin.py REPLICATE.csv OUT.csv` | Gate-minus-Swin-B differences of logistic recalibration on identical replicate draws (asserts that adaptation and evaluation subsets match). Writes `evidence/recalibration_gate_vs_swin_paired.csv`. The primary run is not paired across models. |
| `summarize_recalibration_draws.py REPLICATE.csv PRIMARY.csv EVIDENCE_DIR FIGURE.png` | Writes `evidence/recalibration_draw_distribution.csv`, `evidence/recalibration_poor_adaptation.csv`, `evidence/recalibration_primary_run_distribution.csv` and `figures/recalibration_draws.png`. |

The draws share evaluation images, so their spread is a descriptive range, not a confidence interval.
