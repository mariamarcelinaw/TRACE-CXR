# Reproducibility levels and commands

The CLI in this repository is a CPU-only evidence checker written for the public release. The original training and evaluation code is in [`training/`](../training/README.md), and the frozen weights are in [Release v1.0.0](https://github.com/mariamarcelinaw/TRACE-CXR/releases/tag/v1.0.0).

- `python reproduce.py verify`: check the SHA-256 checksums of the repository files.
- `python reproduce.py audit`: check the frozen fitting-role policy (fusion on development out-of-fold scores, calibration on RSNA calibration, Kermany evaluation-only) and scan for files that must not be public.
- `python reproduce.py calibration`: validate the six frozen temperature/threshold entries and the fusion formulas.
- `python reproduce.py metrics`: recompute locked-test metrics from the private `TRACE_CXR_INTERNAL_PREDICTIONS.csv`; otherwise SKIPPED.
- `python reproduce.py splits`: check identifier-level disjointness, fold validity and partition sizes from the public `data/rsna_split_manifest.csv` (or a private `TRACE_CXR_RSNA_SPLIT_MANIFEST.csv` given with `--data-root`).
- `python reproduce.py xai`: recompute the Kermany AOPC means from the private `TRACE_CXR_XAI_FAITHFULNESS.csv`; otherwise SKIPPED.
- `python reproduce.py timing`: summarize the private raw timing file of the interleaved benchmark (`INTERLEAVED_LATENCY_RAW_TIMINGS.csv`, or a single archived file whose name ends in `SIX_CANDIDATE_RAW_TIMINGS.csv`) and checks the six candidate means against `configs/historical.json`. If only the older component timing file `TRACE_CXR_INFERENCE_TIME.csv` is present, it is summarized and flagged as `PASS_OLDER_COMPONENT_TIMING_ONLY`, because it is not the matched latency comparison; otherwise SKIPPED.
- `python reproduce.py all`: run every check that the available files allow. It never trains, downloads or uses a GPU.

`analysis/gate_convergence_check.py <folder>` repeats the gate training-length check from the private out-of-fold scores (`TRACE_CXR_OOF_FUSION.csv`) and split manifest. It uses only development data and needs numpy, scipy and scikit-learn.

The per-image data are intentionally not distributed. GPU training, attribution and DICOM inference need the datasets, a GPU and the code in `training/`.

The archived KermanyNorm scores were reproduced on the original GPU server. The interleaved latency results were recomputed from the raw per-image timing records; the GPU benchmark itself was not re-run by this tool.

A non-zero exit status means a check failed; the CLI then prints the error and writes no `PASS` result. An archived value counts as reproduced only when the per-image data are present and the numerical check passes.
