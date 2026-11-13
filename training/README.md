# Historical training and evaluation code

This folder holds the code that produced the frozen models and the main results of the paper (Tables II–IV). It is released **as run**, for transparency. Only the dataset paths in `configs/trace_cxr.yaml` were replaced with placeholders. The CPU evidence checker in the repository root does not depend on this folder.

Running it needs the RSNA and Kermany images (see [Data access](../docs/data_access.md)), a CUDA GPU and the packages in `requirements.txt`. The paper used Python 3.13.12, PyTorch 2.10.0, torchvision 0.25.0 and CUDA 12.8 on one A100-SXM4-40GB.

## Files

| File | Purpose |
|---|---|
| `trace_cxr.py` | Main pipeline: preflight, RSNA split, five-fold out-of-fold training of both backbones (60 epochs), reliability-gate fitting, final-model training, calibration lock, locked-test inference and statistics |
| `trace_cxr_controls.py` | Matched fusion controls (learned constant weight, logistic stacking), fitted on the same out-of-fold scores |
| `trace_cxr_kermany.py` | Frozen Kermany inference with the original raster path (**KermanyRaw**: JPEG read without percentile scaling) |
| `kermany_consistency_check.py` | Frozen Kermany inference with the RSNA-consistent 0.5–99.5 percentile scaling (**KermanyNorm**, the primary external result); writes `TRACE_CXR_KERMANY_NORM_PREDICTIONS.csv`, re-checks that the Raw path reproduces the archived Raw scores, and writes a time-stamped protocol before inference (archived copy: [`../evidence/kermany_norm_protocol.json`](../evidence/kermany_norm_protocol.json)) |
| `patch_raster_normalization.py` | Optional one-line patch that makes `read_cxr` apply the same percentile scaling to raster images permanently (equivalent to the KermanyNorm reader above) |
| `trace_cxr_shift_analysis.py` | Prior-shift (Saerens EM) and class-conditional shift analyses on saved predictions |
| `trace_cxr_external.py` | Six-candidate frozen metrics, paired bootstrap, prior-shift diagnostics and the primary local recalibration for an external cohort from saved predictions. In this study it was run only on Kermany (`--name KermanyNorm`, reusing `TRACE_CXR_KERMANY_NORM_PREDICTIONS.csv`); its `_FEWSHOT.csv` output is the source of Table IV. No other cohort was evaluated, and the VinDr names in its usage text are only an example |
| `trace_cxr_xai.py` | InceptionV3 Grad-CAM, Swin-B Integrated Gradients, deletion AOPC and pointing game |
| `gate_sensitivity.py` | Post hoc check of the gate's training length and initialization |
| `measure_inference_time.py` | The **earlier** sequential per-stage timing. The matched latency reported in the paper comes from a separate interleaved six-candidate benchmark |
| `configs/trace_cxr.yaml` | Configuration used for the paper; replace the `/path/to/...` entries |

## Running the main pipeline

```bash
python trace_cxr.py preflight --config configs/trace_cxr.yaml
python trace_cxr.py split     --config configs/trace_cxr.yaml
python trace_cxr.py oof       --config configs/trace_cxr.yaml
python trace_cxr.py gate      --config configs/trace_cxr.yaml
python trace_cxr.py final     --config configs/trace_cxr.yaml
python trace_cxr.py lock      --config configs/trace_cxr.yaml
python trace_cxr.py internal  --config configs/trace_cxr.yaml
python trace_cxr.py stats     --config configs/trace_cxr.yaml
```

To reuse the exact partitions of the paper rather than regenerate them, compare the output of `split` with [`../data/rsna_split_manifest.csv`](../data/rsna_split_manifest.csv) (`python reproduce.py splits` in the repository root checks that file).

The frozen weights produced by `final` and `gate` (`TRACE_CXR_FINAL_INCEPTIONV3.pt`, `TRACE_CXR_FINAL_SWINB.pt`, `TRACE_CXR_RELIABILITY_GATE.pt`) are provided as assets of GitHub Release v1.0.0, with SHA-256 values in `SHA256SUMS.txt`.

To reproduce the primary external scores (KermanyNorm) after `lock`, run

```bash
python kermany_consistency_check.py --config configs/trace_cxr.yaml   # GPU; uses trace_cxr_kermany.py and trace_cxr_external.py
```

The script also runs the six-candidate external analysis (`TRACE_CXR_EXT_KermanyNorm_*.csv`, including the primary local recalibration of Table IV) on both versions. `trace_cxr_kermany.py all` alone reproduces only the earlier KermanyRaw scores. The explainability analyses of the paper used the KermanyRaw path.

## Differences from the paper's wording

- `ECE_oracle` and "oracle prior correction" in the shift-analysis scripts correspond to the **true-prevalence reference** in the paper.
- The InceptionV3 `transform_input` fix (see [Limitations](../docs/limitations.md)) is already applied in `trace_cxr.py`.
- The patient-overlap and patient-disjoint subset analysis is in [`../analysis/patient_overlap/`](../analysis/patient_overlap/README.md). The corrected development analysis and the exploratory corrected fold ensemble were run with separate scripts and are available here only as aggregate results. The interleaved latency benchmark is in [`../analysis/latency/`](../analysis/latency/README.md). The 500-draw local-recalibration code is in [`../analysis/recalibration_draws/`](../analysis/recalibration_draws/README.md), and the source-prior sensitivity of the prior correction is in [`../analysis/prior_shift/`](../analysis/prior_shift/README.md). Their aggregate results are in [`../evidence/`](../evidence/) and [Results](../docs/results.md).
- The splits are disjoint by RSNA `patientId`, **not** by patient. The code does not do patient-grouped splitting; the header comment of `trace_cxr.py` that mentions a "patient/image-level RSNA split" predates this check and is left unchanged so that the file stays identical to the version that was run.
