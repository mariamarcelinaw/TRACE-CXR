# TRACE-CXR

**Reliability-aware fusion and calibration transport for pneumonia classification on chest X-rays.**

Research software and aggregate evidence for the paper *TRACE-CXR: Evaluating Reliability-Aware Fusion and Calibration Transport for Pneumonia Classification on Chest X-Rays* (IEEE DeSE 2026). The code is released under the [MIT License](LICENSE). This is **not a clinical diagnostic system**.

The repository contains a CPU-only command-line tool that checks the frozen configuration, the fusion and calibration formulas and the published aggregate results; the RSNA partition list (`data/`); and the historical training and evaluation code (`training/`). The frozen model weights are attached to [Release v1.0.0](https://github.com/mariamarcelinaw/TRACE-CXR/releases/tag/v1.0.0). It does not contain medical images, labels, per-image predictions or NIH patient identifiers (the RSNA `patientId` in `data/` identifies an image, not a patient).

![TRACE-CXR workflow](figures/pipeline.png)


## Key results

| Evidence | Result | Caveat |
|---|---|---|
| RSNA partitions | 18,678 development; 2,669 calibration; 5,337 locked test | Disjoint by RSNA `patientId` (one image each); patient independence is discussed under [Limitations](#limitations) |
| RSNA locked test, ROC-AUC | Gate 0.9020; constant weight 0.9020; Swin-B 0.8977 | Gate matches the constant weight; image-level bootstrap CIs condition on fixed models |
| Kermany external, ROC-AUC (KermanyNorm) | Swin-B 0.9377; gate 0.9206 | With the reported models Swin-B ranks first; in the exploratory nested-selection ensemble the two are statistically indistinguishable (gate 0.9361, Swin-B 0.9327); the raw-JPEG sensitivity check is in [Results](docs/results.md) |
| Frozen external calibration | Gate ECE15 0.390; calibration slope 2.12 | RSNA-fitted calibration needs local adjustment; reliability diagrams in [Results](docs/results.md) |
| Local recalibration with 20 labeled Kermany images | Gate ECE15 0.093; Se 0.839; Sp 0.814 | Retrospective; sensitivity gain comes with lower specificity |
| Matched end-to-end latency (A100, 200 RSNA images × 3 interleaved rounds) | Swin-B 100.4 ms; gate 113.2 ms (+12.8 ms) | Single GPU session; includes DICOM reading and preprocessing |
| Branch attribution faithfulness (AOPC gain over random deletion) | See [Results](docs/results.md) | Explains each branch, not the complete gated predictor |

![ROC-AUC on the external cohort](figures/performance.png)

## Install and quick start

Python 3.10+ is enough. The CLI uses only the standard library and never trains models, downloads images, runs GPU inference or fits anything on external labels.

```bash
python reproduce.py --help
python reproduce.py verify      # check file checksums
python reproduce.py all         # run every check that the public files allow
python -m unittest discover -s tests -p 'test_*.py' -v
```

Checks that need per-image records are reported as **SKIPPED**, not as reproduced. Reports are written to `reports/` (ignored by Git).

Authorized users who hold the original per-image CSV files can run the stronger checks:

```bash
python reproduce.py metrics --data-root /path/to/private/evidence
python reproduce.py xai     --data-root /path/to/private/evidence
python reproduce.py timing  --data-root /path/to/private/evidence
```

No case-level data are written to the reports. Re-running training or GPU inference needs the datasets, a GPU and the code in [`training/`](training/README.md); the frozen weights are in [Release v1.0.0](https://github.com/mariamarcelinaw/TRACE-CXR/releases/tag/v1.0.0). See [Reproducibility](docs/reproducibility.md) and [Data access](docs/data_access.md).

## What is and is not reproducible here

| Component | In this repository | Status |
|---|---|---|
| Aggregate results | Yes | Checksummed; traced to archived result files |
| Frozen temperatures, thresholds and fusion weights | Yes (`configs/historical.json`) | Checked by `reproduce.py calibration` |
| Split and fold validity | Yes (`data/rsna_split_manifest.csv`) | Checked by `reproduce.py splits`; image-identifier level |
| Patient overlap (NIH patients) | Code in `analysis/patient_overlap/`; aggregate outputs in `evidence/patient_overlap_*.txt` | Needs the public RSNA–NIH mapping file and the private prediction file |
| Nested-selection development analysis (inner-validation checkpoints, cross-fitted fusion, five gate seeds) | Split list `data/nested_inner_split_manifest.csv`; aggregates `evidence/nested_selection_*.csv`; code `analysis/nested_selection/` | Code reruns from the private nested-selection OOF file; backbone training not included |
| Locked-test metrics | Only with the private prediction file | Recomputed from per-image scores |
| Local-recalibration draws | Yes (`evidence/recalibration_*.csv`, `analysis/recalibration_draws/`) | Full draw distributions, fitted intercepts/slopes and pre-specified poor-adaptation rates; code reruns from the private score file |
| Prior-shift correction, source-prior sensitivity | Yes (`evidence/prior_source_sensitivity.csv`, `analysis/prior_shift/`) | Code reruns from the private score files |
| Training and GPU inference | Code in `training/`; weights in Release v1.0.0 | Needs the datasets and a GPU; not re-run by this tool |
| Attribution maps | No | Aggregate values only |
| Latency | Only with the private raw timing file | Recomputed from per-image timings |

## Limitations

The RSNA partitions are disjoint by `patientId`, which in the Kaggle release identifies a unique image, not a person. Through the RSNA–NIH mapping file, 3,645 of the 5,337 locked-test images (68.3%) belong to NIH patients who also appear in development (3,739, 70.1%, in development or calibration). On the 1,598 locked-test images from patients absent from development and calibration (1,497 patients; patient-level bootstrap), the gate reached ROC-AUC 0.9138 versus 0.9035 for Swin-B and again matched the constant weight, while frozen-threshold sensitivity was 0.595 for both, so the full-test sensitivity may be optimistic. Patient-grouped repartitioning requires full retraining and is left to future work. The main out-of-fold scores share checkpoint selection with their folds; the nested-selection sensitivity analysis ([`analysis/nested_selection/`](analysis/nested_selection/README.md)) gave the same ranking of the fusion rules and was not used for the final models. Independent backbone training seeds were not run. Three settings were finalized after initial scores had been computed: the constant-weight and stacking controls were locked, InceptionV3 `transform_input` was enabled at inference (temperatures and thresholds refitted on the RSNA calibration split), and the Kermany percentile scaling (KermanyNorm) was adopted to match the RSNA preprocessing under a time-stamped protocol ([`evidence/kermany_norm_protocol.json`](evidence/kermany_norm_protocol.json)); none used test labels for fitting, and both Kermany versions are reported. Local adaptation is retrospective and uses Kermany labels, so it is kept separate from the frozen external evaluation. Details are in [Limitations](docs/limitations.md).

## Citation and license

Code and documentation: MIT License (see [LICENSE](LICENSE)). The RSNA and Kermany datasets and the torchvision pretrained weights are **not** included and remain under their own terms. The released model weights were trained on RSNA Pneumonia Detection Challenge data; their use is subject to that dataset's terms, for research only and not for clinical decision-making. Citation metadata are in [`CITATION.cff`](CITATION.cff); the proceedings DOI will be added after publication.
