# Corrected development analysis (Sections IV-B and IV-C)

The original out-of-fold (OOF) checkpoints were each selected on their own held-out fold. To remove that dependence, both backbones were retrained once per development fold with the checkpoint chosen on an **inner validation subset of that fold's training portion** (60 epochs, highest inner-validation ROC-AUC, ties broken by lower BCE). The outer held-out fold was used only to produce its OOF scores. The fusion rules were then cross-fitted over the same five folds: each rule was fitted on four folds and scored on the fifth.

| Item | Where |
|---|---|
| Inner-train / inner-validation / outer-held-out list per fold (RSNA `patientId`, no labels) | [`../../data/corrected_inner_split_manifest.csv`](../../data/corrected_inner_split_manifest.csv) |
| Counts per fold (13,148–13,149 / 1,794 / 3,735–3,736 images) | [`../../evidence/corrected_development_inner_outer_counts.csv`](../../evidence/corrected_development_inner_outer_counts.csv) |
| Held-out ROC-AUC, AP, Brier, NLL of the six candidates | [`../../evidence/corrected_development_metrics.csv`](../../evidence/corrected_development_metrics.csv) |
| Gate-minus-comparator paired bootstrap (500 image resamples) | [`../../evidence/corrected_development_paired_bootstrap.csv`](../../evidence/corrected_development_paired_bootstrap.csv) |
| Per-fold training sizes, constant weight and gate α | [`../../evidence/corrected_development_fold_audit.csv`](../../evidence/corrected_development_fold_audit.csv) |
| Five gate initializations (held-out ROC-AUC, α, differences) | [`../../evidence/gate_initialization_stability.csv`](../../evidence/gate_initialization_stability.csv) |

`corrected_crossfit_check.py CORRECTED_OOF_CSV SPLIT_MANIFEST_CSV OUT_DIR` recomputes all four evidence CSV files from the private corrected backbone OOF scores (18,678 rows, SHA-256 `fdb731e7…cf4b74a`) and the private split manifest with labels. It reads development data only. On the archived inputs it reproduces the archived values to six decimals. It needs numpy, pandas, scipy, scikit-learn and PyTorch (CPU is enough).

Paper values: held-out ROC-AUC 0.8894 (gate), 0.8892 (constant weight), 0.8844 (Swin-B); gate minus constant weight +0.0002 [+0.0000, +0.0003]; over five gate seeds, ROC-AUC 0.8891–0.8894, gate minus constant weight −0.0001 to +0.0002, mean α 0.380–0.499.

Limitations: the corrected backbones were trained once (one seed per fold), and the folds share training images, so this is a cross-fitted development estimate rather than fully nested validation. The partitions are disjoint by RSNA `patientId`, not by NIH patient. The script that retrained the backbones depends on server-side modules that are not part of this release; the split list above lets readers check that inner validation never touched the outer held-out fold. The corrected models were not used for the main tables.
