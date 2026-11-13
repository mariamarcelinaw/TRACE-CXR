# Evidence

Only aggregate results are included. Per-image records, file paths, checkpoint binaries and medical images are not. This directory holds:

- the SHA-256 checksums of the repository files (`checksums.sha256`);
- a table of aggregate results with their provenance (`verified_results.csv`);
- 15-bin reliability values (`reliability_bins.csv`) and calibration summaries (`calibration_summary.csv`);
- the local-recalibration draw summary for every model, budget, method and sampling strategy (`recalibration_draw_summary.csv`), full draw distributions (`recalibration_draw_distribution.csv`, `recalibration_primary_run_distribution.csv`), pre-specified poor-adaptation rates (`recalibration_poor_adaptation.csv`) and the class mix of the labeled subsets (`recalibration_unstratified_class_mix.csv`);
- gate-minus-Swin-B recalibration differences on identical replicate draws (`recalibration_gate_vs_swin_paired.csv`) and the Monte Carlo stability check of the RSNA BA bootstrap interval (`bootstrap_seed_sensitivity_result.txt`);
- the prior-shift correction under two source-prior definitions (`prior_source_sensitivity.csv`);
- the NIH-patient overlap audit (`patient_overlap_result.txt`, `patient_overlap_extra_result.txt`);
- the gate training-length check on the original out-of-fold scores (`gate_convergence_result.txt`);
- the corrected development analysis: inner/outer counts, held-out metrics, paired bootstrap, per-fold audit and five gate initializations (`corrected_development_*.csv`, `gate_initialization_stability.csv`).

Each file is described in `manifest.csv`. These files are **not sufficient** for independent retraining or for re-running every bootstrap analysis.
