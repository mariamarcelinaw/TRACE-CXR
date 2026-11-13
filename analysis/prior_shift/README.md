# Prior-shift correction: choice of the source prior

`prior_source_sensitivity.py FOLDER OUT.csv` recomputes the label-free EM prior correction and the true-prevalence reference on KermanyNorm (paper Section IV-E, Table IV) under two definitions of the source prior `pi_s`: the mean calibrated score on the RSNA calibration split (the paper's primary choice, the prior implied by the calibrated model) and the observed calibration-split prevalence (0.225). It uses the same `prior_adjust` and `em_prior` functions as `training/trace_cxr_shift_analysis.py`, refits nothing and writes only aggregate values.

It needs the private files `TRACE_CXR_CONTROL_LOCK.json`, `TRACE_CXR_CALIBRATION_PREDICTIONS.csv` and `TRACE_CXR_KERMANY_NORM_PREDICTIONS.csv` (not distributed) and numpy and pandas. The output for the archived inputs is [`evidence/prior_source_sensitivity.csv`](../../evidence/prior_source_sensitivity.csv); its `mean_score` rows reproduce the archived primary shift analysis exactly.
