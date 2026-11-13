# RSNA patient overlap and patient-disjoint evaluation (Sections III-A and IV-A)

The RSNA `patientId` identifies an image. These scripts map each one to its NIH ChestX-ray14 patient with the public RSNA–NIH mapping file `pneumonia-challenge-dataset-mappings_2018.json` (30,000 records; not redistributed here, obtain it from the dataset custodians; SHA-256 of the copy used: `803ce79e…10bf27`). They also need the private `TRACE_CXR_RSNA_SPLIT_MANIFEST.csv` (or the public `data/rsna_split_manifest.csv`, which has the same `split` and `fold` columns and names the identifier `patientId`, as in the Kaggle `stage_2_train_labels.csv`; the private file calls it `id`, and both scripts accept either name), `TRACE_CXR_INTERNAL_PREDICTIONS.csv` and `TRACE_CXR_CONTROL_LOCK.json`. Only aggregate numbers are printed.

| Script | Output in `evidence/` | Paper values it produces |
|---|---|---|
| `rsna_patient_overlap_v2.py --mapping … --split … --preds … --lock …` | `patient_overlap_result.txt` | 11,452 NIH patients (9,150 / 2,125 / 3,727 per partition); 3,645 (68.3%) and 3,739 (70.1%) locked-test images from patients in development or in development/calibration; image- and patient-level bootstrap on the full locked test; patient-disjoint subset (1,598 images, 1,497 patients): gate 0.9138 [0.8916, 0.9342], Swin-B 0.9035, gate − Swin-B +0.0103 [+0.0022, +0.0197] |
| `patient_overlap_extra.py MAPPING SPLIT PREDS LOCK` | `patient_overlap_extra_result.txt` | mapping audit (one-to-one, no duplicates, no unmapped image); 1,531–1,608 patients shared between each OOF fold and the other folds; frozen-threshold Se 0.595 / Sp 0.934 (gate) and 0.595 / 0.924 (Swin-B) on the patient-disjoint subset |

Bootstrap: 2,000 resamples, seed 20260908, percentile 95% CI around the full-sample estimate; the patient bootstrap resamples NIH patients with all their images.
