# Data and checkpoint access

RSNA radiographs, Kermany images, DICOM metadata and per-image predictions are **not included**. The repository holds only the RSNA `patientId`-to-partition list (`data/rsna_split_manifest.csv`, columns `patientId`, `split`, `fold`; no labels or paths). `patientId` is the identifier of the Kaggle `stage_2_train_labels.csv` file; it names one DICOM image, not a biological patient. `data/corrected_inner_split_manifest.csv` lists, for each development fold of the corrected analysis, which `patientId` values formed the inner training, inner validation and outer held-out sets (no labels). The frozen weights are attached to [Release v1.0.0](https://github.com/mariamarcelinaw/TRACE-CXR/releases/tag/v1.0.0) with SHA-256 values; they were trained on RSNA challenge data and are for research use under that dataset's terms. Obtain the images from their dataset custodians under their own terms:

- RSNA Pneumonia Detection Challenge (Kaggle competition data; NIH ChestX-ray14 mapping published with Shih et al., *Radiology: Artificial Intelligence*, 2019).
- Kermany et al., *Cell*, 2018 (Mendeley Data, "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification").

The MIT License of this repository covers its code and documentation only. It grants no rights to these datasets or to the torchvision pretrained weights.

The study's internal records hold file digests and checkpoint SHA-256 values; these do not transfer any distribution rights.

The CLI accepts a local `--data-root` for authorized private CSV files. It never copies those per-image records into the repository and writes only aggregate output. Keep such files outside version control; `.gitignore` already excludes common data, checkpoint and output paths.
