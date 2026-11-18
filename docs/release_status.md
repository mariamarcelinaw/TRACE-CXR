# Release status

- **License.** Code and documentation are released under the MIT License ([LICENSE](../LICENSE)).
- **Included.** CPU evidence checker, aggregate evidence, the RSNA `patientId`-to-partition list (`data/`), historical training and evaluation code (`training/`). Model weights are attached to [Release v1.0.0](https://github.com/mariamarcelinaw/TRACE-CXR/releases/tag/v1.0.0).
- **Excluded.** Medical images, per-image predictions, labels and NIH patient identifiers are not part of the release (the RSNA `patientId` in `data/` identifies an image, not a patient). The RSNA and Kermany datasets and the torchvision weights keep their own terms.
- **Citation.** `CITATION.cff` describes version 1.0.0. The repository URL is included; add the proceedings DOI once it exists.
- **Continuous integration.** `.github/workflows/tests.yml` runs the unit tests and the public checks on every push; the first GitHub-hosted run happens after upload.
