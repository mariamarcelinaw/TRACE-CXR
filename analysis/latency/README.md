# Interleaved latency benchmark (Section IV-F)

`benchmark_interleaved.py` is the as-run script of the matched six-candidate benchmark (only the private default `--root` path was removed). It needs a CUDA GPU, the RSNA DICOM images and a private `reference/` folder next to the script holding `trace_cxr.py` (from `../../training/`), `trace_cxr.yaml`, `TRACE_CXR_RSNA_SPLIT_MANIFEST.csv`, `TRACE_CXR_INTERNAL_PREDICTIONS.csv`, `TRACE_CXR_CONTROL_LOCK.json` and `TRACE_CXR_RELIABILITY_GATE.pt`; the final backbone weights are the Release v1.0.0 assets. `reference/` is ignored by Git.

Protocol as run: 200 locked-test images plus 20 warm-up images per candidate, drawn with the study seed; 3 rounds; batch size 1; float32 without autocast; `torch.inference_mode()`; both backbones resident in GPU memory for every candidate; candidate order shuffled per round (seed 20260908 + round) and interleaved within each image; `torch.cuda.synchronize()` at every timer boundary; timing from DICOM read to the CPU decision. Before timing, the outputs of all six candidates are checked against the archived locked-test predictions (tolerance 5e-4) and the checkpoint SHA-256 values are verified.

The raw timing file is not distributed; `python reproduce.py timing --data-root <folder>` recomputes the reported means from it.
