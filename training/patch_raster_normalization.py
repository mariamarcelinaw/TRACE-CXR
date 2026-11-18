#!/usr/bin/env python3
"""
Fix: apply the DICOM intensity normalization (trace_cxr._to_uint8, percentile 0.5-99.5 stretch)
to raster images too, so Kermany JPEGs are preprocessed exactly like RSNA DICOMs.
DICOM handling is untouched, so RSNA results do not change. Idempotent; backup kept.
Usage (program directory):  python patch_raster_normalization.py
"""
import hashlib, shutil
from pathlib import Path
P = Path(__file__).resolve().parent / "trace_cxr.py"
MARK = "TRACE-CXR FIX raster normalization"
OLD = '    return Image.open(path).convert("L").convert("RGB")\n'
NEW = (f'    # {MARK}: same percentile stretch as DICOM (consistent preprocessing across cohorts)\n'
       '    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)\n'
       '    return Image.fromarray(_to_uint8(arr, None)).convert("RGB")\n')
src = P.read_text(encoding="utf-8")
if MARK in src:
    print("[OK] raster normalization already applied")
else:
    if src.count(OLD) != 1:
        raise SystemExit("[ERROR] raster branch of read_cxr not found exactly once")
    shutil.copy2(P, P.with_suffix(".py.pre_raster_norm"))
    before = hashlib.sha256(src.encode()).hexdigest(); src = src.replace(OLD, NEW); P.write_text(src, encoding="utf-8")
    print(f"[PATCHED] {P.name}\n  sha256 before {before}\n  sha256 after  {hashlib.sha256(src.encode()).hexdigest()}")
