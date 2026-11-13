#!/usr/bin/env python3
"""Supplementary aggregate checks for the RSNA patient analysis (no patient IDs printed).

usage: python patient_overlap_extra.py MAPPING_JSON SPLIT_MANIFEST_CSV INTERNAL_PREDICTIONS_CSV CONTROL_LOCK_JSON

1. mapping audit: one RSNA patientId -> one NIH image; NIH patient = first 8 digits of img_id
2. patients shared between each OOF fold and the other development folds
3. frozen-threshold Se/Sp of the gate and Swin-B on the patient-disjoint locked-test subset
"""
import json, sys
import numpy as np, pandas as pd

mp, sp, pp, lp = sys.argv[1:5]
m = pd.DataFrame(json.load(open(mp)))
m['pat'] = m.img_id.str[:8]
print(f"mapping records {len(m)}; unique RSNA ids {m.subset_img_id.nunique()}; unique NIH images {m.img_id.nunique()}; "
      f"img_id pattern ok {int(m.img_id.str.fullmatch(r'[0-9]{8}_[0-9]{3}[.]png').sum())}")
s = pd.read_csv(sp)
j = s.merge(m[['subset_img_id', 'img_id', 'pat']], left_on='id', right_on='subset_img_id', how='left')
print(f"split rows {len(j)}; unmapped {int(j.pat.isna().sum())}; duplicate NIH images {int(j.img_id.duplicated().sum())}")
c = j.pat.value_counts()
print(f"patients {len(c)}; with >1 image {int((c > 1).sum())}; max images per patient {int(c.max())}")
dev = j[j.split == 'development'].copy(); dev['fold'] = dev.fold.astype(float).astype(int)
for f in range(5):
    a = set(dev[dev.fold == f].pat); b = set(dev[dev.fold != f].pat)
    print(f"OOF fold {f}: patients {len(a)}, shared with other folds {len(a & b)}")
p = pd.read_csv(pp); lock = json.load(open(lp))
seen = set(j[j.split.isin(['development', 'calibration'])].pat)
p = p.merge(j[['id', 'pat']], on='id'); clean = ~p.pat.isin(seen)
print(f"patient-disjoint locked-test subset: {int(clean.sum())} images, {p[clean].pat.nunique()} patients, "
      f"{int(p[clean].label.sum())} positives")
def lg(x):
    x = np.clip(x, 1e-6, 1 - 1e-6); return np.log(x / (1 - x))
y = p.label.values
scores = {'gated_fusion': p.gated_calibrated.values,
          'swin_b': 1 / (1 + np.exp(-lg(p.swin_b.values) / lock['models']['swin_b']['temperature']))}
for name, q in scores.items():
    pred = q >= lock['models'][name]['threshold']
    for label, k in (('patient-disjoint', clean.values), ('full test', np.ones(len(p), bool))):
        yy, pr = y[k], pred[k]
        print(f"{name:12s} {label:16s} Se {(pr & (yy == 1)).sum() / (yy == 1).sum():.3f}  Sp {(~pr & (yy == 0)).sum() / (yy == 0).sum():.3f}")
