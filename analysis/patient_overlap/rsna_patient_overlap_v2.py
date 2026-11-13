#!/usr/bin/env python3
"""TRACE-CXR: NIH-patient overlap + paired bootstrap on the patient-disjoint test subset.

Run (OUT = folder with the private prediction files):
  OUT=/path/to/outputs/trace_cxr_inception_swinb_v2
  python rsna_patient_overlap_v2.py --mapping pneumonia-challenge-dataset-mappings_2018.json \
     --split $OUT/TRACE_CXR_RSNA_SPLIT_MANIFEST.csv --preds $OUT/TRACE_CXR_INTERNAL_PREDICTIONS.csv \
     --lock $OUT/TRACE_CXR_CONTROL_LOCK.json | tee PATIENT_OVERLAP_RESULT_V2.txt

Only aggregate numbers are printed (no patient IDs).
Bootstrap: 2,000 resamples, seed 20260908, percentile 95% CI, full-sample point estimate.
  image   = resample images (as in Table III)
  patient = resample NIH patients with all their images (cluster bootstrap)
"""
import argparse, csv, json, re, sys
import numpy as np
from scipy.stats import rankdata

NIH_RE = re.compile(r"(\d{8})_\d{3}")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
NBOOT, SEED = 2000, 20260908


def load_mapping(path):
    data = json.load(open(path))
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), list(data.values()))
    rk = nk = None
    for rec in data[:200]:
        for k, v in rec.items():
            if rk is None and UUID_RE.match(str(v)):
                rk = k
            if nk is None and NIH_RE.search(str(v)):
                nk = k
    if not (rk and nk):
        sys.exit(f"mapping keys not found; sample record: {data[0]}")
    m = {}
    for rec in data:
        u, n = str(rec.get(rk, "")), NIH_RE.search(str(rec.get(nk, "")))
        if UUID_RE.match(u) and n:
            m[u] = n.group(1)
    return m


def auc(y, s):
    r = rankdata(s)
    npos = y.sum(); nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def sig(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def candidates(P, lock):
    pI = np.array([float(r["inception_v3"]) for r in P])
    pS = np.array([float(r["swin_b"]) for r in P])
    pG = np.array([float(r["gated_fusion"]) for r in P])
    a = lock["const_alpha_inception"]; ls = lock["logit_stack"]
    raw = {"inception_v3": pI, "swin_b": pS, "mean_fusion": (pI + pS) / 2, "gated_fusion": pG,
           "const_alpha": a * pI + (1 - a) * pS,
           "logit_stack": sig(ls["intercept"] + ls["coef_logit_inception"] * logit(pI) + ls["coef_logit_swin"] * logit(pS))}
    return {k: sig(logit(v) / lock["models"][k]["temperature"]) for k, v in raw.items()}


def boot_indices(groups, rng, unit):
    if unit == "image":
        n = sum(len(g) for g in groups)
        return [rng.integers(0, n, n) for _ in range(NBOOT)]
    out = []
    for _ in range(NBOOT):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(np.concatenate([groups[i] for i in pick]))
    return out


def report(title, y, Q, pats, unit):
    order = {}
    for i, p in enumerate(pats):
        order.setdefault(p, []).append(i)
    groups = [np.array(v) for v in order.values()]
    if unit == "image":
        groups = [np.arange(len(y))]
    rng = np.random.default_rng(SEED)
    idx = boot_indices(groups if unit == "patient" else [np.arange(len(y))], rng, unit)
    idx = [ii for ii in idx if 0 < y[ii].sum() < len(ii)]
    print(f"\n== {title} | bootstrap unit: {unit} | n={len(y)} images, {len(order)} patients, "
          f"{int(y.sum())} positive (prevalence {y.mean():.3f}), valid resamples {len(idx)} ==")
    for k in ["inception_v3", "swin_b", "mean_fusion", "gated_fusion", "const_alpha", "logit_stack"]:
        b = [auc(y[ii], Q[k][ii]) for ii in idx]
        print(f"  AUC {k:13s} {auc(y, Q[k]):.4f} [{np.percentile(b, 2.5):.4f}, {np.percentile(b, 97.5):.4f}]")
    for c in ["swin_b", "mean_fusion", "const_alpha", "inception_v3"]:
        d = [auc(y[ii], Q["gated_fusion"][ii]) - auc(y[ii], Q[c][ii]) for ii in idx]
        full = auc(y, Q["gated_fusion"]) - auc(y, Q[c])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  dAUC gate - {c:13s} {full:+.4f} [{lo:+.4f}, {hi:+.4f}]")


def main():
    ap = argparse.ArgumentParser()
    for a in ["--mapping", "--split", "--preds", "--lock"]:
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    m = load_mapping(a.mapping)
    rows = list(csv.DictReader(open(a.split)))
    miss = sum(r["id"] not in m for r in rows)
    print(f"mapping entries {len(m)}; split rows {len(rows)}; unmapped {miss}")
    pats = {}
    for r in rows:
        if r["id"] in m:
            pats.setdefault(r["split"], set()).add(m[r["id"]])
    allp = set().union(*pats.values())
    print(f"NIH patients in all {len(rows)} images: {len(allp)}")
    for s, ps in pats.items():
        print(f"  {s:12s} patients {len(ps)}")
    seen = pats.get("development", set()) | pats.get("calibration", set())
    dev = pats.get("development", set())

    lock = json.load(open(a.lock))
    P = [r for r in csv.DictReader(open(a.preds)) if r["id"] in m]
    n_dev = sum(m[r["id"]] in dev for r in P)
    n_seen = sum(m[r["id"]] in seen for r in P)
    print(f"locked-test images {len(P)}: patient also in development {n_dev} ({n_dev/len(P):.1%}); "
          f"in development or calibration {n_seen} ({n_seen/len(P):.1%})")

    y = np.array([int(r["label"]) for r in P]); Q = candidates(P, lock)
    tp = [m[r["id"]] for r in P]
    print(f"check: full-test gate AUC {auc(y, Q['gated_fusion']):.4f} (paper 0.9020)")
    report("FULL locked test", y, Q, tp, "image")
    report("FULL locked test", y, Q, tp, "patient")
    keep = np.array([m[r["id"]] not in seen for r in P])
    report("CLEAN subset (patients not in development/calibration)", y[keep],
           {k: v[keep] for k, v in Q.items()}, [p for p, k in zip(tp, keep) if k], "patient")


if __name__ == "__main__":
    main()
