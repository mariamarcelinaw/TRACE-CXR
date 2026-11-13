from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone
import textwrap
import shutil

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter

from trace_cxr import (
    CXRDataset,
    ROOT,
    TRACE_EXP,
    TRACE_NAME,
    device_name,
    get_out,
    load_cfg,
    load_final_model,
    set_seed,
    update_experiment_manifest,
    write_json,
)
from trace_cxr_kermany import kermany_index


# ============================================================
# TRACE-CXR XAI FINAL REVISION
#
# QUALITATIVE
# - balanced confusion categories: TP, TN, FP, FN
# - InceptionV3: predicted-branch-class Grad-CAM
# - Swin-B: predicted-branch-class Integrated Gradients
# - fused qualitative panel added for supervisor-requested
#   before / after Inception / after Swin-B / after combined view
# - Swin-B attention aggregation retained as supplementary map
# - metadata: GT, TRACE prediction, branch probabilities,
#   gate alpha, fused probability, frozen threshold, entropy,
#   disagreement, and XAI target class for each branch
# - RSNA positive examples can show GT pneumonia boxes
#
# QUANTITATIVE
# RSNA positive cases with bounding boxes:
# - Pointing Game / Hit Rate
# - Energy Inside GT Box
# - IoU
# - Dice
# - normalized attribution entropy
# - attribution concentration
#
# FAITHFULNESS (RSNA and Kermany):
# - predicted-class top-attribution deletion
# - random deletion baseline
# - confidence drop at configured removal fractions
# - AOPC
# - deletion AUC
#
# IMPORTANT SCOPE
# These explanations are branch-scoped.
# They are NOT a causal explanation of the complete fused system.
# ============================================================


# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------

def _cfg_xai(cfg: dict) -> dict:
    x = dict(cfg.get("xai", {}))

    defaults = {
        "cases_per_confusion_group": 3,
        "qualitative_ig_steps": 32,
        "quantitative_ig_steps": 16,
        "rsna_localization_max_cases": 200,
        "faithfulness_max_cases_per_cohort": 100,
        "attribution_top_fraction": 0.20,
        "concentration_top_fraction": 0.10,
        "deletion_fractions": [0.0, 0.10, 0.20, 0.30, 0.40, 0.50],
        "faithfulness_random_repeats": 5,
        "bootstrap_iterations": 2000,
        "qualitative_overlay_alpha": 0.86,
        "qualitative_overlay_gamma": 0.65,
        # Identical, explicitly display-only processing for BOTH branch maps.
        "display_percentile": 99.5,
        "display_smoothing_sigma_px": 2.0,
        "display_low_cutoff": 0.035,
        "export_swin_diagnostic": True,
        "export_raw_qualitative_maps": True,
        "export_supplementary_attention": False,
        "export_difference_map": True,
    }

    for k, v in defaults.items():
        x.setdefault(k, v)

    return x


def validate_visual_config(xcfg: dict) -> None:
    if not (0 < float(xcfg["display_percentile"]) <= 100):
        raise ValueError("xai.display_percentile must be in (0, 100].")
    if float(xcfg["display_smoothing_sigma_px"]) < 0:
        raise ValueError("xai.display_smoothing_sigma_px must be >= 0.")
    if not (0 < float(xcfg["qualitative_overlay_alpha"]) <= 1):
        raise ValueError("xai.qualitative_overlay_alpha must be in (0, 1].")
    if float(xcfg["qualitative_overlay_gamma"]) <= 0:
        raise ValueError("xai.qualitative_overlay_gamma must be > 0.")
    if not (0 <= float(xcfg["display_low_cutoff"]) < 1):
        raise ValueError("xai.display_low_cutoff must be in [0, 1).")
    if int(xcfg["qualitative_ig_steps"]) < 1:
        raise ValueError("xai.qualitative_ig_steps must be >= 1.")


def stable_seed(*parts, base_seed=0) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2**32 - 1)


def normalize_map(arr):
    a = np.asarray(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = a - a.min()
    mx = a.max()
    if mx <= 0:
        return np.zeros_like(a, dtype=np.float32)
    return (a / mx).astype(np.float32)


def class_name(label: int) -> str:
    return "PNEUMONIA" if int(label) == 1 else "NORMAL"


def load_lock(out: Path) -> dict:
    p = out / "TRACE_CXR_DEPLOYMENT_LOCK.json"
    if not p.exists():
        raise FileNotFoundError(
            "TRACE_CXR_DEPLOYMENT_LOCK.json not found. "
            "Run trace_cxr.py lock first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def branch_threshold(lock: dict, model_name: str) -> float:
    return float(lock["models"][model_name]["threshold"])


def model_positive_probability(model, x: torch.Tensor) -> float:
    dev = device_name()
    model = model.to(dev).eval()
    with torch.no_grad():
        logits = model(x.unsqueeze(0).to(dev))
        if hasattr(logits, "logits"):
            logits = logits.logits
        p = torch.sigmoid(logits.view(-1)[0]).item()
    return float(p)


def target_class_confidence(model, x: torch.Tensor, target_class: int) -> float:
    p = model_positive_probability(model, x)
    return p if int(target_class) == 1 else (1.0 - p)


def target_score_from_output(output, target_class: int):
    if hasattr(output, "logits"):
        output = output.logits
    score = output.view(-1)[0]
    return score if int(target_class) == 1 else -score


def top_fraction_mask(attr, fraction: float) -> np.ndarray:
    a = np.asarray(attr, dtype=np.float64)
    flat = a.reshape(-1)
    n = len(flat)
    k = max(1, min(n, int(round(float(fraction) * n))))
    idx = np.argpartition(flat, n - k)[n - k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask.reshape(a.shape)


def normalized_attribution_entropy(attr) -> float:
    a = np.maximum(np.asarray(attr, dtype=np.float64), 0)
    total = float(a.sum())
    if total <= 0:
        return 1.0
    p = a.reshape(-1) / total
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    h = -float(np.sum(p * np.log(p)))
    return float(h / np.log(a.size))


def attribution_concentration(attr, top_fraction: float = 0.10) -> float:
    a = np.maximum(np.asarray(attr, dtype=np.float64), 0)
    total = float(a.sum())
    if total <= 0:
        return 0.0
    m = top_fraction_mask(a, top_fraction)
    return float(a[m].sum() / total)


def resize_map(arr, target_hw):
    t = torch.tensor(np.asarray(arr), dtype=torch.float32)[None, None]
    out = F.interpolate(
        t,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return out.numpy()


# ------------------------------------------------------------
# InceptionV3 class-specific Grad-CAM
# ------------------------------------------------------------

class InceptionGradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer = self.model.Mixed_7c

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, target_class: int):
        dev = device_name()
        self.model = self.model.to(dev).eval()

        batch = x.unsqueeze(0).to(dev)
        self.model.zero_grad(set_to_none=True)

        output = self.model(batch)
        score = target_score_from_output(output, target_class)
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        return normalize_map(cam.detach().cpu().numpy())

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


# ------------------------------------------------------------
# Swin-B class-specific Integrated Gradients
# ------------------------------------------------------------

def integrated_gradients(
    model,
    x: torch.Tensor,
    target_class: int,
    steps: int = 32,
    return_raw: bool = False,
):
    dev = device_name()
    model = model.to(dev).eval()

    inp = x.unsqueeze(0).to(dev)
    baseline = torch.zeros_like(inp)
    total_grad = torch.zeros_like(inp)

    # Midpoint Riemann approximation avoids duplicated endpoints.
    alphas = (torch.arange(steps, device=dev, dtype=inp.dtype) + 0.5) / steps

    for alpha in alphas:
        xi = (baseline + alpha * (inp - baseline)).detach()
        xi.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        output = model(xi)
        score = target_score_from_output(output, target_class)
        score.backward()

        total_grad += xi.grad.detach()

    avg_grad = total_grad / float(steps)
    ig = (inp - baseline) * avg_grad

    # Absolute channel aggregation for spatial importance.
    attr = ig.abs().sum(dim=1)[0].detach().cpu().numpy().astype(np.float32)
    normalized = normalize_map(attr)
    # Existing localization/faithfulness calls still receive EXACTLY the same
    # map as V4. Raw IG is exposed only for qualitative diagnostics.
    return (normalized, attr) if return_raw else normalized


# ------------------------------------------------------------
# Swin-B hierarchical attention aggregation (supplementary)
# ------------------------------------------------------------

def _relative_position_bias(module):
    try:
        n = module.window_size[0] * module.window_size[1]
        bias = module.relative_position_bias_table[
            module.relative_position_index.reshape(-1)
        ]
        bias = (
            bias.view(n, n, -1)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
        )
        return bias
    except Exception:
        return None


def swin_attention_density(module, x):
    if x.ndim != 4:
        return None

    b, h, w, c = x.shape
    wh, ww = module.window_size
    sh, sw = module.shift_size

    pad_b = (wh - h % wh) % wh
    pad_r = (ww - w % ww) % ww

    xp = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
    hp, wp = xp.shape[1], xp.shape[2]

    if sh or sw:
        xp = torch.roll(xp, shifts=(-sh, -sw), dims=(1, 2))

    nwin = (hp // wh) * (wp // ww)

    xw = (
        xp.view(b, hp // wh, wh, wp // ww, ww, c)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(b * nwin, wh * ww, c)
    )

    qkv = F.linear(xw, module.qkv.weight, module.qkv.bias)
    qkv = (
        qkv.reshape(
            b * nwin,
            wh * ww,
            3,
            module.num_heads,
            c // module.num_heads,
        )
        .permute(2, 0, 3, 1, 4)
    )

    q, k, _ = qkv
    q = q * ((c // module.num_heads) ** -0.5)
    attention = q @ k.transpose(-2, -1)

    bias = _relative_position_bias(module)
    if bias is not None:
        attention = attention + bias.to(attention.device)

    if sh or sw:
        mask = xp.new_zeros((hp, wp))

        h_slices = ((0, -wh), (-wh, -sh), (-sh, None))
        w_slices = ((0, -ww), (-ww, -sw), (-sw, None))

        counter = 0
        for h0, h1 in h_slices:
            for w0, w1 in w_slices:
                mask[h0:h1, w0:w1] = counter
                counter += 1

        mw = (
            mask.view(hp // wh, wh, wp // ww, ww)
            .permute(0, 2, 1, 3)
            .reshape(nwin, wh * ww)
        )

        am = mw.unsqueeze(1) - mw.unsqueeze(2)
        am = (
            am.masked_fill(am != 0, float(-100.0))
            .masked_fill(am == 0, float(0.0))
        )

        attention = (
            attention.view(
                b,
                nwin,
                module.num_heads,
                wh * ww,
                wh * ww,
            )
            + am.unsqueeze(0).unsqueeze(2)
        )

        attention = attention.view(
            b * nwin,
            module.num_heads,
            wh * ww,
            wh * ww,
        )

    attention = attention.softmax(dim=-1)

    # Attention received by each token, averaged over heads and queries.
    density = attention.mean(dim=1).mean(dim=1)

    density_map = (
        density.view(b, hp // wh, wp // ww, wh, ww)
        .permute(0, 1, 3, 2, 4)
        .reshape(b, hp, wp)
    )

    if sh or sw:
        density_map = torch.roll(
            density_map,
            shifts=(sh, sw),
            dims=(1, 2),
        )

    return density_map[:, :h, :w]


def swin_attention_aggregate(model, x: torch.Tensor):
    dev = device_name()
    model = model.to(dev).eval()

    captured = []
    handles = []

    def hook(module, inputs, output):
        try:
            dm = swin_attention_density(module, inputs[0].detach())
            if dm is not None:
                captured.append(dm.detach().cpu())
        except Exception as e:
            print("[XAI WARN] Swin attention block skipped:", e)

    for module in model.modules():
        if module.__class__.__name__.startswith("ShiftedWindowAttention"):
            handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        _ = model(x.unsqueeze(0).to(dev))

    for h in handles:
        h.remove()

    if not captured:
        raise RuntimeError("No Swin attention maps captured.")

    target_size = x.shape[-2:]
    aggregate = None
    total_weight = 0.0

    for idx, density in enumerate(captured, start=1):
        up = F.interpolate(
            density.unsqueeze(1).float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        weight = float(idx)
        aggregate = up * weight if aggregate is None else aggregate + up * weight
        total_weight += weight

    return normalize_map((aggregate / total_weight).numpy())


# ------------------------------------------------------------
# Images / RSNA boxes
# ------------------------------------------------------------

def denormalized_gray(x: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    image = (x.cpu() * std + mean).clamp(0, 1)
    return image.mean(dim=0).numpy()


def load_rsna_boxes(labels_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)

    required = {"patientId", "Target", "x", "y", "width", "height"}
    if not required.issubset(df.columns):
        print(
            "[XAI WARN] RSNA CSV has no complete bounding-box columns. "
            "Localization stage will be unavailable."
        )
        return pd.DataFrame(
            columns=["id", "x", "y", "width", "height"]
        )

    b = df[
        (df["Target"] == 1)
        & df["x"].notna()
        & df["y"].notna()
        & df["width"].notna()
        & df["height"].notna()
    ][["patientId", "x", "y", "width", "height"]].copy()

    return b.rename(columns={"patientId": "id"})


def original_hw(path: str | Path):
    path = str(path)

    if path.lower().endswith(".dcm"):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        rows = int(getattr(ds, "Rows"))
        cols = int(getattr(ds, "Columns"))
        return rows, cols

    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    return h, w


def scaled_boxes_for_image(
    case_id: str,
    path: str | Path,
    boxes_df: pd.DataFrame,
    image_size: int,
):
    rows = boxes_df[boxes_df["id"].astype(str) == str(case_id)]

    if rows.empty:
        return []

    h0, w0 = original_hw(path)
    sx = image_size / float(w0)
    sy = image_size / float(h0)

    boxes = []

    for _, r in rows.iterrows():
        x0 = max(0.0, float(r["x"]) * sx)
        y0 = max(0.0, float(r["y"]) * sy)
        w = max(0.0, float(r["width"]) * sx)
        h = max(0.0, float(r["height"]) * sy)

        x1 = min(float(image_size), x0 + w)
        y1 = min(float(image_size), y0 + h)

        boxes.append((x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)))

    return boxes


def boxes_to_mask(boxes, image_size: int):
    mask = np.zeros((image_size, image_size), dtype=bool)

    for x, y, w, h in boxes:
        x0 = max(0, int(math.floor(x)))
        y0 = max(0, int(math.floor(y)))
        x1 = min(image_size, int(math.ceil(x + w)))
        y1 = min(image_size, int(math.ceil(y + h)))

        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True

    return mask


# ------------------------------------------------------------
# Confusion-group qualitative selection
# ------------------------------------------------------------

def add_confusion_fields(predictions: pd.DataFrame):
    x = predictions.copy()

    x["trace_pred"] = (
        x["gated_calibrated"] >= x["threshold"]
    ).astype(int)

    x["trace_class_confidence"] = np.where(
        x["trace_pred"] == 1,
        x["gated_calibrated"],
        1.0 - x["gated_calibrated"],
    )

    conditions = [
        (x["label"] == 1) & (x["trace_pred"] == 1),
        (x["label"] == 0) & (x["trace_pred"] == 0),
        (x["label"] == 0) & (x["trace_pred"] == 1),
        (x["label"] == 1) & (x["trace_pred"] == 0),
    ]

    x["confusion_group"] = np.select(
        conditions,
        ["TP", "TN", "FP", "FN"],
        default="UNKNOWN",
    )

    x["uncertainty_score"] = x["entropy"] + x["disagreement"]

    return x


def select_confusion_cases(predictions: pd.DataFrame, cases_per_group: int):
    x = add_confusion_fields(predictions)

    selections = []

    rules = {
        "TP": ("trace_class_confidence", False, "TP_high_confidence"),
        "TN": ("trace_class_confidence", False, "TN_high_confidence"),
        "FP": ("uncertainty_score", False, "FP_high_uncertainty"),
        "FN": ("uncertainty_score", False, "FN_high_uncertainty"),
    }

    for group, (sort_col, ascending, output_group) in rules.items():
        z = (
            x[x["confusion_group"] == group]
            .sort_values([sort_col, "id"], ascending=[ascending, True])
            .head(cases_per_group)
            .copy()
        )

        if z.empty:
            print(f"[XAI WARN] no cases available for {group}")
            continue

        z["xai_group"] = output_group
        selections.append(z)

    if not selections:
        return pd.DataFrame()

    return pd.concat(selections, ignore_index=True)


# ------------------------------------------------------------
# Localization metrics
# ------------------------------------------------------------

def localization_metrics(attr, gt_mask, top_fraction: float, concentration_fraction: float):
    a = normalize_map(attr)

    if not np.any(gt_mask):
        return {
            "pointing_hit": np.nan,
            "energy_in_box": np.nan,
            "iou": np.nan,
            "dice": np.nan,
            "attribution_entropy": normalized_attribution_entropy(a),
            "attribution_concentration": attribution_concentration(
                a, concentration_fraction
            ),
        }

    peak = np.unravel_index(np.argmax(a), a.shape)
    pointing_hit = float(bool(gt_mask[peak]))

    total = float(a.sum())
    energy_in_box = float(a[gt_mask].sum() / total) if total > 0 else 0.0

    pred_mask = top_fraction_mask(a, top_fraction)
    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    union = int(np.logical_or(pred_mask, gt_mask).sum())

    iou = float(inter / union) if union > 0 else 0.0

    denom = int(pred_mask.sum()) + int(gt_mask.sum())
    dice = float((2 * inter) / denom) if denom > 0 else 0.0

    return {
        "pointing_hit": pointing_hit,
        "energy_in_box": energy_in_box,
        "iou": iou,
        "dice": dice,
        "attribution_entropy": normalized_attribution_entropy(a),
        "attribution_concentration": attribution_concentration(
            a, concentration_fraction
        ),
    }


# ------------------------------------------------------------
# Faithfulness / deletion
# ------------------------------------------------------------

def masked_input_from_map(x: torch.Tensor, attr, fraction: float, rng=None):
    arr = np.asarray(attr)

    if rng is None:
        mask = top_fraction_mask(arr, fraction)
    else:
        n = arr.size
        k = max(1, min(n, int(round(float(fraction) * n))))
        ids = rng.choice(n, size=k, replace=False)
        mask_flat = np.zeros(n, dtype=bool)
        mask_flat[ids] = True
        mask = mask_flat.reshape(arr.shape)

    masked = x.clone()
    m = torch.tensor(mask, dtype=torch.bool)

    # Zero in normalized input space = ImageNet mean baseline.
    masked[:, m] = 0.0

    return masked


def faithfulness_case(
    model,
    x: torch.Tensor,
    attr,
    target_class: int,
    fractions,
    random_repeats: int,
    rng_seed: int,
):
    fractions = [float(f) for f in fractions]

    base_conf = target_class_confidence(model, x, target_class)

    top_conf = []
    random_conf_mean = []

    for fraction in fractions:
        if fraction <= 0:
            top_conf.append(base_conf)
            random_conf_mean.append(base_conf)
            continue

        xm = masked_input_from_map(x, attr, fraction, rng=None)
        top_conf.append(target_class_confidence(model, xm, target_class))

        random_confs = []
        for repeat in range(int(random_repeats)):
            rng = np.random.default_rng(rng_seed + repeat * 1009 + int(fraction * 10000))
            xr = masked_input_from_map(x, attr, fraction, rng=rng)
            random_confs.append(
                target_class_confidence(model, xr, target_class)
            )

        random_conf_mean.append(float(np.mean(random_confs)))

    x_axis = np.asarray(fractions, dtype=float)
    top_conf = np.asarray(top_conf, dtype=float)
    random_conf_mean = np.asarray(random_conf_mean, dtype=float)

    span = float(x_axis.max() - x_axis.min())
    if span <= 0:
        deletion_auc = float(top_conf.mean())
        random_deletion_auc = float(random_conf_mean.mean())
    else:
        deletion_auc = float(np.trapezoid(top_conf, x_axis) / span)
        random_deletion_auc = float(np.trapezoid(random_conf_mean, x_axis) / span)

    nonzero = x_axis > 0

    aopc = float(np.mean(base_conf - top_conf[nonzero]))
    random_aopc = float(np.mean(base_conf - random_conf_mean[nonzero]))

    result = {
        "base_target_confidence": float(base_conf),
        "deletion_auc": deletion_auc,
        "random_deletion_auc": random_deletion_auc,
        "aopc": aopc,
        "random_aopc": random_aopc,
        "aopc_gain_over_random": float(aopc - random_aopc),
    }

    for fraction, tc, rc in zip(x_axis, top_conf, random_conf_mean):
        if fraction <= 0:
            continue

        tag = f"{int(round(fraction * 100)):02d}"

        result[f"top_drop_{tag}pct"] = float(base_conf - tc)
        result[f"random_drop_{tag}pct"] = float(base_conf - rc)

    return result


# ------------------------------------------------------------
# Bootstrap summary
# ------------------------------------------------------------

def bootstrap_mean_ci(values, n_boot: int, seed: int):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]

    if len(v) == 0:
        return np.nan, np.nan, np.nan

    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])

    rng = np.random.default_rng(seed)
    means = []

    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(v), len(v))
        means.append(float(v[idx].mean()))

    means = np.asarray(means)

    return (
        float(v.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def summarize_metric_table(
    df: pd.DataFrame,
    group_cols,
    metric_cols,
    n_boot: int,
    seed: int,
):
    rows = []

    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(group_cols, keys))

        for metric in metric_cols:
            if metric not in group.columns:
                continue

            mean, lo, hi = bootstrap_mean_ci(
                group[metric].to_numpy(),
                n_boot,
                stable_seed(*keys, metric, base_seed=seed),
            )

            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n": int(group[metric].notna().sum()),
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "median": float(group[metric].median()),
                    "std": float(group[metric].std(ddof=1))
                    if group[metric].notna().sum() > 1
                    else 0.0,
                }
            )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Figure output
# ------------------------------------------------------------

XAI_VERSION = "trace_xai_v5_display_only_robust"
MANIFEST_NAME = "TRACE_CXR_XAI_MANIFEST_V5.csv"
OLD_MANIFEST_NAME = "TRACE_CXR_XAI_MANIFEST.csv"


def _draw_boxes(ax, boxes):
    for x, y, w, h in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="cyan", linewidth=1.3))


def fused_attribution_map(inception_cam, swin_ig, alpha_inception: float):
    """Descriptive mixture of same-class, per-method normalized attribution.

    IMPORTANT: not a derivative, causal attribution, or post-calibration XAI of
    the full gate. It is a qualitative visualization only.
    """
    alpha_inception = float(np.clip(alpha_inception, 0.0, 1.0))
    inc = normalize_map(inception_cam)
    swin = normalize_map(swin_ig)
    # No second normalization: this keeps the mixture in [0,1] and retains
    # dampening when evidence locations disagree.
    return (alpha_inception * inc + (1.0 - alpha_inception) * swin).astype(np.float32)


def difference_map(inception_cam, swin_ig):
    return (normalize_map(inception_cam) - normalize_map(swin_ig)).astype(np.float32)


def display_only_attribution(attr, percentile=99.5, sigma_px=2.0):
    """Visual rendering only. Never feed this map to localization/faithfulness.

    Same smoothing and percentile policy applies to BOTH branches; original
    method-normalized maps remain unchanged. No heat is introduced for zeros.
    """
    arr = np.asarray(attr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    if arr.ndim != 2:
        raise ValueError(f"Expected HxW attribution, got {arr.shape}")
    if not np.any(arr > 0):
        return np.zeros(arr.shape, dtype=np.float32)
    if float(sigma_px) > 0:
        arr = gaussian_filter(arr, sigma=float(sigma_px), mode="reflect")
    scale = float(np.percentile(arr, float(percentile)))
    if scale <= np.finfo(np.float64).eps * float(arr.max()):
        scale = float(arr.max())
    if scale <= 0:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip(arr / scale, 0.0, 1.0).astype(np.float32)


def qualitative_ig_diagnostics(raw_ig):
    """Report whether the actual Swin IG exists before any display scaling."""
    a = np.asarray(raw_ig, dtype=np.float64)
    finite = np.isfinite(a)
    good = np.where(finite, np.maximum(a, 0.0), 0.0)
    total = float(good.sum())
    positive = good[good > 0]
    if not finite.any() or total <= 0.0:
        state = "ZERO_OR_INVALID_RAW_IG"
    elif np.isclose(float(good.max()), 0.0, atol=0.0):
        state = "NUMERICALLY_TINY_RAW_IG"
    else:
        state = "NONZERO_RAW_IG"
    n_top = max(1, int(math.ceil(good.size * 0.01)))
    top_mass = (
        float(np.partition(good.reshape(-1), -n_top)[-n_top:].sum() / total)
        if total > 0 else float("nan")
    )
    return {
        "raw_ig_status": state,
        "raw_ig_nonfinite_pixels": int((~finite).sum()),
        "raw_ig_nonzero_fraction": float((good > 0).mean()),
        "raw_ig_max": float(good.max()),
        "raw_ig_positive_p50": float(np.percentile(positive, 50)) if len(positive) else 0.0,
        "raw_ig_positive_p99": float(np.percentile(positive, 99)) if len(positive) else 0.0,
        "raw_ig_top1pct_energy": top_mass,
    }


def save_swin_diagnostic(gray, raw_normalized, display_swin, boxes, title, path):
    """Expose unmodified normalized IG alongside the enhanced DISPLAY map."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.7))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=1)
    _draw_boxes(axes[0], boxes)
    axes[0].set_title("Original CXR")
    axes[1].imshow(raw_normalized, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Swin IG · raw normalized (NO overlay)")
    axes[2].imshow(gray, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(display_swin, cmap="inferno", vmin=0, vmax=1,
                   alpha=0.9*np.power(display_swin, 0.65))
    _draw_boxes(axes[2], boxes)
    axes[2].set_title("Swin IG · display enhanced")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.text(0.5, 0.025,
             "Enhancement is for visualization only; not a new attribution or improved localization.",
             ha="center", fontsize=8)
    fig.subplots_adjust(top=0.83, bottom=0.14, wspace=0.07)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _overlay(ax, gray, importance, overlay_alpha, gamma, low_cutoff=0.035):
    ax.imshow(gray, cmap="gray", vmin=0.0, vmax=1.0)
    a = np.clip(np.asarray(importance, dtype=np.float32), 0.0, 1.0)
    # Dim low-importance background rather than coating the entire CXR.
    visibility = np.clip((a-float(low_cutoff))/(1.0-float(low_cutoff)), 0.0, 1.0)
    ax.imshow(a, cmap="inferno", vmin=0.0, vmax=1.0,
              alpha=float(overlay_alpha) * np.power(visibility, float(gamma)))


def save_case_figure(gray, inception_cam, swin_ig, fused_map, boxes,
                     title_line1: str, title_line2: str, path: Path,
                     overlay_alpha: float = 0.86, overlay_gamma: float = 0.65,
                     low_cutoff: float = 0.035):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15.8, 5.1))
    axes[0].imshow(gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Before · Original CXR", fontsize=10)
    panels = [
        (axes[1], inception_cam, "After InceptionV3 · Grad-CAM"),
        (axes[2], swin_ig, "After Swin-B · IG"),
        (axes[3], fused_map, "After combined · weighted map*"),
    ]
    for ax, attr, label in panels:
        _overlay(ax, gray, attr, overlay_alpha, overlay_gamma, low_cutoff)
        ax.set_title(label, fontsize=10)
    for ax in axes:
        _draw_boxes(ax, boxes)
        ax.axis("off")
    fig.suptitle(title_line1, fontsize=11, y=0.985)
    wrapped_meta = textwrap.fill(title_line2, width=125, break_long_words=False)
    fig.text(0.5, 0.925, wrapped_meta, ha="center", va="top", fontsize=9)
    fig.subplots_adjust(top=0.80, bottom=0.32, left=0.015, right=0.99, wspace=0.05)
    # Shared colorbar is a relative display scale only. The different XAI
    # methods do NOT have a directly comparable attribution magnitude.
    cax = fig.add_axes([0.345, 0.20, 0.31, 0.025])
    cbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap="inferno"),
        cax=cax, orientation="horizontal",
    )
    cbar.set_label("Per-method display-only intensity (0–1; robust scaled)", fontsize=9)
    fig.text(0.5, 0.045,
             "*Display-only smoothing/percentile scaling; raw maps are archived. "
             "Combined map is descriptive, NOT full-fusion causal attribution.",
             ha="center", fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_supplementary_case_figure(gray, swin_attention, diff_map, boxes,
                                   title_line1: str, title_line2: str, path: Path,
                                   overlay_alpha: float = 0.70,
                                   overlay_gamma: float = 1.50):
    if swin_attention is None and diff_map is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(swin_attention is not None) + int(diff_map is not None)
    fig, axes = plt.subplots(1, n, figsize=(5.0*n, 4.0), squeeze=False)
    axs = list(axes[0]); i = 0
    if swin_attention is not None:
        _overlay(axs[i], gray, swin_attention, overlay_alpha, overlay_gamma)
        axs[i].set_title("Swin-B attention · structural diagnostic (not XAI)")
        i += 1
    if diff_map is not None:
        axs[i].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axs[i].imshow(diff_map, cmap="RdBu_r", vmin=-1, vmax=1,
                      alpha=0.65 * np.power(np.abs(diff_map), 1.3))
        axs[i].set_title("Difference · Inc minus Swin (same class)")
        cbar = fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap="RdBu_r"),
                            ax=axs[i], orientation="horizontal", fraction=0.065, pad=0.07)
        cbar.set_label("Normalized attribution difference; red = Inception", fontsize=8)
    for ax in axs:
        _draw_boxes(ax, boxes)
        ax.axis("off")
    fig.suptitle(title_line1 + "\n" + title_line2, fontsize=9)
    fig.subplots_adjust(top=0.78, bottom=0.15)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def qualitative_signature(cfg, out):
    """Invalidate qualitative cache when config, predictions, lock or weights change."""
    xcfg = _cfg_xai(cfg)
    keys = ("cases_per_confusion_group", "qualitative_ig_steps",
            "qualitative_overlay_alpha", "qualitative_overlay_gamma",
            "display_percentile", "display_smoothing_sigma_px",
            "display_low_cutoff", "export_swin_diagnostic", "export_raw_qualitative_maps",
            "export_difference_map", "export_supplementary_attention")
    paths = [
        out / "TRACE_CXR_DEPLOYMENT_LOCK.json",
        out / "TRACE_CXR_INTERNAL_PREDICTIONS.csv",
        out / "TRACE_CXR_KERMANY_PREDICTIONS.csv",
    ]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    # Full checkpoint hashing would be costly; size/mtime flags replacement.
    ckpts = [out / "TRACE_CXR_FINAL_INCEPTIONV3.pt",
             out / "TRACE_CXR_FINAL_SWINB.pt"]
    for p in ckpts:
        if not p.exists():
            raise FileNotFoundError(p)
    payload = {
        "version": XAI_VERSION,
        "source_sha256": file_digest(Path(__file__).resolve()),
        "options": {k: xcfg[k] for k in keys},
        "inputs_sha256": {p.name: file_digest(p) for p in paths},
        "checkpoints": {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                        for p in ckpts},
        "rsna_labels": str(cfg["paths"]["rsna_labels_csv"]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

# ------------------------------------------------------------
# Common cohort loading
# ------------------------------------------------------------

def cohort_sources(cfg: dict, out: Path):
    rsna_manifest = pd.read_csv(
        out / "TRACE_CXR_RSNA_SPLIT_MANIFEST.csv"
    )

    rsna_paths = rsna_manifest[
        rsna_manifest["split"] == "locked_test"
    ][["id", "path", "label"]].copy()

    kermany_paths = kermany_index(
        cfg["paths"]["kermany_root"]
    )[["id", "path", "label"]].copy()

    return [
        (
            "RSNA",
            pd.read_csv(out / "TRACE_CXR_INTERNAL_PREDICTIONS.csv"),
            rsna_paths,
        ),
        (
            "Kermany",
            pd.read_csv(out / "TRACE_CXR_KERMANY_PREDICTIONS.csv"),
            kermany_paths,
        ),
    ]


def make_input(sample_row, cfg: dict, model_name: str):
    sample_df = pd.DataFrame(
        [
            {
                "id": sample_row["id"],
                "path": sample_row["path"],
                "label": int(sample_row["label"]),
                "source": sample_row.get("source", "XAI"),
            }
        ]
    )

    return CXRDataset(
        sample_df,
        cfg["experiment"]["image_size"],
        model_name,
        train=False,
    )[0][0]


# ------------------------------------------------------------
# Stage 1: qualitative
# ------------------------------------------------------------

def run_qualitative(cfg: dict, force: bool = False):
    """V5 qualitative output is separate from V4; preserves all earlier CSVs."""
    out = get_out(cfg)
    xcfg = _cfg_xai(cfg)
    validate_visual_config(xcfg)
    manifest_file = out / MANIFEST_NAME
    signature = qualitative_signature(cfg, out)

    existing = pd.DataFrame()
    if manifest_file.exists():
        try:
            candidate = pd.read_csv(manifest_file)
        except Exception as exc:
            raise RuntimeError(f"Cannot read existing XAI manifest: {exc}") from exc
        if "xai_version" not in candidate.columns or not candidate["xai_version"].eq(XAI_VERSION).all():
            raise RuntimeError("Unexpected XAI version in V5-only manifest; inspect file manually.")
        if not force:
            if not candidate.empty and (candidate["signature"] != signature).any():
                raise RuntimeError(
                    "Qualitative config/inputs/checkpoint/code changed since V5 run. "
                    "Use 'python trace_cxr_xai.py qualitative --force' to "
                    "rebuild V5 figures; frozen models will NOT be retrained."
                )
            existing = candidate
        else:
            print("[XAI] --force: rebuilding V5 qualitative figures and manifest.")

    # Never carry over V4 or earlier rows into the V5-only manifest.
    rows = [] if existing.empty else existing.to_dict("records")
    completed = {(str(r["cohort"]), str(r["id"]), str(r["group"])): r
                 for r in rows}

    lock = load_lock(out)
    inception = load_final_model("inception_v3", cfg, out)
    swin = load_final_model("swin_b", cfg, out)
    cam = InceptionGradCAM(inception)
    boxes_df = load_rsna_boxes(cfg["paths"]["rsna_labels_csv"])
    xai_root = out / "xai_v5"
    expected = set()

    try:
        for cohort, pred, path_table in cohort_sources(cfg, out):
            merged = pred.merge(path_table[["id", "path"]], on="id", how="left", validate="one_to_one")
            selected = select_confusion_cases(merged, int(xcfg["cases_per_confusion_group"]))
            if selected.empty:
                continue
            for _, row in selected.iterrows():
                key = (cohort, str(row["id"]), str(row["xai_group"]))
                expected.add(key)
                image_path = (
                    xai_root / cohort / str(row["xai_group"])
                    / f"TRACE_CXR_V5_{cohort}_{row['xai_group']}_{row['id']}.png"
                )
                cached = completed.get(key)
                if cached is not None and image_path.exists() and not force:
                    required_cached = ["supplementary_file", "swin_diagnostic_file", "raw_maps_file"]
                    all_present = True
                    for field in required_cached:
                        stored = cached.get(field, "")
                        if isinstance(stored, str) and stored and not (out / stored).exists():
                            all_present = False
                            break
                    if all_present:
                        print("[XAI SKIP]", key)
                        continue

                x = make_input(row, cfg, "inception_v3")
                trace_target = int(
                    float(row["gated_calibrated"]) >= float(row["threshold"])
                )
                inc_branch_pred = int(
                    float(row["inception_v3_calibrated"]) >= branch_threshold(lock, "inception_v3")
                )
                swin_branch_pred = int(
                    float(row["swin_b_calibrated"]) >= branch_threshold(lock, "swin_b")
                )
                # Key correction: both maps explain THE SAME TARGET CLASS.
                inc_cam = cam(x, target_class=trace_target)
                swin_ig, raw_swin_ig = integrated_gradients(
                    swin, x, target_class=trace_target,
                    steps=int(xcfg["qualitative_ig_steps"]),
                    return_raw=True,
                )
                ig_diag = qualitative_ig_diagnostics(raw_swin_ig)
                if ig_diag["raw_ig_status"] != "NONZERO_RAW_IG":
                    print(f"[XAI WARN] {key}: {ig_diag['raw_ig_status']}; no heat will be fabricated")
                alpha_inc = float(row["gate_alpha_inception"])
                # Original method-normalized maps are untouched and archived.
                raw_fused_map = fused_attribution_map(inc_cam, swin_ig, alpha_inc)
                raw_diff_map = difference_map(inc_cam, swin_ig)
                # Identical visualization-only spatial policy for both methods.
                display_inc = display_only_attribution(
                    inc_cam, xcfg["display_percentile"], xcfg["display_smoothing_sigma_px"]
                )
                display_swin = display_only_attribution(
                    swin_ig, xcfg["display_percentile"], xcfg["display_smoothing_sigma_px"]
                )
                display_fused = fused_attribution_map(display_inc, display_swin, alpha_inc)
                display_diff = difference_map(display_inc, display_swin)
                gray = denormalized_gray(x)

                boxes = []
                if cohort == "RSNA" and int(row["label"]) == 1 and not boxes_df.empty:
                    boxes = scaled_boxes_for_image(
                        str(row["id"]), row["path"], boxes_df,
                        int(cfg["experiment"]["image_size"]),
                    )
                title1 = (
                    f"{TRACE_NAME} | {cohort} | {row['xai_group']} | "
                    f"GT={class_name(int(row['label']))} | "
                    f"TRACE Pred={class_name(trace_target)}"
                )
                title2 = (
                    f"pInc={float(row['inception_v3_calibrated']):.3f} "
                    f"(pred={class_name(inc_branch_pred)}) | "
                    f"pSwin={float(row['swin_b_calibrated']):.3f} "
                    f"(pred={class_name(swin_branch_pred)}) | "
                    f"target=BOTH→{class_name(trace_target)} | "
                    f"αInc={alpha_inc:.3f} | αSwin={1.0-alpha_inc:.3f} | "
                    f"pTRACE={float(row['gated_calibrated']):.3f} | "
                    f"thr={float(row['threshold']):.3f} | "
                    f"H={float(row['entropy']):.3f} | "
                    f"D={float(row['disagreement']):.3f}"
                )
                save_case_figure(
                    gray, display_inc, display_swin, display_fused, boxes,
                    title1, title2, image_path,
                    overlay_alpha=float(xcfg["qualitative_overlay_alpha"]),
                    overlay_gamma=float(xcfg["qualitative_overlay_gamma"]),
                    low_cutoff=float(xcfg["display_low_cutoff"]),
                )
                diagnostic_file = None
                if bool(xcfg["export_swin_diagnostic"]):
                    diagnostic_file = image_path.with_name(image_path.stem + "_swin_diagnostic.png")
                    save_swin_diagnostic(
                        gray, swin_ig, display_swin, boxes,
                        f"{TRACE_NAME} | {cohort} | {row['id']} | {ig_diag['raw_ig_status']}",
                        diagnostic_file,
                    )
                raw_maps_file = None
                if bool(xcfg["export_raw_qualitative_maps"]):
                    raw_maps_file = image_path.with_name(image_path.stem + "_raw_maps.npz")
                    temp_maps = raw_maps_file.with_name(raw_maps_file.stem + ".tmp.npz")
                    np.savez_compressed(
                        temp_maps,
                        inception_gradcam_raw_normalized=inc_cam.astype(np.float32),
                        swin_ig_absolute_raw=raw_swin_ig.astype(np.float32),
                        swin_ig_raw_normalized=swin_ig.astype(np.float32),
                        combined_original_normalized_maps=raw_fused_map.astype(np.float32),
                        raw_difference_map=raw_diff_map.astype(np.float32),
                        inception_display_only=display_inc.astype(np.float32),
                        swin_display_only=display_swin.astype(np.float32),
                        combined_display_only=display_fused.astype(np.float32),
                        gate_alpha_inception=np.array(alpha_inc, dtype=np.float32),
                        trace_target=np.array(trace_target, dtype=np.int64),
                    )
                    temp_maps.replace(raw_maps_file)
                supplemental = None
                attn = None
                if bool(xcfg["export_supplementary_attention"]):
                    try:
                        attn = swin_attention_aggregate(swin, x)
                    except Exception as exc:
                        print(f"[XAI WARN] Optional Swin attention unavailable for {key}: {exc}")
                if attn is not None or bool(xcfg["export_difference_map"]):
                    supplemental = image_path.with_name(image_path.stem + "_supp.png")
                    save_supplementary_case_figure(
                        gray, attn,
                        display_diff if bool(xcfg["export_difference_map"]) else None,
                        boxes, title1, title2, supplemental,
                        overlay_alpha=float(xcfg["qualitative_overlay_alpha"]),
                        overlay_gamma=float(xcfg["qualitative_overlay_gamma"]),
                    )
                record = {
                    "xai_version": XAI_VERSION,
                    "signature": signature,
                    "framework": TRACE_NAME,
                    "cohort": cohort,
                    "id": row["id"],
                    "group": row["xai_group"],
                    "confusion_group": row["confusion_group"],
                    "label": int(row["label"]),
                    "trace_prediction": trace_target,
                    "shared_xai_target": trace_target,
                    "trace_probability": float(row["gated_calibrated"]),
                    "trace_threshold": float(row["threshold"]),
                    "inception_probability": float(row["inception_v3_calibrated"]),
                    "inception_prediction": inc_branch_pred,
                    "inception_threshold": branch_threshold(lock, "inception_v3"),
                    "inception_xai_target": trace_target,
                    "swin_probability": float(row["swin_b_calibrated"]),
                    "swin_prediction": swin_branch_pred,
                    "swin_threshold": branch_threshold(lock, "swin_b"),
                    "swin_xai_target": trace_target,
                    "gate_alpha_inception": alpha_inc,
                    "gate_alpha_swin": 1.0-alpha_inc,
                    "entropy": float(row["entropy"]),
                    "disagreement": float(row["disagreement"]),
                    "has_rsna_bbox": bool(boxes),
                    "image_file": str(image_path.relative_to(out)),
                    "supplementary_file": str(supplemental.relative_to(out)) if supplemental else "",
                    "swin_diagnostic_file": str(diagnostic_file.relative_to(out)) if diagnostic_file else "",
                    "raw_maps_file": str(raw_maps_file.relative_to(out)) if raw_maps_file else "",
                    "display_percentile": float(xcfg["display_percentile"]),
                    "display_smoothing_sigma_px": float(xcfg["display_smoothing_sigma_px"]),
                    "display_low_cutoff": float(xcfg["display_low_cutoff"]),
                    **ig_diag,
                    "panel_1": "Before / Original CXR",
                    "panel_2": "After InceptionV3 / shared-target Grad-CAM",
                    "panel_3": "After Swin-B / shared-target Integrated Gradients",
                    "panel_4": "After Combined / descriptive gate-weighted branch maps",
                    "fusion_xai_formula": "alpha*display_only(Grad-CAM)+(1-alpha)*display_only(IG); NO post-mixture renorm",
                    "scope_note": (
                        "All XAI maps target the TRACE-predicted class. Display-only "
                        "smoothing/percentile normalization is applied identically to both "
                        "branches BEFORE visual combination; original numerical maps are "
                        "saved separately. The combined panel is descriptive, NOT a "
                        "full-fusion causal attribution or proof of improved localization."
                    ),
                }
                completed[key] = record
                rows = [r for r in rows
                        if (str(r["cohort"]), str(r["id"]), str(r["group"])) != key]
                rows.append(record)
                # Atomic save: server termination should not partially corrupt CSV.
                temp = manifest_file.with_suffix(".csv.tmp")
                pd.DataFrame(rows).to_csv(temp, index=False)
                temp.replace(manifest_file)
                print("[XAI DONE]", key)
    finally:
        cam.close()

    # Trim stale V5 entries if the deterministic selected case list changed.
    rows = [r for r in rows
            if (str(r["cohort"]), str(r["id"]), str(r["group"])) in expected]
    if rows:
        df = pd.DataFrame(rows).sort_values(["cohort", "group", "id"])
        if df.duplicated(subset=["cohort", "id", "group"]).any():
            raise RuntimeError("Duplicate XAI case in clean V5 manifest.")
        if not (df["inception_xai_target"] == df["swin_xai_target"]).all():
            raise RuntimeError("Shared-target XAI invariant violated.")
        df.to_csv(manifest_file, index=False)
    else:
        pd.DataFrame(columns=["xai_version", "cohort", "id", "group", "image_file"]).to_csv(
            manifest_file, index=False
        )
    write_json(out / "TRACE_CXR_XAI_V5_DISPLAY_METHOD.json", {
        "version": XAI_VERSION,
        "manifest": MANIFEST_NAME,
        "folder": "xai_v5",
        "same_target_class": "TRACE-CXR predicted class for both branches",
        "display_only_policy": {
            "percentile": float(xcfg["display_percentile"]),
            "gaussian_sigma_pixels": float(xcfg["display_smoothing_sigma_px"]),
            "overlay_alpha": float(xcfg["qualitative_overlay_alpha"]),
            "overlay_gamma": float(xcfg["qualitative_overlay_gamma"]),
            "low_cutoff": float(xcfg["display_low_cutoff"]),
            "same_transform_both_branches": True,
            "combined_panel": "alpha*display(Inception Grad-CAM)+(1-alpha)*display(Swin IG)",
            "combined_panel_not_full_fusion_attribution": True,
            "raw_original_maps_saved_per_case": bool(xcfg["export_raw_qualitative_maps"]),
            "swin_raw_vs_display_diagnostic_per_case": bool(xcfg["export_swin_diagnostic"]),
        },
        "quantitative_xai_unchanged": True,
        "prior_v4_manifest_preserved": True,
    })
    print(f"[DONE] qualitative V5: {len(rows)} cases; manifest contains V5 only")

# ------------------------------------------------------------
# Stage 2: RSNA quantitative localization
# ------------------------------------------------------------

def _deterministic_subset(df: pd.DataFrame, max_cases: int, seed: int):
    if max_cases is None or int(max_cases) <= 0 or len(df) <= int(max_cases):
        return df.sort_values("id").reset_index(drop=True)

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(df), size=int(max_cases), replace=False))
    return df.iloc[idx].sort_values("id").reset_index(drop=True)


def run_localization(cfg: dict, force: bool = False):
    out = get_out(cfg)
    xcfg = _cfg_xai(cfg)

    result_file = out / "TRACE_CXR_XAI_LOCALIZATION_RSNA.csv"
    summary_file = out / "TRACE_CXR_XAI_LOCALIZATION_SUMMARY.csv"

    if force:
        for p in (result_file, summary_file):
            if p.exists():
                p.unlink()

    boxes_df = load_rsna_boxes(cfg["paths"]["rsna_labels_csv"])

    if boxes_df.empty:
        print("[XAI] Localization skipped: RSNA bbox annotations unavailable.")
        return

    pred = pd.read_csv(out / "TRACE_CXR_INTERNAL_PREDICTIONS.csv")
    manifest = pd.read_csv(out / "TRACE_CXR_RSNA_SPLIT_MANIFEST.csv")

    paths = manifest[
        manifest["split"] == "locked_test"
    ][["id", "path", "label"]]

    cases = pred.merge(paths[["id", "path"]], on="id", how="left")

    bbox_ids = set(boxes_df["id"].astype(str))
    cases = cases[
        (cases["label"] == 1)
        & cases["id"].astype(str).isin(bbox_ids)
    ].copy()

    cases = _deterministic_subset(
        cases,
        int(xcfg["rsna_localization_max_cases"]),
        stable_seed("localization", base_seed=cfg["experiment"]["seed"]),
    )

    print(
        f"[XAI] RSNA localization cases: {len(cases)} "
        f"(max={xcfg['rsna_localization_max_cases']}; 0 means all)"
    )

    inception = load_final_model("inception_v3", cfg, out)
    swin = load_final_model("swin_b", cfg, out)
    cam = InceptionGradCAM(inception)

    existing = pd.DataFrame()
    if result_file.exists():
        try:
            existing = pd.read_csv(result_file)
        except Exception:
            existing = pd.DataFrame()

    rows = [] if existing.empty else existing.to_dict("records")
    completed = set()

    if not existing.empty:
        for _, r in existing.iterrows():
            completed.add((str(r["id"]), str(r["branch"])))

    image_size = int(cfg["experiment"]["image_size"])
    top_fraction = float(xcfg["attribution_top_fraction"])
    concentration_fraction = float(xcfg["concentration_top_fraction"])

    try:
        for _, row in cases.iterrows():
            x = make_input(row, cfg, "inception_v3")

            boxes = scaled_boxes_for_image(
                str(row["id"]),
                row["path"],
                boxes_df,
                image_size,
            )

            gt_mask = boxes_to_mask(boxes, image_size)

            branch_maps = {}

            if (str(row["id"]), "inception_v3") not in completed:
                branch_maps["inception_v3"] = cam(x, target_class=1)

            if (str(row["id"]), "swin_b") not in completed:
                branch_maps["swin_b"] = integrated_gradients(
                    swin,
                    x,
                    target_class=1,
                    steps=int(xcfg["quantitative_ig_steps"]),
                )

            for branch, attr in branch_maps.items():
                m = localization_metrics(
                    attr,
                    gt_mask,
                    top_fraction,
                    concentration_fraction,
                )

                record = {
                    "framework": TRACE_NAME,
                    "cohort": "RSNA_locked_test",
                    "id": row["id"],
                    "branch": branch,
                    "target_class": 1,
                    "target_class_name": "PNEUMONIA",
                    "n_boxes": len(boxes),
                    "attribution_top_fraction": top_fraction,
                    "concentration_top_fraction": concentration_fraction,
                    **m,
                }

                rows.append(record)
                completed.add((str(row["id"]), branch))
                pd.DataFrame(rows).to_csv(result_file, index=False)

                print(
                    f"[LOC] {row['id']} {branch} "
                    f"hit={m['pointing_hit']:.0f} "
                    f"energy={m['energy_in_box']:.3f} "
                    f"IoU={m['iou']:.3f}"
                )

    finally:
        cam.close()

    results = pd.DataFrame(rows)

    metrics = [
        "pointing_hit",
        "energy_in_box",
        "iou",
        "dice",
        "attribution_entropy",
        "attribution_concentration",
    ]

    summary = summarize_metric_table(
        results,
        group_cols=["branch"],
        metric_cols=metrics,
        n_boot=int(xcfg["bootstrap_iterations"]),
        seed=int(cfg["experiment"]["seed"]),
    )

    summary.to_csv(summary_file, index=False)

    print("[DONE] RSNA quantitative localization")
    print(summary.to_string(index=False))


# ------------------------------------------------------------
# Stage 3: faithfulness
# ------------------------------------------------------------

def balanced_faithfulness_subset(
    predictions: pd.DataFrame,
    max_cases: int,
    seed: int,
):
    x = add_confusion_fields(predictions)

    if max_cases is None or int(max_cases) <= 0 or len(x) <= int(max_cases):
        return x.sort_values(["confusion_group", "id"]).reset_index(drop=True)

    groups = ["TP", "TN", "FP", "FN"]
    target_total = int(max_cases)
    base = target_total // len(groups)
    remainder = target_total % len(groups)

    chosen = []

    for gi, group in enumerate(groups):
        z = x[x["confusion_group"] == group].copy()
        n_take = base + (1 if gi < remainder else 0)
        n_take = min(n_take, len(z))

        if n_take <= 0:
            continue

        rng = np.random.default_rng(
            stable_seed("faithfulness", group, base_seed=seed)
        )
        idx = np.sort(rng.choice(len(z), size=n_take, replace=False))
        chosen.append(z.iloc[idx])

    if not chosen:
        return x.head(0)

    out = pd.concat(chosen, ignore_index=True)

    # If some error category is too small, fill remainder from unused cases.
    if len(out) < target_total:
        used = set(out["id"].astype(str))
        remainder_df = x[~x["id"].astype(str).isin(used)].copy()

        need = min(target_total - len(out), len(remainder_df))
        if need > 0:
            rng = np.random.default_rng(
                stable_seed("faithfulness_fill", base_seed=seed)
            )
            idx = np.sort(rng.choice(len(remainder_df), size=need, replace=False))
            out = pd.concat([out, remainder_df.iloc[idx]], ignore_index=True)

    return out.sort_values(["confusion_group", "id"]).reset_index(drop=True)


def run_faithfulness(cfg: dict, force: bool = False):
    out = get_out(cfg)
    xcfg = _cfg_xai(cfg)

    result_file = out / "TRACE_CXR_XAI_FAITHFULNESS.csv"
    summary_file = out / "TRACE_CXR_XAI_FAITHFULNESS_SUMMARY.csv"

    if force:
        for p in (result_file, summary_file):
            if p.exists():
                p.unlink()

    inception = load_final_model("inception_v3", cfg, out)
    swin = load_final_model("swin_b", cfg, out)
    lock = load_lock(out)
    cam = InceptionGradCAM(inception)

    existing = pd.DataFrame()
    if result_file.exists():
        try:
            existing = pd.read_csv(result_file)
        except Exception:
            existing = pd.DataFrame()

    rows = [] if existing.empty else existing.to_dict("records")
    completed = set()

    if not existing.empty:
        for _, r in existing.iterrows():
            completed.add((str(r["cohort"]), str(r["id"]), str(r["branch"])))

    fractions = list(xcfg["deletion_fractions"])
    repeats = int(xcfg["faithfulness_random_repeats"])

    try:
        for cohort, pred, path_table in cohort_sources(cfg, out):
            merged = pred.merge(
                path_table[["id", "path"]],
                on="id",
                how="left",
            )

            subset = balanced_faithfulness_subset(
                merged,
                int(xcfg["faithfulness_max_cases_per_cohort"]),
                int(cfg["experiment"]["seed"]),
            )

            print(
                f"[XAI] faithfulness {cohort}: {len(subset)} cases "
                f"(max={xcfg['faithfulness_max_cases_per_cohort']})"
            )

            for _, row in subset.iterrows():
                x = make_input(row, cfg, "inception_v3")

                inc_target = int(
                    float(row["inception_v3_calibrated"])
                    >= branch_threshold(lock, "inception_v3")
                )

                swin_target = int(
                    float(row["swin_b_calibrated"])
                    >= branch_threshold(lock, "swin_b")
                )

                if (cohort, str(row["id"]), "inception_v3") not in completed:
                    inc_attr = cam(x, inc_target)

                    fm = faithfulness_case(
                        inception,
                        x,
                        inc_attr,
                        inc_target,
                        fractions,
                        repeats,
                        stable_seed(
                            cohort,
                            row["id"],
                            "inception_v3",
                            base_seed=cfg["experiment"]["seed"],
                        ),
                    )

                    record = {
                        "framework": TRACE_NAME,
                        "cohort": cohort,
                        "id": row["id"],
                        "confusion_group": add_confusion_fields(
                            pd.DataFrame([row])
                        )["confusion_group"].iloc[0],
                        "branch": "inception_v3",
                        "target_class": inc_target,
                        "target_class_name": class_name(inc_target),
                        "xai_method": "Grad-CAM",
                        "attribution_entropy": normalized_attribution_entropy(inc_attr),
                        "attribution_concentration": attribution_concentration(
                            inc_attr,
                            float(xcfg["concentration_top_fraction"]),
                        ),
                        **fm,
                    }

                    rows.append(record)
                    completed.add((cohort, str(row["id"]), "inception_v3"))
                    pd.DataFrame(rows).to_csv(result_file, index=False)

                if (cohort, str(row["id"]), "swin_b") not in completed:
                    swin_attr = integrated_gradients(
                        swin,
                        x,
                        swin_target,
                        steps=int(xcfg["quantitative_ig_steps"]),
                    )

                    fm = faithfulness_case(
                        swin,
                        x,
                        swin_attr,
                        swin_target,
                        fractions,
                        repeats,
                        stable_seed(
                            cohort,
                            row["id"],
                            "swin_b",
                            base_seed=cfg["experiment"]["seed"],
                        ),
                    )

                    record = {
                        "framework": TRACE_NAME,
                        "cohort": cohort,
                        "id": row["id"],
                        "confusion_group": add_confusion_fields(
                            pd.DataFrame([row])
                        )["confusion_group"].iloc[0],
                        "branch": "swin_b",
                        "target_class": swin_target,
                        "target_class_name": class_name(swin_target),
                        "xai_method": "Integrated Gradients",
                        "attribution_entropy": normalized_attribution_entropy(swin_attr),
                        "attribution_concentration": attribution_concentration(
                            swin_attr,
                            float(xcfg["concentration_top_fraction"]),
                        ),
                        **fm,
                    }

                    rows.append(record)
                    completed.add((cohort, str(row["id"]), "swin_b"))
                    pd.DataFrame(rows).to_csv(result_file, index=False)

                print("[FAITH]", cohort, row["id"])

    finally:
        cam.close()

    results = pd.DataFrame(rows)

    metric_cols = [
        "deletion_auc",
        "random_deletion_auc",
        "aopc",
        "random_aopc",
        "aopc_gain_over_random",
        "attribution_entropy",
        "attribution_concentration",
    ]

    for f in fractions:
        if float(f) <= 0:
            continue
        tag = f"{int(round(float(f) * 100)):02d}"
        metric_cols.extend(
            [
                f"top_drop_{tag}pct",
                f"random_drop_{tag}pct",
            ]
        )

    summary = summarize_metric_table(
        results,
        group_cols=["cohort", "branch"],
        metric_cols=metric_cols,
        n_boot=int(xcfg["bootstrap_iterations"]),
        seed=int(cfg["experiment"]["seed"]),
    )

    summary.to_csv(summary_file, index=False)

    print("[DONE] faithfulness analysis")
    print(
        summary[
            summary["metric"].isin(
                [
                    "aopc",
                    "random_aopc",
                    "aopc_gain_over_random",
                    "deletion_auc",
                    "attribution_entropy",
                    "attribution_concentration",
                ]
            )
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# Stage 4: paper summary / method lock
# ------------------------------------------------------------

def run_summary(cfg: dict, force: bool = False):
    out = get_out(cfg)
    xcfg = _cfg_xai(cfg)

    rows = []

    loc_summary = out / "TRACE_CXR_XAI_LOCALIZATION_SUMMARY.csv"
    faith_summary = out / "TRACE_CXR_XAI_FAITHFULNESS_SUMMARY.csv"

    if loc_summary.exists():
        loc = pd.read_csv(loc_summary)

        wanted = loc[
            loc["metric"].isin(
                [
                    "pointing_hit",
                    "energy_in_box",
                    "iou",
                    "dice",
                    "attribution_entropy",
                    "attribution_concentration",
                ]
            )
        ].copy()

        wanted["analysis"] = "RSNA_bbox_localization"
        wanted["cohort"] = "RSNA_locked_test"
        rows.append(wanted)

    if faith_summary.exists():
        faith = pd.read_csv(faith_summary)

        wanted = faith[
            faith["metric"].isin(
                [
                    "aopc",
                    "random_aopc",
                    "aopc_gain_over_random",
                    "deletion_auc",
                    "random_deletion_auc",
                    "attribution_entropy",
                    "attribution_concentration",
                    "top_drop_20pct",
                    "random_drop_20pct",
                ]
            )
        ].copy()

        wanted["analysis"] = "faithfulness"
        rows.append(wanted)

    if rows:
        paper = pd.concat(rows, ignore_index=True, sort=False)
        paper.to_csv(
            out / "TRACE_CXR_XAI_PAPER_TABLE.csv",
            index=False,
        )

    write_json(
        out / "TRACE_CXR_XAI_METHOD_LOCK_V5.json",
        {
            "framework": TRACE_NAME,
            "experiment_id": TRACE_EXP,
            "qualitative_case_selection": {
                "TP": "highest TRACE-CXR class confidence",
                "TN": "highest TRACE-CXR class confidence",
                "FP": "highest entropy + branch disagreement",
                "FN": "highest entropy + branch disagreement",
                "cases_per_group": int(xcfg["cases_per_confusion_group"]),
            },
            "inception_primary_xai": {
                "method": "Grad-CAM",
                "layer": "Mixed_7c",
                "qualitative_target": "TRACE-CXR predicted class shared with Swin-B",
                "localization_target": "PNEUMONIA positive class",
            },
            "swin_primary_xai": {
                "method": "Integrated Gradients",
                "qualitative_target": "TRACE-CXR predicted class shared with InceptionV3",
                "localization_target": "PNEUMONIA positive class",
                "qualitative_steps": int(xcfg["qualitative_ig_steps"]),
                "quantitative_steps": int(xcfg["quantitative_ig_steps"]),
                "baseline": "zero in normalized input space (ImageNet-mean baseline)",
            },
            "swin_supplementary_xai": {
                "method": "hierarchical window-attention aggregation",
                "scope": "qualitative architectural visualization only",
                "limitation": (
                    "Not exact end-to-end causal attention rollout across PatchMerging."
                ),
            },
            "rsna_localization": {
                "max_cases": int(xcfg["rsna_localization_max_cases"]),
                "case_rule": (
                    "deterministic sample of locked-test GT-positive cases with "
                    "RSNA bounding-box annotations; 0 means all eligible cases"
                ),
                "metrics": [
                    "Pointing Game / Hit Rate",
                    "Energy Inside Ground-Truth Box",
                    "IoU",
                    "Dice",
                    "Normalized Attribution Entropy",
                    "Attribution Concentration",
                ],
                "attribution_top_fraction": float(xcfg["attribution_top_fraction"]),
                "concentration_top_fraction": float(
                    xcfg["concentration_top_fraction"]
                ),
            },
            "faithfulness": {
                "max_cases_per_cohort": int(
                    xcfg["faithfulness_max_cases_per_cohort"]
                ),
                "case_rule": (
                    "deterministic confusion-balanced sample where possible"
                ),
                "deletion_fractions": list(xcfg["deletion_fractions"]),
                "random_repeats": int(xcfg["faithfulness_random_repeats"]),
                "metrics": [
                    "predicted-class confidence drop",
                    "AOPC",
                    "random-baseline AOPC",
                    "AOPC gain over random",
                    "deletion AUC",
                ],
            },
            "qualitative_version": XAI_VERSION,
            "display_percentile": float(xcfg["display_percentile"]),
            "display_smoothing_sigma_px": float(xcfg["display_smoothing_sigma_px"]),
            "display_low_cutoff": float(xcfg["display_low_cutoff"]),
            "qualitative_manifest": MANIFEST_NAME,
            "qualitative_figure_folder": "xai_v5",
            "fusion_visualization": {
                "definition": "alpha*norm(Grad-CAM)+(1-alpha)*norm(Integrated Gradients)",
                "shared_target": "TRACE-CXR predicted class for BOTH branches",
                "normalization": "original branch maps separately normalized [0,1] for numerical archive; display-only percentile scaling + light Gaussian smoothing applied identically to both branches before qualitative combination; no post-mixture renormalization",
                "overlay": "per-pixel alpha proportional to display intensity^gamma after low cutoff",
                "raw_map_archive": "per-case *_raw_maps.npz; not used to recompute localization or faithfulness",
                "swin_diagnostic": "per-case *_swin_diagnostic.png includes original normalized IG without CXR overlay",
                "limitations": "Descriptive mixture; not a derivative of calibrated fusion probability and not a causal full-system explanation",
            },
            "external_tuning": False,
            "scope_statement": (
                "Branch-level shared-target maps and descriptive gate-weighted visualization "
                "do not establish full-system causality, superior focus, or clinical validity."
            ),
        },
    )

    print("[DONE] TRACE-CXR XAI paper summary")



# ------------------------------------------------------------
# Descriptive gate / calibration audit; NO fitting or external tuning
# ------------------------------------------------------------

def _safe_spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    result = spearmanr(a, b, nan_policy="omit")
    return float(result.statistic)


def run_audit(cfg: dict, force: bool = False):
    """Purely descriptive analysis of existing frozen predictions; NEVER refits."""
    out = get_out(cfg)
    files = [out / "TRACE_CXR_INTERNAL_PREDICTIONS.csv",
             out / "TRACE_CXR_KERMANY_PREDICTIONS.csv",
             out / "TRACE_CXR_DEPLOYMENT_LOCK.json"]
    for p in files:
        if not p.exists():
            raise FileNotFoundError(p)
    lock = load_lock(out)
    diagnostic_rows, binned_rows, calibration_rows = [], [], []
    for cohort, fn, metrics_fn in (
        ("RSNA_locked_test", "TRACE_CXR_INTERNAL_PREDICTIONS.csv", "TRACE_CXR_INTERNAL_METRICS.csv"),
        ("Kermany_external", "TRACE_CXR_KERMANY_PREDICTIONS.csv", "TRACE_CXR_KERMANY_METRICS.csv"),
    ):
        df = pd.read_csv(out / fn)
        need = {"label", "inception_v3", "swin_b", "gate_alpha_inception",
                "gated_calibrated", "threshold", "entropy", "disagreement"}
        if not need.issubset(df.columns):
            raise ValueError(f"{fn}: missing {sorted(need-set(df.columns))}")
        if not df["gate_alpha_inception"].between(0, 1).all():
            raise ValueError(f"{fn}: gate alpha outside [0,1]")
        df = add_confusion_fields(df)
        df["gate_input_raw_disagreement"] = np.abs(df["inception_v3"]-df["swin_b"])
        for group, sub in [("all", df), ("correct", df[df["trace_pred"] == df["label"]]),
                           ("error", df[df["trace_pred"] != df["label"]])]:
            if sub.empty:
                continue
            alpha = sub["gate_alpha_inception"]
            diagnostic_rows.append({
                "cohort": cohort, "group": group, "n": len(sub),
                "alpha_mean": float(alpha.mean()),
                "alpha_std": float(alpha.std(ddof=1)) if len(sub)>1 else float("nan"),
                "alpha_min": float(alpha.min()), "alpha_p05": float(alpha.quantile(.05)),
                "alpha_p50": float(alpha.median()), "alpha_p95": float(alpha.quantile(.95)),
                "alpha_max": float(alpha.max()),
                "alpha_range": float(alpha.max()-alpha.min()),
                "alpha_raw_disagreement_spearman": _safe_spearman(
                    alpha, sub["gate_input_raw_disagreement"]),
                "alpha_calibrated_disagreement_spearman": _safe_spearman(
                    alpha, sub["disagreement"]),
                "alpha_fused_entropy_spearman": _safe_spearman(alpha, sub["entropy"]),
                "error_rate": float(np.mean(sub["trace_pred"] != sub["label"])),
            })
        # Quantile bins are observational, and use no external labels for tuning.
        if df["gate_alpha_inception"].nunique() <= 1:
            df["alpha_bin"] = "single_constant_bin"
        else:
            df["alpha_bin"] = pd.qcut(df["gate_alpha_inception"], q=5, duplicates="drop")
        for label, group in df.groupby("alpha_bin", observed=True):
            binned_rows.append({
                "cohort": cohort, "alpha_bin": str(label), "n": int(len(group)),
                "alpha_mean": float(group["gate_alpha_inception"].mean()),
                "error_rate": float(np.mean(group["trace_pred"] != group["label"])),
                "raw_disagreement_mean": float(group["gate_input_raw_disagreement"].mean()),
                "fused_entropy_mean": float(group["entropy"].mean()),
                "pneumonia_prevalence": float(group["label"].mean()),
            })
        metrics_path = out / metrics_fn
        metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        for model, pcol in (
            ("inception_v3", "inception_v3_calibrated"),
            ("swin_b", "swin_b_calibrated"),
            ("mean_fusion", "mean_fusion_calibrated"),
            ("gated_fusion", "gated_fusion_calibrated"),
        ):
            if pcol not in df:
                continue
            values = df[pcol]
            reported = metrics.loc[metrics["model"] == model] if not metrics.empty else pd.DataFrame()
            calibration_rows.append({
                "cohort": cohort, "model": model, "n": int(len(df)),
                "prevalence": float(df["label"].mean()),
                "mean_calibrated_probability": float(values.mean()),
                "mean_probability_minus_prevalence": float(values.mean()-df["label"].mean()),
                "ece15_reported": (float(reported.iloc[0]["ece15"])
                                   if len(reported) and "ece15" in reported else float("nan")),
                "brier_reported": (float(reported.iloc[0]["brier"])
                                   if len(reported) and "brier" in reported else float("nan")),
                "frozen_temperature_from_RSNA": float(lock["models"][model]["temperature"]),
                "frozen_threshold_from_RSNA": float(lock["models"][model]["threshold"]),
                "tuning_on_this_cohort": "NO",
            })

    pd.DataFrame(diagnostic_rows).to_csv(out / "TRACE_CXR_GATE_DIAGNOSTICS.csv", index=False)
    pd.DataFrame(binned_rows).to_csv(out / "TRACE_CXR_GATE_ALPHA_BINS.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(out / "TRACE_CXR_CALIBRATION_TRANSPORT_AUDIT.csv", index=False)
    write_json(out / "TRACE_CXR_GATE_AUDIT_METHOD.json", {
        "version": XAI_VERSION,
        "analysis": "descriptive-only frozen-cohort gate and calibration audit",
        "raw_gate_inputs": ["inception_v3", "swin_b", "entropy of each raw probability", "abs raw probability difference"],
        "reported_correlations": "Spearman observational association; NOT gate calibration or a causal contribution estimate",
        "alpha_bins": "cohort-specific quantile bins; not comparable as identical numeric ranges across cohorts",
        "calibration": "RSNA-fitted temperature and threshold remain frozen on Kermany; no recalibration",
        "note": "External cohort used for audit/reporting only; NEVER model selection or parameter changes",
    })
    print("[AUDIT] Gate diagnostics and calibration transport report generated; no fitting performed.")
    print(pd.DataFrame(diagnostic_rows).to_string(index=False))
    print(pd.DataFrame(calibration_rows).query("model == 'gated_fusion'").to_string(index=False))


# ------------------------------------------------------------
# Status / all
# ------------------------------------------------------------

def run_status(cfg: dict):
    out = get_out(cfg)

    files = [
        "TRACE_CXR_XAI_MANIFEST_V5.csv",
        "TRACE_CXR_XAI_V5_DISPLAY_METHOD.json",
        "TRACE_CXR_XAI_MANIFEST.csv",  # prior V4 stays intact
        "TRACE_CXR_XAI_LOCALIZATION_RSNA.csv",
        "TRACE_CXR_XAI_LOCALIZATION_SUMMARY.csv",
        "TRACE_CXR_XAI_FAITHFULNESS.csv",
        "TRACE_CXR_XAI_FAITHFULNESS_SUMMARY.csv",
        "TRACE_CXR_XAI_PAPER_TABLE.csv",
        "TRACE_CXR_XAI_METHOD_LOCK_V5.json",
        "TRACE_CXR_XAI_METHOD_LOCK.json",  # prior method lock retained
        "TRACE_CXR_GATE_DIAGNOSTICS.csv",
        "TRACE_CXR_GATE_ALPHA_BINS.csv",
        "TRACE_CXR_CALIBRATION_TRANSPORT_AUDIT.csv",
    ]

    for name in files:
        p = out / name

        extra = ""
        if p.exists() and p.suffix.lower() == ".csv":
            try:
                extra = f" rows={len(pd.read_csv(p))}"
            except Exception:
                pass

        print(
            f"{'DONE' if p.exists() else 'PENDING':8s} "
            f"{name}{extra}"
        )


def run_xai_pipeline(cfg: dict, force: bool = False):
    required = [
        get_out(cfg) / "TRACE_CXR_INTERNAL_PREDICTIONS.csv",
        get_out(cfg) / "TRACE_CXR_KERMANY_PREDICTIONS.csv",
        get_out(cfg) / "TRACE_CXR_FINAL_INCEPTIONV3.pt",
        get_out(cfg) / "TRACE_CXR_FINAL_SWINB.pt",
        get_out(cfg) / "TRACE_CXR_DEPLOYMENT_LOCK.json",
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Required frozen TRACE-CXR outputs are missing:\n"
            + "\n".join(missing)
        )

    run_qualitative(cfg, force=force)
    run_localization(cfg, force=force)
    run_faithfulness(cfg, force=force)
    run_summary(cfg, force=force)
    run_audit(cfg, force=force)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "TRACE-CXR final XAI: before/after qualitative XAI, "
            "RSNA bbox localization, and deletion faithfulness."
        )
    )

    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=[
            "qualitative",
            "localization",
            "faithfulness",
            "summary",
            "audit",
            "status",
            "all",
        ],
    )

    parser.add_argument(
        "--config",
        default="configs/trace_cxr.yaml",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the selected XAI stage instead of resuming/skipping.",
    )

    args = parser.parse_args()

    cfg = load_cfg(ROOT / args.config)
    set_seed(cfg["experiment"]["seed"])

    if args.stage == "status":
        run_status(cfg)
        return

    if args.stage == "qualitative":
        run_qualitative(cfg, force=args.force)

    elif args.stage == "localization":
        run_localization(cfg, force=args.force)

    elif args.stage == "faithfulness":
        run_faithfulness(cfg, force=args.force)

    elif args.stage == "summary":
        run_summary(cfg, force=args.force)
        run_audit(cfg, force=args.force)

    elif args.stage == "audit":
        run_audit(cfg, force=args.force)

    else:
        run_xai_pipeline(cfg, force=args.force)

    update_experiment_manifest(
        cfg,
        f"xai_final_{args.stage}",
    )


if __name__ == "__main__":
    main()
