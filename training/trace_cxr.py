from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import yaml
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    Inception_V3_Weights,
    Swin_B_Weights,
    inception_v3,
    swin_b,
)

ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = Path(__file__).resolve()

TRACE_NAME = "TRACE-CXR"
TRACE_EXP = "trace_cxr_inception_swinb_v2"
TRACE_SCHEMA = "trace_cxr_transport_v2"

MODEL_NAMES = ("inception_v3", "swin_b")

# ============================================================
# TRACE-CXR V2
# InceptionV3 + Swin-B
#
# IMPORTANT TRAINING LOCK
# - exactly 60 epochs per CNN/Transformer training run
# - NO early stopping
# - every epoch is executed unless the run is resumed
# - best checkpoint is selected AFTER/WHILE the 60-epoch run
#   using highest validation ROC-AUC
# - ties are resolved by lower validation BCE loss
#
# Main responsibilities:
# - RSNA preflight
# - new patient/image-level RSNA split
# - 5-fold OOF InceptionV3 + Swin-B
# - reliability-aware fusion gate
# - final InceptionV3 + Swin-B
# - RSNA-only calibration + threshold lock
# - locked internal evaluation
# - paired statistics
# - manifest / paper table
#
# Kermany:
#   trace_cxr_kermany.py
#
# XAI:
#   trace_cxr_xai.py
# ============================================================


# ------------------------------------------------------------
# General utilities
# ------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def sha256(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_out(cfg: dict) -> Path:
    return ensure_dir(ROOT / cfg["paths"]["output_dir"])


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)


def safe_torch_load(path: str | Path, map_location: str = "cpu"):
    p = Path(path)
    try:
        try:
            return torch.load(p, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(p, map_location=map_location)
    except Exception as e:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bad = p.with_name(p.name + f".corrupt_{stamp}")
        try:
            p.rename(bad)
        except Exception:
            pass
        print(f"[CORRUPT] {p}: {e}")
        print(f"[ACTION] isolated as {bad}")
        return None


def valid_checkpoint(
    path: str | Path,
    required=(
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_val_auc",
        "best_val_loss",
    ),
) -> bool:
    p = Path(path)
    if not p.exists() or p.stat().st_size < 1024:
        return False
    obj = safe_torch_load(p)
    return isinstance(obj, dict) and all(k in obj for k in required)


def valid_done(path: str | Path, expected_run_id: str | None = None) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    try:
        x = json.loads(p.read_text(encoding="utf-8"))
        return x.get("status") == "complete" and (
            expected_run_id is None or x.get("run_id") == expected_run_id
        )
    except Exception:
        return False


def environment_info() -> dict:
    import torchvision

    return {
        "timestamp_utc": now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def update_experiment_manifest(cfg: dict, stage: str) -> None:
    out = get_out(cfg)

    key_files = [
        "TRACE_CXR_RSNA_SPLIT_MANIFEST.csv",
        "TRACE_CXR_OOF_PREDICTIONS.csv",
        "TRACE_CXR_OOF_FUSION.csv",
        "TRACE_CXR_RELIABILITY_GATE.pt",
        "TRACE_CXR_FINAL_INCEPTIONV3.pt",
        "TRACE_CXR_FINAL_SWINB.pt",
        "TRACE_CXR_DEPLOYMENT_LOCK.json",
        "TRACE_CXR_INTERNAL_PREDICTIONS.csv",
        "TRACE_CXR_INTERNAL_METRICS.csv",
        "TRACE_CXR_KERMANY_PREDICTIONS.csv",
        "TRACE_CXR_KERMANY_METRICS.csv",
        "TRACE_CXR_PAIRED_STATISTICS.csv",
        "TRACE_CXR_SHIFT_ANALYSIS.json",
        "TRACE_CXR_XAI_MANIFEST.csv",
        "TRACE_CXR_PAPER_TABLE_MAIN.csv",
    ]

    artifacts = {}
    for name in key_files:
        p = out / name
        if p.exists():
            artifacts[name] = {
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            }

    source_hashes = {
        "trace_cxr.py": sha256(ROOT / "trace_cxr.py"),
        "trace_cxr_kermany.py": sha256(ROOT / "trace_cxr_kermany.py"),
        "trace_cxr_xai.py": sha256(ROOT / "trace_cxr_xai.py"),
    }

    manifest = {
        "framework": TRACE_NAME,
        "experiment_id": TRACE_EXP,
        "schema": TRACE_SCHEMA,
        "experiment": cfg["experiment"],
        "training_lock": {
            "epochs": cfg["training"]["epochs"],
            "early_stopping": False,
            "checkpoint_selection": "highest_validation_roc_auc_tie_lower_validation_bce",
            "mixed_precision": cfg["training"]["mixed_precision"],
            "gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
        },
        "stage_last_completed_or_checked": stage,
        "updated_utc": now_iso(),
        "source_sha256": source_hashes,
        "config_sha256": sha256(ROOT / "configs" / "trace_cxr.yaml"),
        "environment": environment_info(),
        "artifacts": artifacts,
        "external_tuning": False,
        "integrity_note": (
            "Kermany is evaluation-only. No backbone, gate, temperature, "
            "threshold, or proposed-model selection may be changed from "
            "Kermany results."
        ),
    }

    write_json(
        out / "TRACE_CXR_EXPERIMENT_MANIFEST.json",
        manifest,
    )


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

def _to_uint8(arr: np.ndarray, photometric: str | None = None) -> np.ndarray:
    x = arr.astype(np.float32)

    if photometric == "MONOCHROME1":
        x = x.max() - x

    lo, hi = np.percentile(x, [0.5, 99.5])

    if hi <= lo:
        lo = float(x.min())
        hi = float(x.max()) + 1e-6

    x = np.clip(x, lo, hi)
    x = (x - lo) / max(hi - lo, 1e-6)

    return (x * 255.0).round().astype(np.uint8)


def read_cxr(path: str | Path) -> Image.Image:
    path = str(path)

    if path.lower().endswith(".dcm"):
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        arr = (
            arr * float(getattr(ds, "RescaleSlope", 1.0))
            + float(getattr(ds, "RescaleIntercept", 0.0))
        )

        return Image.fromarray(
            _to_uint8(
                arr,
                getattr(ds, "PhotometricInterpretation", None),
            )
        ).convert("RGB")

    return Image.open(path).convert("L").convert("RGB")


def build_transforms(
    image_size: int,
    model_name: str,
    train: bool = False,
):
    # Both torchvision pretrained families use ImageNet normalization.
    # We keep a common spatial size so branch probabilities are aligned
    # on the same underlying resized CXR.
    interpolation = transforms.InterpolationMode.BICUBIC

    ops = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=interpolation,
        )
    ]

    if train:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
            ),
        ]

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]

    return transforms.Compose(ops)


class CXRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int,
        model_name: str,
        train: bool = False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.tf = build_transforms(
            image_size,
            model_name,
            train=train,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = read_cxr(row["path"])
        x = self.tf(image)

        return (
            x,
            int(row["label"]),
            str(row["id"]),
        )


def rsna_index(
    labels_csv: str | Path,
    dicom_dir: str | Path,
) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)

    if not {"patientId", "Target"}.issubset(df.columns):
        raise ValueError(
            "RSNA CSV must contain patientId and Target."
        )

    image_level = (
        df.groupby("patientId", as_index=False)["Target"]
        .max()
    )

    image_level["path"] = image_level["patientId"].map(
        lambda x: str(
            Path(dicom_dir) / f"{x}.dcm"
        )
    )

    image_level["source"] = "RSNA"

    return image_level.rename(
        columns={
            "patientId": "id",
            "Target": "label",
        }
    )[["id", "path", "label", "source"]]


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

def build_model(
    name: str,
    pretrained: bool = True,
) -> nn.Module:
    if name == "inception_v3":
        model = inception_v3(
            weights=(
                Inception_V3_Weights.DEFAULT
                if pretrained
                else None
            ),
            aux_logits=True,
        )

        # TRACE-CXR FIX transform_input: torchvision enables transform_input only when weights are
        # passed, and the flag is not stored in state_dict. Training/OOF used True,
        # so set it explicitly for every build, including load_final_model().
        model.transform_input = True

        model.fc = nn.Linear(
            model.fc.in_features,
            1,
        )

        if model.AuxLogits is not None:
            model.AuxLogits.fc = nn.Linear(
                model.AuxLogits.fc.in_features,
                1,
            )

        return model

    if name == "swin_b":
        model = swin_b(
            weights=(
                Swin_B_Weights.DEFAULT
                if pretrained
                else None
            )
        )

        model.head = nn.Linear(
            model.head.in_features,
            1,
        )

        return model

    raise ValueError(f"Unknown model: {name}")


def forward_logits(
    model: nn.Module,
    x: torch.Tensor,
    model_name: str,
):
    output = model(x)

    if model_name == "inception_v3":
        if hasattr(output, "logits"):
            main_logits = output.logits.view(-1)
            aux_logits = (
                output.aux_logits.view(-1)
                if output.aux_logits is not None
                else None
            )
            return main_logits, aux_logits

    return output.view(-1), None


def model_learning_rate(
    cfg: dict,
    model_name: str,
) -> float:
    if model_name == "inception_v3":
        return float(
            cfg["training"]["lr_inception"]
        )

    if model_name == "swin_b":
        return float(
            cfg["training"]["lr_swin"]
        )

    raise ValueError(model_name)


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_auc: float,
    best_val_loss: float,
    best_epoch: int,
    run_id: str,
    model_name: str,
    epoch_metrics: dict,
) -> None:
    payload = {
        "run_id": run_id,
        "model_name": model_name,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_auc": float(best_val_auc),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epoch_metrics": epoch_metrics,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "saved_utc": now_iso(),
    }

    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def evaluate_validation(
    model: nn.Module,
    val_loader,
    model_name: str,
    dev: str,
):
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()

    losses = []
    y_true = []
    y_prob = []

    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(dev)
            y_float = y.float().to(dev)

            logits, _ = forward_logits(
                model,
                x,
                model_name,
            )

            loss = loss_fn(
                logits,
                y_float,
            )

            probability = torch.sigmoid(
                logits
            ).cpu().numpy()

            losses.append(
                float(loss.cpu())
            )

            y_true.extend(
                y.numpy().tolist()
            )
            y_prob.extend(
                probability.tolist()
            )

    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)

    val_loss = float(np.mean(losses))
    val_auc = float(
        roc_auc_score(
            y_true,
            y_prob,
        )
    )
    val_auprc = float(
        average_precision_score(
            y_true,
            y_prob,
        )
    )
    val_ba_05 = float(
        balanced_accuracy_score(
            y_true,
            (y_prob >= 0.5).astype(int),
        )
    )

    return {
        "val_loss": val_loss,
        "val_auc": val_auc,
        "val_auprc": val_auprc,
        "val_ba_at_0_5": val_ba_05,
    }


def fit_model_fixed_60(
    model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    checkpoint_dir: Path,
    run_id: str,
    model_name: str,
    force: bool = False,
) -> nn.Module:
    """
    Fixed-length training.

    There is NO early stopping.

    Exactly cfg['training']['epochs'] epochs are completed.
    If interrupted, the next run resumes from the last completed epoch.

    Best checkpoint rule:
    1. highest validation ROC-AUC
    2. tie -> lower validation BCE loss
    """
    checkpoint_dir = ensure_dir(
        checkpoint_dir
    )

    last = checkpoint_dir / "last.pt"
    best = checkpoint_dir / "best.pt"
    done = checkpoint_dir / "done.json"
    history_file = (
        checkpoint_dir / "history.csv"
    )

    if force:
        for p in (
            last,
            best,
            done,
            history_file,
        ):
            if p.exists():
                p.unlink()

    dev = device_name()
    use_amp = bool(
        cfg["training"]["mixed_precision"]
        and torch.cuda.is_available()
    )

    model = model.to(dev)

    train_loader = DataLoader(
        CXRDataset(
            train_df,
            cfg["experiment"]["image_size"],
            model_name,
            train=True,
        ),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        CXRDataset(
            val_df,
            cfg["experiment"]["image_size"],
            model_name,
            train=False,
        ),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    lr = model_learning_rate(
        cfg,
        model_name,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"]["min_lr"],
    )

    scaler = make_grad_scaler(
        use_amp
    )

    loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_val_auc = -np.inf
    best_val_loss = np.inf
    best_epoch = -1

    if (
        valid_done(done, run_id)
        and valid_checkpoint(best)
        and not force
    ):
        obj = safe_torch_load(best)
        model.load_state_dict(
            obj["model_state_dict"]
        )

        print(
            f"[COMPLETE] {run_id}: "
            f"60 epochs already complete; "
            f"best epoch={obj['best_epoch'] + 1}, "
            f"best AUC={obj['best_val_auc']:.6f}"
        )

        return model

    if valid_checkpoint(last):
        obj = safe_torch_load(last)

        if (
            obj is not None
            and obj.get("run_id") == run_id
            and obj.get("model_name") == model_name
        ):
            model.load_state_dict(
                obj["model_state_dict"]
            )

            optimizer.load_state_dict(
                obj["optimizer_state_dict"]
            )

            scheduler.load_state_dict(
                obj["scheduler_state_dict"]
            )

            try:
                scaler.load_state_dict(
                    obj.get(
                        "scaler_state_dict",
                        {},
                    )
                )
            except Exception:
                pass

            start_epoch = (
                int(obj["epoch"]) + 1
            )

            best_val_auc = float(
                obj["best_val_auc"]
            )

            best_val_loss = float(
                obj["best_val_loss"]
            )

            best_epoch = int(
                obj["best_epoch"]
            )

            try:
                torch.set_rng_state(
                    obj["torch_rng_state"]
                )

                if (
                    torch.cuda.is_available()
                    and obj.get(
                        "cuda_rng_state_all"
                    )
                    is not None
                ):
                    torch.cuda.set_rng_state_all(
                        obj[
                            "cuda_rng_state_all"
                        ]
                    )

            except Exception:
                pass

            print(
                f"[RESUME] {run_id}: "
                f"continue from epoch "
                f"{start_epoch + 1}/"
                f"{cfg['training']['epochs']}"
            )

    history_rows = []

    if history_file.exists():
        try:
            history_rows = (
                pd.read_csv(history_file)
                .to_dict("records")
            )
        except Exception:
            history_rows = []

    accumulation_steps = int(
        cfg["training"][
            "gradient_accumulation_steps"
        ]
    )

    total_epochs = int(
        cfg["training"]["epochs"]
    )

    if total_epochs != 60:
        raise ValueError(
            "TRACE-CXR V2 is scientifically locked to exactly 60 epochs."
        )

    for epoch in range(
        start_epoch,
        total_epochs,
    ):
        model.train()
        optimizer.zero_grad(
            set_to_none=True
        )

        train_losses = []

        for batch_index, (
            x,
            y,
            _,
        ) in enumerate(train_loader):
            x = x.to(
                dev,
                non_blocking=True,
            )

            y_float = (
                y.float()
                .to(
                    dev,
                    non_blocking=True,
                )
            )

            with autocast_context(
                use_amp
            ):
                main_logits, aux_logits = (
                    forward_logits(
                        model,
                        x,
                        model_name,
                    )
                )

                main_loss = loss_fn(
                    main_logits,
                    y_float,
                )

                if (
                    model_name
                    == "inception_v3"
                    and aux_logits is not None
                ):
                    aux_loss = loss_fn(
                        aux_logits,
                        y_float,
                    )

                    loss = (
                        main_loss
                        + cfg["training"][
                            "inception_aux_weight"
                        ]
                        * aux_loss
                    )

                else:
                    loss = main_loss

                scaled_loss = (
                    loss
                    / accumulation_steps
                )

            scaler.scale(
                scaled_loss
            ).backward()

            should_step = (
                (batch_index + 1)
                % accumulation_steps
                == 0
                or (batch_index + 1)
                == len(train_loader)
            )

            if should_step:
                scaler.step(
                    optimizer
                )
                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            train_losses.append(
                float(loss.detach().cpu())
            )

        val_metrics = evaluate_validation(
            model,
            val_loader,
            model_name,
            dev,
        )

        train_loss = float(
            np.mean(train_losses)
        )

        current_auc = float(
            val_metrics["val_auc"]
        )

        current_loss = float(
            val_metrics["val_loss"]
        )

        is_better = (
            current_auc
            > best_val_auc + 1e-12
        ) or (
            abs(
                current_auc
                - best_val_auc
            )
            <= 1e-12
            and current_loss
            < best_val_loss
        )

        if is_better:
            best_val_auc = current_auc
            best_val_loss = current_loss
            best_epoch = epoch

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": current_loss,
            "val_auc": current_auc,
            "val_auprc": float(
                val_metrics["val_auprc"]
            ),
            "val_ba_at_0_5": float(
                val_metrics[
                    "val_ba_at_0_5"
                ]
            ),
            "lr": float(
                optimizer.param_groups[0]["lr"]
            ),
            "is_best": bool(is_better),
        }

        history_rows = [
            r
            for r in history_rows
            if int(r.get("epoch", -1))
            != epoch + 1
        ]

        history_rows.append(
            epoch_metrics
        )

        pd.DataFrame(
            history_rows
        ).sort_values(
            "epoch"
        ).to_csv(
            history_file,
            index=False,
        )

        save_training_checkpoint(
            last,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_auc,
            best_val_loss,
            best_epoch,
            run_id,
            model_name,
            epoch_metrics,
        )

        if is_better:
            save_training_checkpoint(
                best,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_auc,
                best_val_loss,
                best_epoch,
                run_id,
                model_name,
                epoch_metrics,
            )

        print(
            f"[{run_id}] "
            f"epoch={epoch + 1:02d}/60 "
            f"train={train_loss:.5f} "
            f"val={current_loss:.5f} "
            f"AUC={current_auc:.5f} "
            f"AUPRC={val_metrics['val_auprc']:.5f} "
            f"BA@0.5={val_metrics['val_ba_at_0_5']:.5f} "
            f"{'[BEST]' if is_better else ''}"
        )

        scheduler.step()

    if not valid_checkpoint(best):
        raise RuntimeError(
            f"No valid best checkpoint for {run_id}."
        )

    obj = safe_torch_load(best)

    model.load_state_dict(
        obj["model_state_dict"]
    )

    write_json(
        done,
        {
            "status": "complete",
            "run_id": run_id,
            "model_name": model_name,
            "training_epochs_completed": 60,
            "early_stopping": False,
            "best_epoch": int(
                obj["best_epoch"]
            )
            + 1,
            "best_val_auc": float(
                obj["best_val_auc"]
            ),
            "best_val_loss": float(
                obj["best_val_loss"]
            ),
            "selection_rule": (
                "highest validation ROC-AUC; "
                "tie -> lower validation BCE"
            ),
            "completed_utc": now_iso(),
            "best_checkpoint_sha256": sha256(
                best
            ),
        },
    )

    print(
        f"[DONE] {run_id}: "
        f"all 60 epochs completed; "
        f"selected epoch={obj['best_epoch'] + 1}; "
        f"best AUC={obj['best_val_auc']:.6f}; "
        f"best val loss={obj['best_val_loss']:.6f}"
    )

    return model


@torch.no_grad()
def predict(
    model: nn.Module,
    df: pd.DataFrame,
    cfg: dict,
    model_name: str,
) -> pd.DataFrame:
    dev = device_name()

    model = model.to(dev).eval()

    loader = DataLoader(
        CXRDataset(
            df,
            cfg["experiment"]["image_size"],
            model_name,
            train=False,
        ),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    rows = []

    for x, y, ids in loader:
        x = x.to(
            dev,
            non_blocking=True,
        )

        logits, _ = forward_logits(
            model,
            x,
            model_name,
        )

        probability = torch.sigmoid(
            logits
        ).cpu().numpy()

        rows.extend(
            (
                str(i),
                int(yy),
                float(pp),
            )
            for i, yy, pp in zip(
                ids,
                y.numpy(),
                probability,
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            "id",
            "label",
            "prob",
        ],
    )


# ------------------------------------------------------------
# Reliability-aware fusion
# ------------------------------------------------------------

def binary_entropy(
    p,
    eps: float = 1e-7,
):
    p = np.clip(
        np.asarray(p, float),
        eps,
        1 - eps,
    )

    return -(
        p * np.log(p)
        + (1 - p) * np.log(1 - p)
    )


def gate_features(
    p_inception,
    p_swin,
):
    p_inception = np.asarray(
        p_inception,
        float,
    )

    p_swin = np.asarray(
        p_swin,
        float,
    )

    return np.column_stack(
        [
            p_inception,
            p_swin,
            binary_entropy(
                p_inception
            ),
            binary_entropy(
                p_swin
            ),
            np.abs(
                p_inception
                - p_swin
            ),
        ]
    ).astype(np.float32)


class ReliabilityGate(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 16,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                5,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def train_gate_fixed(
    p_inception,
    p_swin,
    y,
    cfg: dict,
    checkpoint_dir: Path,
    force: bool = False,
) -> ReliabilityGate:
    """
    Gate training is also fixed-length and has NO early stopping.

    Gate uses all OOF probabilities. Since it has no separate gate
    validation subset, the final epoch is retained for the gate.
    The backbone checkpoint selection remains validation-based.
    """
    checkpoint_dir = ensure_dir(
        checkpoint_dir
    )

    final_file = (
        checkpoint_dir
        / "gate_final.pt"
    )

    if final_file.exists() and not force:
        obj = safe_torch_load(
            final_file
        )

        gate = ReliabilityGate(
            cfg["fusion"]["hidden_dim"]
        )

        gate.load_state_dict(
            obj["model_state_dict"]
        )

        print(
            "[COMPLETE] reliability gate already trained"
        )

        return gate

    gate = ReliabilityGate(
        cfg["fusion"]["hidden_dim"]
    )

    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=cfg["fusion"]["lr"],
        weight_decay=cfg["fusion"][
            "weight_decay"
        ],
    )

    x = torch.tensor(
        gate_features(
            p_inception,
            p_swin,
        )
    )

    pi = torch.tensor(
        np.asarray(
            p_inception,
            np.float32,
        )
    )

    ps = torch.tensor(
        np.asarray(
            p_swin,
            np.float32,
        )
    )

    y_t = torch.tensor(
        np.asarray(
            y,
            np.float32,
        )
    )

    loss_fn = nn.BCELoss()

    epochs = int(
        cfg["fusion"]["epochs"]
    )

    if epochs != 60:
        raise ValueError(
            "TRACE-CXR V2 gate is locked to 60 epochs."
        )

    history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        alpha = gate(x)

        fused = torch.clamp(
            alpha * pi
            + (1 - alpha) * ps,
            1e-6,
            1 - 1e-6,
        )

        loss = loss_fn(
            fused,
            y_t,
        )

        loss.backward()
        optimizer.step()

        history.append(
            {
                "epoch": epoch + 1,
                "oof_bce": float(
                    loss.detach()
                ),
            }
        )

        if (
            (epoch + 1) % 10 == 0
            or epoch == 0
        ):
            print(
                f"[gate] epoch={epoch + 1:02d}/60 "
                f"OOF-BCE={float(loss.detach()):.6f}"
            )

    payload = {
        "model_state_dict": gate.state_dict(),
        "epochs_completed": 60,
        "early_stopping": False,
        "trained_on": (
            "RSNA development OOF predictions only"
        ),
        "saved_utc": now_iso(),
    }

    torch.save(
        payload,
        final_file,
    )

    pd.DataFrame(
        history
    ).to_csv(
        checkpoint_dir
        / "gate_history.csv",
        index=False,
    )

    return gate


@torch.no_grad()
def gated_fusion(
    gate,
    p_inception,
    p_swin,
):
    alpha = gate(
        torch.tensor(
            gate_features(
                p_inception,
                p_swin,
            )
        )
    ).numpy()

    p_inception = np.asarray(
        p_inception,
        float,
    )

    p_swin = np.asarray(
        p_swin,
        float,
    )

    fused = (
        alpha * p_inception
        + (1 - alpha) * p_swin
    )

    return fused, alpha


# ------------------------------------------------------------
# Calibration / metrics
# ------------------------------------------------------------

def logit(
    p,
    eps: float = 1e-6,
):
    p = np.clip(
        np.asarray(p, float),
        eps,
        1 - eps,
    )

    return np.log(
        p / (1 - p)
    )


def fit_temperature(
    p,
    y,
) -> float:
    z = logit(p)
    y = np.asarray(
        y,
        float,
    )

    def objective(log_t):
        t = np.exp(log_t)

        q = 1 / (
            1 + np.exp(-z / t)
        )

        q = np.clip(
            q,
            1e-7,
            1 - 1e-7,
        )

        return float(
            -np.mean(
                y * np.log(q)
                + (1 - y)
                * np.log(1 - q)
            )
        )

    result = minimize_scalar(
        objective,
        bounds=(-3, 3),
        method="bounded",
    )

    return float(
        np.exp(result.x)
    )


def apply_temperature(
    p,
    temperature: float,
):
    z = logit(p) / max(
        float(temperature),
        1e-6,
    )

    return 1 / (
        1 + np.exp(-z)
    )


def choose_balanced_accuracy_threshold(
    p,
    y,
):
    p = np.asarray(
        p,
        float,
    )

    y = np.asarray(
        y,
        int,
    )

    best_t = 0.5
    best_score = -1.0

    for threshold in np.unique(
        np.r_[0.0, p, 1.0]
    ):
        score = balanced_accuracy_score(
            y,
            (
                p >= threshold
            ).astype(int),
        )

        if score > best_score:
            best_score = float(score)
            best_t = float(threshold)

    return best_t, best_score


def expected_calibration_error(
    y,
    p,
    bins: int = 15,
) -> float:
    y = np.asarray(y)
    p = np.asarray(p)

    edges = np.linspace(
        0,
        1,
        bins + 1,
    )

    value = 0.0

    for i in range(bins):
        mask = (
            p >= edges[i]
        ) & (
            (p < edges[i + 1])
            if i < bins - 1
            else (p <= edges[i + 1])
        )

        if mask.any():
            value += (
                mask.mean()
                * abs(
                    y[mask].mean()
                    - p[mask].mean()
                )
            )

    return float(value)


def calibration_slope_intercept(
    y,
    p,
):
    z = logit(p).reshape(-1, 1)

    lr = LogisticRegression(
        C=1e6,
        solver="lbfgs",
    ).fit(
        z,
        np.asarray(y),
    )

    return (
        float(
            lr.intercept_[0]
        ),
        float(
            lr.coef_[0, 0]
        ),
    )


def binary_metrics(
    y,
    p,
    threshold: float,
) -> dict:
    y = np.asarray(
        y,
        int,
    )

    p = np.asarray(
        p,
        float,
    )

    pred = (
        p >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    cal_intercept, cal_slope = (
        calibration_slope_intercept(
            y,
            p,
        )
    )

    return {
        "n": len(y),
        "prevalence": float(y.mean()),
        "roc_auc": float(
            roc_auc_score(
                y,
                p,
            )
        ),
        "auprc": float(
            average_precision_score(
                y,
                p,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y,
                pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                pred,
            )
        ),
        "sensitivity": float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "specificity": float(
            tn / max(tn + fp, 1)
        ),
        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y,
                pred,
            )
        ),
        "brier": float(
            brier_score_loss(
                y,
                p,
            )
        ),
        "nll": float(
            log_loss(
                y,
                np.column_stack(
                    [1 - p, p]
                ),
                labels=[0, 1],
            )
        ),
        "ece15": expected_calibration_error(
            y,
            p,
            15,
        ),
        "calibration_intercept": cal_intercept,
        "calibration_slope": cal_slope,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(
            threshold
        ),
    }


def paired_bootstrap_difference(
    y,
    p_a,
    p_b,
    metric_fn,
    n_boot: int,
    seed: int,
):
    y = np.asarray(y)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)

    rng = np.random.default_rng(
        seed
    )

    values = []

    for _ in range(n_boot):
        idx = rng.integers(
            0,
            len(y),
            len(y),
        )

        try:
            values.append(
                metric_fn(
                    y[idx],
                    p_a[idx],
                )
                - metric_fn(
                    y[idx],
                    p_b[idx],
                )
            )
        except Exception:
            continue

    values = np.asarray(
        values,
        float,
    )

    return {
        "mean_diff": float(
            values.mean()
        ),
        "ci_low": float(
            np.quantile(
                values,
                0.025,
            )
        ),
        "ci_high": float(
            np.quantile(
                values,
                0.975,
            )
        ),
    }


def mcnemar_exact(
    y,
    pred_a,
    pred_b,
):
    y = np.asarray(y)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    a_correct = pred_a == y
    b_correct = pred_b == y

    a_correct_b_wrong = int(
        np.sum(
            a_correct
            & ~b_correct
        )
    )

    a_wrong_b_correct = int(
        np.sum(
            ~a_correct
            & b_correct
        )
    )

    n = (
        a_correct_b_wrong
        + a_wrong_b_correct
    )

    p = (
        1.0
        if n == 0
        else float(
            binomtest(
                min(
                    a_correct_b_wrong,
                    a_wrong_b_correct,
                ),
                n=n,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    )

    return {
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "p_value": p,
    }


# ------------------------------------------------------------
# Final model loaders
# ------------------------------------------------------------

def final_model_filename(
    model_name: str,
) -> str:
    if model_name == "inception_v3":
        return "TRACE_CXR_FINAL_INCEPTIONV3.pt"

    if model_name == "swin_b":
        return "TRACE_CXR_FINAL_SWINB.pt"

    raise ValueError(model_name)


def load_final_model(
    model_name: str,
    cfg: dict,
    out: Path,
) -> nn.Module:
    model = build_model(
        model_name,
        pretrained=False,
    )

    obj = safe_torch_load(
        out
        / final_model_filename(
            model_name
        )
    )

    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Cannot load final model: {model_name}"
        )

    if "model_state_dict" in obj:
        model.load_state_dict(
            obj["model_state_dict"]
        )
    else:
        model.load_state_dict(
            obj
        )

    return model


def load_reliability_gate(
    cfg: dict,
    out: Path,
) -> ReliabilityGate:
    obj = safe_torch_load(
        out
        / "TRACE_CXR_RELIABILITY_GATE.pt"
    )

    gate = ReliabilityGate(
        cfg["fusion"]["hidden_dim"]
    )

    if (
        isinstance(obj, dict)
        and "model_state_dict" in obj
    ):
        gate.load_state_dict(
            obj["model_state_dict"]
        )
    else:
        gate.load_state_dict(
            obj
        )

    return gate.eval()


def backbone_predictions(
    df: pd.DataFrame,
    cfg: dict,
    out: Path,
) -> pd.DataFrame:
    frames = {}

    for model_name in MODEL_NAMES:
        pred = predict(
            load_final_model(
                model_name,
                cfg,
                out,
            ),
            df,
            cfg,
            model_name,
        ).rename(
            columns={
                "prob": model_name
            }
        )

        frames[model_name] = pred

    return frames[
        "inception_v3"
    ].merge(
        frames["swin_b"],
        on=[
            "id",
            "label",
        ],
    )


# ------------------------------------------------------------
# Locked cohort evaluation
# ------------------------------------------------------------

def evaluate_locked_cohort(
    df: pd.DataFrame,
    cohort_name: str,
    cfg: dict,
    out: Path,
    predictions_name: str,
    metrics_name: str,
):
    lock = json.loads(
        (
            out
            / "TRACE_CXR_DEPLOYMENT_LOCK.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    x = backbone_predictions(
        df,
        cfg,
        out,
    )

    x["mean_fusion"] = (
        x["inception_v3"]
        + x["swin_b"]
    ) / 2.0

    gated_raw, alpha = gated_fusion(
        load_reliability_gate(
            cfg,
            out,
        ),
        x["inception_v3"],
        x["swin_b"],
    )

    x["gated_fusion"] = (
        gated_raw
    )

    x["gate_alpha_inception"] = (
        alpha
    )

    rows = []

    for model_name in (
        "inception_v3",
        "swin_b",
        "mean_fusion",
        "gated_fusion",
    ):
        spec = lock[
            "models"
        ][model_name]

        calibrated = apply_temperature(
            x[model_name],
            spec["temperature"],
        )

        calibrated_col = (
            model_name
            + "_calibrated"
        )

        x[
            calibrated_col
        ] = calibrated

        row = binary_metrics(
            x["label"],
            calibrated,
            spec["threshold"],
        )

        row["model"] = model_name
        row["cohort"] = cohort_name

        rows.append(row)

    primary_p = x[
        "gated_fusion_calibrated"
    ]

    primary_t = lock[
        "models"
    ][
        "gated_fusion"
    ][
        "threshold"
    ]

    x["gated_calibrated"] = (
        primary_p
    )

    x["threshold"] = (
        primary_t
    )

    x["entropy"] = (
        binary_entropy(
            primary_p
        )
    )

    x["disagreement"] = np.abs(
        x[
            "inception_v3_calibrated"
        ]
        - x[
            "swin_b_calibrated"
        ]
    )

    x.to_csv(
        out / predictions_name,
        index=False,
    )

    pd.DataFrame(
        rows
    ).to_csv(
        out / metrics_name,
        index=False,
    )


# ------------------------------------------------------------
# Stages
# ------------------------------------------------------------

def stage_preflight(
    cfg: dict,
    force: bool = False,
):
    rsna = rsna_index(
        cfg["paths"][
            "rsna_labels_csv"
        ],
        cfg["paths"][
            "rsna_dicom_dir"
        ],
    )

    print(
        f"FRAMEWORK: {TRACE_NAME}"
    )
    print(
        f"EXPERIMENT: {TRACE_EXP}"
    )
    print(
        f"SCHEMA: {TRACE_SCHEMA}"
    )
    print(
        "BACKBONES: InceptionV3 + Swin-B"
    )
    print(
        "TRAINING: exactly 60 epochs, NO early stopping"
    )
    print(
        "BEST CHECKPOINT: highest val ROC-AUC, tie -> lower val BCE"
    )
    print(
        f"DEVICE: {device_name()}"
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "RSNA:",
        len(rsna),
        "positive:",
        int(
            rsna["label"].sum()
        ),
        "negative:",
        int(
            (
                1
                - rsna["label"]
            ).sum()
        ),
    )

    print(
        "RSNA labels SHA256:",
        sha256(
            cfg["paths"][
                "rsna_labels_csv"
            ]
        ),
    )

    from trace_cxr_kermany import (
        kermany_preflight,
    )

    kermany_preflight(
        cfg
    )


def stage_split(
    cfg: dict,
    force: bool = False,
):
    out = get_out(cfg)

    target = (
        out
        / "TRACE_CXR_RSNA_SPLIT_MANIFEST.csv"
    )

    if (
        target.exists()
        and not force
    ):
        print(
            "[SKIP] RSNA split already exists"
        )
        return

    df = rsna_index(
        cfg["paths"][
            "rsna_labels_csv"
        ],
        cfg["paths"][
            "rsna_dicom_dir"
        ],
    )

    split_cfg = cfg["split"]
    seed = cfg[
        "experiment"
    ]["seed"]

    dev_cal, test = (
        train_test_split(
            df,
            test_size=split_cfg[
                "test_fraction"
            ],
            random_state=seed,
            stratify=df[
                "label"
            ],
        )
    )

    cal_relative = (
        split_cfg[
            "calibration_fraction"
        ]
        / (
            split_cfg[
                "development_fraction"
            ]
            + split_cfg[
                "calibration_fraction"
            ]
        )
    )

    dev, cal = (
        train_test_split(
            dev_cal,
            test_size=cal_relative,
            random_state=seed + 1,
            stratify=dev_cal[
                "label"
            ],
        )
    )

    dev = dev.copy()
    cal = cal.copy()
    test = test.copy()

    dev["split"] = (
        "development"
    )
    cal["split"] = (
        "calibration"
    )
    test["split"] = (
        "locked_test"
    )

    dev["fold"] = -1
    cal["fold"] = np.nan
    test["fold"] = np.nan

    skf = StratifiedKFold(
        n_splits=split_cfg[
            "oof_folds"
        ],
        shuffle=True,
        random_state=seed,
    )

    for fold, (
        _,
        val_idx,
    ) in enumerate(
        skf.split(
            dev,
            dev["label"],
        )
    ):
        dev.iloc[
            val_idx,
            dev.columns.get_loc(
                "fold"
            ),
        ] = fold

    combined = pd.concat(
        [
            dev,
            cal,
            test,
        ],
        ignore_index=True,
    )

    if combined[
        "id"
    ].duplicated().any():
        raise RuntimeError(
            "Patient/image overlap detected in RSNA split."
        )

    combined.to_csv(
        target,
        index=False,
    )

    summary = (
        combined
        .groupby(
            "split"
        )["label"]
        .agg(
            [
                "count",
                "sum",
            ]
        )
        .reset_index()
    )

    summary[
        "negative"
    ] = (
        summary["count"]
        - summary["sum"]
    )

    summary.to_csv(
        out
        / "TRACE_CXR_RSNA_SPLIT_SUMMARY.csv",
        index=False,
    )

    print(
        summary.to_string(
            index=False
        )
    )


def stage_oof(
    cfg: dict,
    force: bool = False,
):
    out = get_out(cfg)

    manifest = pd.read_csv(
        out
        / "TRACE_CXR_RSNA_SPLIT_MANIFEST.csv"
    )

    dev = manifest[
        manifest["split"]
        == "development"
    ].copy()

    fold_predictions = []

    for model_name in MODEL_NAMES:
        for fold in sorted(
            dev["fold"]
            .dropna()
            .unique()
            .astype(int)
        ):
            run_id = (
                f"TRACE_CXR_OOF_"
                f"{model_name}_fold{fold}"
            )

            prediction_file = (
                out
                / f"{run_id}_predictions.csv"
            )

            checkpoint_dir = (
                out
                / "checkpoints"
                / "oof"
                / model_name
                / f"fold{fold}"
            )

            if (
                prediction_file.exists()
                and valid_done(
                    checkpoint_dir
                    / "done.json",
                    run_id,
                )
                and not force
            ):
                print(
                    f"[SKIP COMPLETE] {run_id}"
                )

                fold_predictions.append(
                    pd.read_csv(
                        prediction_file
                    )
                )

                continue

            train_df = dev[
                dev["fold"]
                != fold
            ]

            val_df = dev[
                dev["fold"]
                == fold
            ]

            print(
                f"\n=== {run_id} ==="
            )

            model = fit_model_fixed_60(
                build_model(
                    model_name,
                    pretrained=True,
                ),
                train_df,
                val_df,
                cfg,
                checkpoint_dir,
                run_id,
                model_name,
                force=force,
            )

            pred = predict(
                model,
                val_df,
                cfg,
                model_name,
            )

            pred["model"] = (
                model_name
            )
            pred["fold"] = (
                fold
            )

            pred.to_csv(
                prediction_file,
                index=False,
            )

            fold_predictions.append(
                pred
            )

    oof = pd.concat(
        fold_predictions,
        ignore_index=True,
    )

    oof.to_csv(
        out
        / "TRACE_CXR_OOF_PREDICTIONS.csv",
        index=False,
    )

    print(
        "[DONE] TRACE-CXR OOF predictions:",
        len(oof),
    )


def stage_gate(
    cfg: dict,
    force: bool = False,
):
    out = get_out(cfg)

    gate_file = (
        out
        / "TRACE_CXR_RELIABILITY_GATE.pt"
    )

    fusion_file = (
        out
        / "TRACE_CXR_OOF_FUSION.csv"
    )

    if (
        gate_file.exists()
        and fusion_file.exists()
        and not force
    ):
        print(
            "[SKIP] reliability gate outputs complete"
        )
        return

    long_df = pd.read_csv(
        out
        / "TRACE_CXR_OOF_PREDICTIONS.csv"
    )

    wide = (
        long_df
        .pivot_table(
            index=[
                "id",
                "label",
            ],
            columns="model",
            values="prob",
        )
        .reset_index()
    )

    gate = train_gate_fixed(
        wide[
            "inception_v3"
        ],
        wide[
            "swin_b"
        ],
        wide[
            "label"
        ],
        cfg,
        out
        / "checkpoints"
        / "gate",
        force=force,
    )

    gate_payload = (
        safe_torch_load(
            out
            / "checkpoints"
            / "gate"
            / "gate_final.pt"
        )
    )

    torch.save(
        gate_payload,
        gate_file,
    )

    fused, alpha = (
        gated_fusion(
            gate,
            wide[
                "inception_v3"
            ],
            wide[
                "swin_b"
            ],
        )
    )

    wide[
        "mean_fusion"
    ] = (
        wide[
            "inception_v3"
        ]
        + wide[
            "swin_b"
        ]
    ) / 2.0

    wide[
        "gated_fusion"
    ] = fused

    wide[
        "gate_alpha_inception"
    ] = alpha

    wide.to_csv(
        fusion_file,
        index=False,
    )


def stage_final(
    cfg: dict,
    force: bool = False,
):
    out = get_out(cfg)

    manifest = pd.read_csv(
        out
        / "TRACE_CXR_RSNA_SPLIT_MANIFEST.csv"
    )

    dev = manifest[
        manifest["split"]
        == "development"
    ].copy()

    train_df, val_df = (
        train_test_split(
            dev,
            test_size=cfg[
                "training"
            ][
                "final_holdout_fraction"
            ],
            random_state=(
                cfg[
                    "experiment"
                ][
                    "seed"
                ]
                + 9
            ),
            stratify=dev[
                "label"
            ],
        )
    )

    for model_name in MODEL_NAMES:
        run_id = (
            f"TRACE_CXR_FINAL_"
            f"{model_name}"
        )

        checkpoint_dir = (
            out
            / "checkpoints"
            / "final"
            / model_name
        )

        final_path = (
            out
            / final_model_filename(
                model_name
            )
        )

        if (
            final_path.exists()
            and valid_done(
                checkpoint_dir
                / "done.json",
                run_id,
            )
            and not force
        ):
            print(
                f"[SKIP COMPLETE] {run_id}"
            )
            continue

        print(
            f"\n=== {run_id} ==="
        )

        fit_model_fixed_60(
            build_model(
                model_name,
                pretrained=True,
            ),
            train_df,
            val_df,
            cfg,
            checkpoint_dir,
            run_id,
            model_name,
            force=force,
        )

        best = safe_torch_load(
            checkpoint_dir
            / "best.pt"
        )

        torch.save(
            best,
            final_path,
        )


def stage_lock(
    cfg: dict,
    force: bool = False,
):
    out = get_out(cfg)

    lock_file = (
        out
        / "TRACE_CXR_DEPLOYMENT_LOCK.json"
    )

    cal_file = (
        out
        / "TRACE_CXR_CALIBRATION_PREDICTIONS.csv"
    )

    if (
        lock_file.exists()
        and cal_file.exists()
        and not force
    ):
        print(
            "[SKIP] deployment lock complete"
        )
        return

    manifest = pd.read_csv(
        out
        / "TRACE_CXR_RSNA_SPLIT_MANIFEST.csv"
    )

    cal = manifest[
        manifest["split"]
        == "calibration"
    ].copy()

    df = backbone_predictions(
        cal,
        cfg,
        out,
    )

    df[
        "mean_fusion"
    ] = (
        df[
            "inception_v3"
        ]
        + df[
            "swin_b"
        ]
    ) / 2.0

    gated_raw, alpha = (
        gated_fusion(
            load_reliability_gate(
                cfg,
                out,
            ),
            df[
                "inception_v3"
            ],
            df[
                "swin_b"
            ],
        )
    )

    df[
        "gated_fusion"
    ] = gated_raw

    df[
        "gate_alpha_inception"
    ] = alpha

    lock = {
        "framework": TRACE_NAME,
        "experiment_id": TRACE_EXP,
        "schema": TRACE_SCHEMA,
        "backbones": [
            "InceptionV3",
            "Swin-B",
        ],
        "training_epochs": 60,
        "early_stopping": False,
        "backbone_checkpoint_rule": (
            "highest validation ROC-AUC; "
            "tie -> lower validation BCE"
        ),
        "fitted_on": (
            "RSNA_calibration_only"
        ),
        "external_tuning": False,
        "primary_model": (
            "gated_fusion"
        ),
        "models": {},
    }

    for model_name in (
        "inception_v3",
        "swin_b",
        "mean_fusion",
        "gated_fusion",
    ):
        temperature = (
            fit_temperature(
                df[
                    model_name
                ],
                df[
                    "label"
                ],
            )
        )

        calibrated = (
            apply_temperature(
                df[
                    model_name
                ],
                temperature,
            )
        )

        threshold, cal_ba = (
            choose_balanced_accuracy_threshold(
                calibrated,
                df[
                    "label"
                ],
            )
        )

        df[
            model_name
            + "_calibrated"
        ] = calibrated

        lock[
            "models"
        ][
            model_name
        ] = {
            "temperature": temperature,
            "threshold": threshold,
            "calibration_balanced_accuracy": cal_ba,
        }

    write_json(
        lock_file,
        lock,
    )

    df.to_csv(
        cal_file,
        index=False,
    )

    print(
        json.dumps(
            lock,
            indent=2,
        )
    )


def stage_internal(
    cfg: dict,
    force: bool = False,
):
    out = get_out(cfg)

    metrics_file = (
        out
        / "TRACE_CXR_INTERNAL_METRICS.csv"
    )

    predictions_file = (
        out
        / "TRACE_CXR_INTERNAL_PREDICTIONS.csv"
    )

    if (
        metrics_file.exists()
        and predictions_file.exists()
        and not force
    ):
        print(
            "[SKIP] locked internal outputs complete"
        )
        return

    manifest = pd.read_csv(
        out
        / "TRACE_CXR_RSNA_SPLIT_MANIFEST.csv"
    )

    test = manifest[
        manifest["split"]
        == "locked_test"
    ].copy()

    evaluate_locked_cohort(
        test,
        "RSNA_locked_test",
        cfg,
        out,
        "TRACE_CXR_INTERNAL_PREDICTIONS.csv",
        "TRACE_CXR_INTERNAL_METRICS.csv",
    )


def stage_stats(
    cfg: dict,
    force: bool = False,
):
    out = get_out(cfg)

    target = (
        out
        / "TRACE_CXR_PAIRED_STATISTICS.csv"
    )

    if (
        target.exists()
        and not force
    ):
        print(
            "[SKIP] paired statistics already exist"
        )
        return

    required = [
        out
        / "TRACE_CXR_INTERNAL_PREDICTIONS.csv",
        out
        / "TRACE_CXR_KERMANY_PREDICTIONS.csv",
        out
        / "TRACE_CXR_DEPLOYMENT_LOCK.json",
    ]

    if not all(
        p.exists()
        for p in required
    ):
        raise FileNotFoundError(
            "Run internal and Kermany evaluation before statistics."
        )

    lock = json.loads(
        (
            out
            / "TRACE_CXR_DEPLOYMENT_LOCK.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    rows = []

    for cohort, filename in (
        (
            "RSNA_locked_test",
            "TRACE_CXR_INTERNAL_PREDICTIONS.csv",
        ),
        (
            "Kermany_external",
            "TRACE_CXR_KERMANY_PREDICTIONS.csv",
        ),
    ):
        df = pd.read_csv(
            out / filename
        )

        y = df[
            "label"
        ].to_numpy()

        proposed = df[
            "gated_fusion_calibrated"
        ].to_numpy()

        comparator = df[
            "mean_fusion_calibrated"
        ].to_numpy()

        proposed_threshold = (
            lock[
                "models"
            ][
                "gated_fusion"
            ][
                "threshold"
            ]
        )

        comparator_threshold = (
            lock[
                "models"
            ][
                "mean_fusion"
            ][
                "threshold"
            ]
        )

        auc_result = (
            paired_bootstrap_difference(
                y,
                proposed,
                comparator,
                lambda yy, pp: (
                    roc_auc_score(
                        yy,
                        pp,
                    )
                ),
                cfg[
                    "statistics"
                ][
                    "bootstrap_iterations"
                ],
                cfg[
                    "experiment"
                ][
                    "seed"
                ],
            )
        )

        auc_result.update(
            {
                "cohort": cohort,
                "comparison": (
                    "gated_vs_mean"
                ),
                "metric": "roc_auc",
            }
        )

        rows.append(
            auc_result
        )

        brier_result = (
            paired_bootstrap_difference(
                y,
                proposed,
                comparator,
                lambda yy, pp: (
                    brier_score_loss(
                        yy,
                        pp,
                    )
                ),
                cfg[
                    "statistics"
                ][
                    "bootstrap_iterations"
                ],
                cfg[
                    "experiment"
                ][
                    "seed"
                ],
            )
        )

        brier_result.update(
            {
                "cohort": cohort,
                "comparison": (
                    "gated_vs_mean"
                ),
                "metric": "brier",
            }
        )

        rows.append(
            brier_result
        )

        rng = (
            np.random.default_rng(
                cfg[
                    "experiment"
                ][
                    "seed"
                ]
            )
        )

        ba_values = []

        for _ in range(
            cfg[
                "statistics"
            ][
                "bootstrap_iterations"
            ]
        ):
            idx = rng.integers(
                0,
                len(y),
                len(y),
            )

            ba_values.append(
                balanced_accuracy_score(
                    y[idx],
                    (
                        proposed[idx]
                        >= proposed_threshold
                    ).astype(int),
                )
                - balanced_accuracy_score(
                    y[idx],
                    (
                        comparator[idx]
                        >= comparator_threshold
                    ).astype(int),
                )
            )

        ba_values = np.asarray(
            ba_values,
            float,
        )

        rows.append(
            {
                "cohort": cohort,
                "comparison": (
                    "gated_vs_mean"
                ),
                "metric": (
                    "balanced_accuracy"
                ),
                "mean_diff": float(
                    ba_values.mean()
                ),
                "ci_low": float(
                    np.quantile(
                        ba_values,
                        0.025,
                    )
                ),
                "ci_high": float(
                    np.quantile(
                        ba_values,
                        0.975,
                    )
                ),
            }
        )

        mc = mcnemar_exact(
            y,
            (
                proposed
                >= proposed_threshold
            ).astype(int),
            (
                comparator
                >= comparator_threshold
            ).astype(int),
        )

        mc.update(
            {
                "cohort": cohort,
                "comparison": (
                    "gated_vs_mean"
                ),
                "metric": "mcnemar",
            }
        )

        rows.append(
            mc
        )

    pd.DataFrame(
        rows
    ).to_csv(
        target,
        index=False,
    )


def stage_assets(
    cfg: dict,
    force: bool = False,
):
    out = get_out(cfg)

    target = (
        out
        / "TRACE_CXR_PAPER_TABLE_MAIN.csv"
    )

    if (
        target.exists()
        and not force
    ):
        print(
            "[SKIP] paper table already exists"
        )
        return

    internal = pd.read_csv(
        out
        / "TRACE_CXR_INTERNAL_METRICS.csv"
    )

    external = pd.read_csv(
        out
        / "TRACE_CXR_KERMANY_METRICS.csv"
    )

    table = pd.concat(
        [
            internal,
            external,
        ],
        ignore_index=True,
    )

    table.to_csv(
        target,
        index=False,
    )

    columns = [
        "cohort",
        "model",
        "roc_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "brier",
        "ece15",
    ]

    print(
        table[
            columns
        ].to_string(
            index=False
        )
    )


def stage_status(
    cfg: dict,
    force: bool = False,
):
    out = get_out(cfg)

    files = {
        "RSNA split": (
            "TRACE_CXR_RSNA_SPLIT_MANIFEST.csv"
        ),
        "OOF": (
            "TRACE_CXR_OOF_PREDICTIONS.csv"
        ),
        "gate": (
            "TRACE_CXR_RELIABILITY_GATE.pt"
        ),
        "final Inception": (
            "TRACE_CXR_FINAL_INCEPTIONV3.pt"
        ),
        "final Swin-B": (
            "TRACE_CXR_FINAL_SWINB.pt"
        ),
        "deployment lock": (
            "TRACE_CXR_DEPLOYMENT_LOCK.json"
        ),
        "internal metrics": (
            "TRACE_CXR_INTERNAL_METRICS.csv"
        ),
        "Kermany metrics": (
            "TRACE_CXR_KERMANY_METRICS.csv"
        ),
        "paired stats": (
            "TRACE_CXR_PAIRED_STATISTICS.csv"
        ),
        "shift": (
            "TRACE_CXR_SHIFT_ANALYSIS.json"
        ),
        "XAI": (
            "TRACE_CXR_XAI_MANIFEST.csv"
        ),
        "paper table": (
            "TRACE_CXR_PAPER_TABLE_MAIN.csv"
        ),
        "manifest": (
            "TRACE_CXR_EXPERIMENT_MANIFEST.json"
        ),
    }

    for label, filename in files.items():
        path = out / filename

        print(
            f"{label:22s} "
            f"{'DONE' if path.exists() else 'PENDING'} "
            f"{path}"
        )

    checkpoint_root = (
        out
        / "checkpoints"
    )

    if checkpoint_root.exists():
        print(
            "\nCompletion markers:"
        )

        for p in sorted(
            checkpoint_root.rglob(
                "done.json"
            )
        ):
            print(
                "  DONE",
                p.relative_to(out),
            )

        print(
            "\nResume checkpoints:"
        )

        for p in sorted(
            checkpoint_root.rglob(
                "last.pt"
            )
        ):
            obj = safe_torch_load(
                p
            )

            epoch_text = "?"

            if (
                isinstance(obj, dict)
                and "epoch" in obj
            ):
                epoch_text = (
                    f"{int(obj['epoch']) + 1}/60"
                )

            print(
                "  LAST",
                p.relative_to(out),
                f"epoch={epoch_text}",
                f"{p.stat().st_size / 1024 / 1024:.1f} MB",
            )


MAIN_STAGES = {
    "preflight": stage_preflight,
    "split": stage_split,
    "oof": stage_oof,
    "gate": stage_gate,
    "final": stage_final,
    "lock": stage_lock,
    "internal": stage_internal,
    "stats": stage_stats,
    "assets": stage_assets,
}

MAIN_ORDER = [
    "preflight",
    "split",
    "oof",
    "gate",
    "final",
    "lock",
    "internal",
]


def run_full_pipeline(
    cfg: dict,
    force: bool = False,
):
    for stage in MAIN_ORDER:
        print(
            "\n"
            + "=" * 78
            + f"\nTRACE-CXR STAGE: {stage.upper()}\n"
            + "=" * 78
        )

        MAIN_STAGES[
            stage
        ](
            cfg,
            force,
        )

        update_experiment_manifest(
            cfg,
            stage,
        )

    from trace_cxr_kermany import (
        run_kermany_pipeline,
    )

    from trace_cxr_xai import (
        run_xai_pipeline,
    )

    print(
        "\n"
        + "=" * 78
        + "\nTRACE-CXR STAGE: KERMANY\n"
        + "=" * 78
    )

    run_kermany_pipeline(
        cfg,
        force=force,
    )

    update_experiment_manifest(
        cfg,
        "kermany",
    )

    print(
        "\n"
        + "=" * 78
        + "\nTRACE-CXR STAGE: STATISTICS\n"
        + "=" * 78
    )

    stage_stats(
        cfg,
        force,
    )

    update_experiment_manifest(
        cfg,
        "statistics",
    )

    print(
        "\n"
        + "=" * 78
        + "\nTRACE-CXR STAGE: XAI\n"
        + "=" * 78
    )

    run_xai_pipeline(
        cfg,
        force=force,
    )

    update_experiment_manifest(
        cfg,
        "xai",
    )

    print(
        "\n"
        + "=" * 78
        + "\nTRACE-CXR STAGE: PAPER ASSETS\n"
        + "=" * 78
    )

    stage_assets(
        cfg,
        force,
    )

    update_experiment_manifest(
        cfg,
        "assets",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "TRACE-CXR V2 main pipeline: "
            "InceptionV3 + Swin-B, fixed 60 epochs."
        )
    )

    parser.add_argument(
        "stage",
        choices=list(
            MAIN_STAGES.keys()
        )
        + [
            "kermany",
            "xai",
            "all",
            "status",
        ],
    )

    parser.add_argument(
        "--config",
        default="configs/trace_cxr.yaml",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Intentionally rebuild selected stage. "
            "Do not use merely to resume."
        ),
    )

    args = parser.parse_args()

    cfg = load_cfg(
        ROOT / args.config
    )

    set_seed(
        cfg[
            "experiment"
        ][
            "seed"
        ]
    )

    if args.stage == "status":
        stage_status(
            cfg
        )
        return

    if args.stage == "all":
        run_full_pipeline(
            cfg,
            force=args.force,
        )
        return

    if args.stage == "kermany":
        from trace_cxr_kermany import (
            run_kermany_pipeline,
        )

        run_kermany_pipeline(
            cfg,
            force=args.force,
        )

        update_experiment_manifest(
            cfg,
            "kermany",
        )

        return

    if args.stage == "xai":
        from trace_cxr_xai import (
            run_xai_pipeline,
        )

        run_xai_pipeline(
            cfg,
            force=args.force,
        )

        update_experiment_manifest(
            cfg,
            "xai",
        )

        return

    MAIN_STAGES[
        args.stage
    ](
        cfg,
        args.force,
    )

    update_experiment_manifest(
        cfg,
        args.stage,
    )


if __name__ == "__main__":
    main()
