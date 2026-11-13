from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    ks_2samp,
    wasserstein_distance,
)

from trace_cxr import (
    ROOT,
    TRACE_EXP,
    TRACE_NAME,
    TRACE_SCHEMA,
    evaluate_locked_cohort,
    get_out,
    load_cfg,
    set_seed,
    update_experiment_manifest,
    write_json,
)


def kermany_index(
    root: str | Path,
) -> pd.DataFrame:
    root = Path(root)
    rows = []

    for cls, label in (
        ("NORMAL", 0),
        ("PNEUMONIA", 1),
    ):
        folder = root / cls

        for p in sorted(
            folder.glob("*")
        ):
            if (
                p.is_file()
                and p.suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                }
            ):
                rows.append(
                    {
                        "id": p.stem,
                        "path": str(p),
                        "label": label,
                        "source": "Kermany",
                    }
                )

    if not rows:
        raise FileNotFoundError(
            f"No Kermany images found under "
            f"{root}/NORMAL and {root}/PNEUMONIA"
        )

    return pd.DataFrame(
        rows
    )


def kermany_preflight(
    cfg: dict,
):
    df = kermany_index(
        cfg["paths"][
            "kermany_root"
        ]
    )

    print(
        "Kermany:",
        len(df),
        "positive:",
        int(
            df["label"].sum()
        ),
        "negative:",
        int(
            (
                1
                - df["label"]
            ).sum()
        ),
    )


def stage_external(
    cfg: dict,
    force: bool = False,
):
    out = get_out(cfg)

    prediction_file = (
        out
        / "TRACE_CXR_KERMANY_PREDICTIONS.csv"
    )

    metrics_file = (
        out
        / "TRACE_CXR_KERMANY_METRICS.csv"
    )

    if (
        prediction_file.exists()
        and metrics_file.exists()
        and not force
    ):
        print(
            "[SKIP] frozen Kermany outputs complete"
        )
        return

    lock_file = (
        out
        / "TRACE_CXR_DEPLOYMENT_LOCK.json"
    )

    if not lock_file.exists():
        raise FileNotFoundError(
            "TRACE_CXR_DEPLOYMENT_LOCK.json not found. "
            "Run RSNA lock first."
        )

    lock = json.loads(
        lock_file.read_text(
            encoding="utf-8"
        )
    )

    if (
        lock.get(
            "external_tuning"
        )
        is not False
    ):
        raise RuntimeError(
            "Deployment lock does not explicitly prohibit external tuning."
        )

    df = kermany_index(
        cfg["paths"][
            "kermany_root"
        ]
    )

    evaluate_locked_cohort(
        df,
        "Kermany_external",
        cfg,
        out,
        "TRACE_CXR_KERMANY_PREDICTIONS.csv",
        "TRACE_CXR_KERMANY_METRICS.csv",
    )

    write_json(
        out
        / "TRACE_CXR_KERMANY_FREEZE_CERTIFICATE.json",
        {
            "framework": TRACE_NAME,
            "experiment_id": TRACE_EXP,
            "schema": TRACE_SCHEMA,
            "backbones": [
                "InceptionV3",
                "Swin-B",
            ],
            "training_epochs": 60,
            "early_stopping": False,
            "external_dataset": "Kermany",
            "external_tuning": False,
            "statement": (
                "Kermany used the RSNA-frozen InceptionV3, "
                "Swin-B, reliability gate, calibration temperatures, "
                "and operating thresholds without external refitting."
            ),
        },
    )


def stage_shift(
    cfg: dict,
    force: bool = False,
):
    out = get_out(cfg)

    target = (
        out
        / "TRACE_CXR_SHIFT_ANALYSIS.json"
    )

    if (
        target.exists()
        and not force
    ):
        print(
            "[SKIP] shift analysis already exists"
        )
        return

    internal_file = (
        out
        / "TRACE_CXR_INTERNAL_PREDICTIONS.csv"
    )

    external_file = (
        out
        / "TRACE_CXR_KERMANY_PREDICTIONS.csv"
    )

    if (
        not internal_file.exists()
        or not external_file.exists()
    ):
        raise FileNotFoundError(
            "Internal and Kermany predictions are required."
        )

    internal = pd.read_csv(
        internal_file
    )

    external = pd.read_csv(
        external_file
    )

    ks = ks_2samp(
        internal[
            "gated_calibrated"
        ],
        external[
            "gated_calibrated"
        ],
    )

    result = {
        "framework": TRACE_NAME,
        "comparison": (
            "RSNA_locked_test_vs_Kermany_external"
        ),
        "probability_ks_stat": float(
            ks.statistic
        ),
        "probability_ks_p": float(
            ks.pvalue
        ),
        "probability_wasserstein": float(
            wasserstein_distance(
                internal[
                    "gated_calibrated"
                ],
                external[
                    "gated_calibrated"
                ],
            )
        ),
        "probability_mean_internal": float(
            internal[
                "gated_calibrated"
            ].mean()
        ),
        "probability_mean_external": float(
            external[
                "gated_calibrated"
            ].mean()
        ),
        "entropy_mean_internal": float(
            internal[
                "entropy"
            ].mean()
        ),
        "entropy_mean_external": float(
            external[
                "entropy"
            ].mean()
        ),
        "disagreement_mean_internal": float(
            internal[
                "disagreement"
            ].mean()
        ),
        "disagreement_mean_external": float(
            external[
                "disagreement"
            ].mean()
        ),
        "interpretation_limit": (
            "These are prediction-level distributional differences; "
            "they do not establish a causal source of domain shift."
        ),
    }

    write_json(
        target,
        result,
    )

    uncertainty = pd.concat(
        [
            internal.assign(
                cohort="RSNA_locked_test"
            ),
            external.assign(
                cohort="Kermany_external"
            ),
        ],
        ignore_index=True,
    )

    uncertainty[
        "uncertainty_score"
    ] = (
        uncertainty[
            "entropy"
        ]
        + uncertainty[
            "disagreement"
        ]
    )

    uncertainty.to_csv(
        out
        / "TRACE_CXR_UNCERTAINTY_CASES.csv",
        index=False,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


def run_kermany_pipeline(
    cfg: dict,
    force: bool = False,
):
    kermany_preflight(
        cfg
    )

    stage_external(
        cfg,
        force=force,
    )

    stage_shift(
        cfg,
        force=force,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "TRACE-CXR frozen Kermany evaluation "
            "for InceptionV3 + Swin-B."
        )
    )

    parser.add_argument(
        "stage",
        choices=[
            "preflight",
            "external",
            "shift",
            "all",
        ],
        nargs="?",
        default="all",
    )

    parser.add_argument(
        "--config",
        default="configs/trace_cxr.yaml",
    )

    parser.add_argument(
        "--force",
        action="store_true",
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

    if args.stage == "preflight":
        kermany_preflight(
            cfg
        )

    elif args.stage == "external":
        stage_external(
            cfg,
            force=args.force,
        )

    elif args.stage == "shift":
        stage_shift(
            cfg,
            force=args.force,
        )

    else:
        run_kermany_pipeline(
            cfg,
            force=args.force,
        )

    update_experiment_manifest(
        cfg,
        f"kermany_{args.stage}",
    )


if __name__ == "__main__":
    main()
