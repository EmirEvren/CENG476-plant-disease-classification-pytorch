from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PROJECT_ROOT / "outputs" / "evaluation"
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints"
OUT = PROJECT_ROOT / "outputs" / "audit" / "full_control"

RUNS = [
    "efficientnet_b0_official_ultrastrict_full",
    "efficientnet_b0_official_ultrastrict_seed123",
    "efficientnet_b0_official_ultrastrict_seed777",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for run in RUNS:
        summary_path = EVAL_ROOT / run / "test_summary.json"
        checkpoint_path = CHECKPOINT_ROOT / f"{run}_best.pt"
        if not summary_path.is_file() or not checkpoint_path.is_file():
            missing.append(run)
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        rows.append({
            "run_name": run,
            "seed": int(checkpoint.get("seed", -1)),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "validation_macro_f1": float(checkpoint["validation_macro_f1"]),
            "test_accuracy": float(summary["test_accuracy"]),
            "test_macro_f1": float(summary["test_macro_f1"]),
            "test_errors": int(summary["test_errors"]),
            "test_examples": int(summary["test_examples"]),
        })

    if missing:
        raise FileNotFoundError(
            "Missing completed seed runs: " + ", ".join(missing)
        )

    frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    frame.to_csv(OUT / "efficientnet_seed_stability_runs.csv", index=False)
    acc = frame["test_accuracy"].to_numpy()
    f1 = frame["test_macro_f1"].to_numpy()
    payload = {
        "seeds": frame["seed"].astype(int).tolist(),
        "fixed_manifest": True,
        "same_locked_test": True,
        "test_examples_each": int(frame["test_examples"].iloc[0]),
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1)),
        "accuracy_min": float(acc.min()),
        "accuracy_max": float(acc.max()),
        "accuracy_range_percentage_points": float((acc.max() - acc.min()) * 100.0),
        "macro_f1_mean": float(f1.mean()),
        "macro_f1_std": float(f1.std(ddof=1)),
        "macro_f1_min": float(f1.min()),
        "macro_f1_max": float(f1.max()),
        "note": (
            "Seed stability is descriptive. No test result was used to choose or discard a seed; "
            "all predefined seeds are reported."
        ),
    }
    with (OUT / "efficientnet_seed_stability_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    display = frame.copy()
    display["test_accuracy"] *= 100.0
    print("="*88)
    print("EFFICIENTNET ULTRA-STRICT SEED STABILITY")
    print("="*88)
    print(display.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(f"Accuracy mean: {payload['accuracy_mean']*100:.3f}%")
    print(f"Accuracy std: {payload['accuracy_std']*100:.3f} pp")
    print(f"Accuracy range: {payload['accuracy_range_percentage_points']:.3f} pp")
    print("Output:", OUT / "efficientnet_seed_stability_summary.json")


if __name__ == "__main__":
    main()
