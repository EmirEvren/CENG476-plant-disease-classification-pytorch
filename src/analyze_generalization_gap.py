from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIST = PROJECT_ROOT / "outputs" / "histories"
CKPT = PROJECT_ROOT / "outputs" / "checkpoints"
EVAL = PROJECT_ROOT / "outputs" / "evaluation"
OUT = PROJECT_ROOT / "outputs" / "audit" / "full_control"

RUNS = [
    ("Baseline CNN", "baseline_cnn_official_ultrastrict_full"),
    ("ResNet18", "resnet18_official_ultrastrict_full"),
    ("EfficientNet-B0", "efficientnet_b0_official_ultrastrict_full"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, run in RUNS:
        history_path = HIST / f"{run}_history.csv"
        checkpoint_path = CKPT / f"{run}_best.pt"
        summary_path = EVAL / run / "test_summary.json"
        for path in (history_path, checkpoint_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        history = pd.read_csv(history_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        with summary_path.open("r", encoding="utf-8") as f:
            test = json.load(f)
        epoch = int(checkpoint["epoch"])
        match = history[history["epoch"].astype(int) == epoch]
        if len(match) != 1:
            raise RuntimeError(f"Best epoch {epoch} missing/duplicated in {history_path}")
        r = match.iloc[0]
        rows.append({
            "model": model,
            "best_epoch": epoch,
            "augmented_train_accuracy": float(r["train_accuracy"]),
            "clean_train_eval_accuracy": float(r["train_eval_accuracy"]),
            "validation_accuracy": float(r["validation_accuracy"]),
            "test_accuracy": float(test["test_accuracy"]),
            "clean_train_minus_validation_pp": (float(r["train_eval_accuracy"]) - float(r["validation_accuracy"])) * 100,
            "validation_minus_test_pp": (float(r["validation_accuracy"]) - float(test["test_accuracy"])) * 100,
            "augmented_train_macro_f1": float(r["train_macro_f1"]),
            "clean_train_eval_macro_f1": float(r["train_eval_macro_f1"]),
            "validation_macro_f1": float(r["validation_macro_f1"]),
            "test_macro_f1": float(test["test_macro_f1"]),
            "clean_train_minus_validation_macro_f1": float(r["train_eval_macro_f1"]) - float(r["validation_macro_f1"]),
            "validation_minus_test_macro_f1": float(r["validation_macro_f1"]) - float(test["test_macro_f1"]),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "generalization_gap.csv", index=False)
    display = frame.copy()
    for col in ["augmented_train_accuracy", "clean_train_eval_accuracy", "validation_accuracy", "test_accuracy"]:
        display[col] *= 100
    print("="*112)
    print("BEST-CHECKPOINT TRAIN / VALIDATION / TEST GENERALIZATION GAP")
    print("="*112)
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nOutput:", OUT / "generalization_gap.csv")


if __name__ == "__main__":
    main()
