import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = PROJECT_ROOT / "outputs" / "evaluation"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures"

RUNS = [
    ("Baseline CNN", "baseline_cnn_official_ultrastrict_full"),
    ("ResNet18", "resnet18_official_ultrastrict_full"),
    ("EfficientNet-B0", "efficientnet_b0_official_ultrastrict_full"),
    (
        "ResNet18 + EfficientNet-B0 Ensemble",
        "resnet18_efficientnet_official_ultrastrict_soft_voting",
    ),
]


def load_summary(run_name):
    path = EVALUATION_ROOT / run_name / "test_summary.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing evaluation summary: {path}\nRun all ultra-strict evaluations first."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle), path


def main():
    rows = []
    for label, run_name in RUNS:
        summary, path = load_summary(run_name)
        if int(summary.get("test_examples", -1)) != 10709:
            raise RuntimeError(
                f"{run_name} does not use the unchanged 10,709-image locked test."
            )

        rows.append(
            {
                "model": label,
                "run_name": run_name,
                "test_examples": int(summary["test_examples"]),
                "accuracy": float(summary["test_accuracy"]),
                "macro_precision": float(summary["test_macro_precision"]),
                "macro_recall": float(summary["test_macro_recall"]),
                "macro_f1": float(summary["test_macro_f1"]),
                "weighted_f1": float(summary["test_weighted_f1"]),
                "top3_accuracy": float(summary["test_top3_accuracy"]),
                "macro_roc_auc_ovr": summary.get("test_macro_roc_auc_ovr"),
                "errors": int(
                    summary.get(
                        "errors",
                        summary.get(
                            "test_errors",
                            round(
                                (1.0 - summary["test_accuracy"])
                                * summary["test_examples"]
                            ),
                        ),
                    )
                ),
                "summary_path": str(path.resolve()),
            }
        )

    frame = pd.DataFrame(rows)
    output_dir = EVALUATION_ROOT / "official_ultrastrict_final_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "final_model_comparison.csv"
    json_path = output_dir / "final_model_comparison.json"
    figure_path = FIGURE_ROOT / "official_ultrastrict_final_model_comparison.png"

    frame.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    figure, axis = plt.subplots(figsize=(11, 6))
    bars = axis.bar(frame["model"], frame["accuracy"] * 100)
    axis.bar_label(bars, fmt="%.2f%%", padding=3)
    axis.set_ylabel("Locked official test accuracy (%)")
    axis.set_title("PlantVillage - Ultra-Strict Leakage-Quarantined Comparison")
    lower = max(0.0, min(frame["accuracy"] * 100) - 5.0)
    axis.set_ylim(lower, 100.5)
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    print("=" * 100)
    print("FINAL ULTRA-STRICT LEAKAGE-QUARANTINED RESULTS")
    print("=" * 100)
    display = frame[
        [
            "model",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_roc_auc_ovr",
            "errors",
        ]
    ].copy()
    display["accuracy"] = display["accuracy"] * 100
    print(display.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("Figure:", figure_path)


if __name__ == "__main__":
    main()
