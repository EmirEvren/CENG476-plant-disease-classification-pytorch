import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = {
    0.2: "baseline_dropout02_pilot",
    0.4: "baseline_b64_lr5e4_pilot",
    0.6: "baseline_dropout06_pilot",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline CNN dropout ablation runs using validation Macro-F1."
        )
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "histories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "comparison",
    )
    return parser.parse_args()


def _load_run(history_dir, dropout, run_name):
    history_path = history_dir / f"{run_name}_history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(
            f"Missing history for dropout={dropout}: {history_path}\n"
            "Run the required pilot experiment first."
        )

    frame = pd.read_csv(history_path)
    required = {"epoch", "validation_loss", "validation_macro_f1"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{history_path} is missing columns: {sorted(missing)}"
        )

    best_index = frame["validation_macro_f1"].idxmax()
    best_row = frame.loc[best_index]
    final_row = frame.iloc[-1]

    return frame, {
        "dropout": dropout,
        "run_name": run_name,
        "epochs_completed": int(frame["epoch"].iloc[-1]),
        "best_epoch": int(best_row["epoch"]),
        "best_validation_macro_f1": float(best_row["validation_macro_f1"]),
        "best_validation_loss": float(best_row["validation_loss"]),
        "final_validation_macro_f1": float(final_row["validation_macro_f1"]),
        "final_validation_loss": float(final_row["validation_loss"]),
    }


def main():
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    rows = []
    for dropout, run_name in DEFAULT_RUNS.items():
        frame, summary = _load_run(
            arguments.history_dir,
            dropout,
            run_name,
        )
        histories[dropout] = frame
        rows.append(summary)

    comparison = pd.DataFrame(rows).sort_values("dropout")
    csv_path = arguments.output_dir / "dropout_ablation.csv"
    comparison.to_csv(csv_path, index=False)

    figure_path = (
        PROJECT_ROOT
        / "outputs"
        / "figures"
        / "dropout_ablation_validation_macro_f1.png"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    for dropout, frame in histories.items():
        axis.plot(
            frame["epoch"],
            frame["validation_macro_f1"],
            marker="o",
            label=f"Dropout={dropout:.1f}",
        )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation Macro-F1")
    axis.set_title("Baseline CNN Dropout Ablation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    best = comparison.loc[comparison["best_validation_macro_f1"].idxmax()]
    print(comparison.to_string(index=False))
    print()
    print(
        "Best dropout by validation Macro-F1:",
        f"{best['dropout']:.1f}",
        f"({best['best_validation_macro_f1']:.4f})",
    )
    print("Saved:", csv_path)
    print("Saved:", figure_path)


if __name__ == "__main__":
    main()
