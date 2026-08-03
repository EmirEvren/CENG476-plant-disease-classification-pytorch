import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_SPECS = [
    {
        "label": "Baseline CNN",
        "run_name": "baseline_b64_lr5e4_full",
        "color": "#4C78A8",
    },
    {
        "label": "ResNet18",
        "run_name": (
            "resnet18_b32_blr1e4_hlr5e4_full"
        ),
        "color": "#F58518",
    },
    {
        "label": "EfficientNet-B0",
        "run_name": (
            "efficientnet_b0_b32_blr1e4_"
            "hlr5e4_full"
        ),
        "color": "#54A24B",
    },
]


def _read_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        return json.load(json_file)


def _resolve_run_files(run_name):
    return {
        "config": (
            PROJECT_ROOT
            / "outputs"
            / "experiments"
            / f"{run_name}_config.json"
        ),
        "history": (
            PROJECT_ROOT
            / "outputs"
            / "histories"
            / f"{run_name}_history.csv"
        ),
        "test_summary": (
            PROJECT_ROOT
            / "outputs"
            / "evaluation"
            / run_name
            / "test_summary.json"
        ),
    }


def _validate_files(model_specs):
    missing_files = []

    for model_spec in model_specs:
        paths = _resolve_run_files(
            model_spec["run_name"]
        )

        for file_type, path in paths.items():
            if not path.is_file():
                missing_files.append(
                    (
                        model_spec["label"],
                        file_type,
                        path,
                    )
                )

    if missing_files:
        lines = [
            f"- {label} / {file_type}: {path}"
            for label, file_type, path
            in missing_files
        ]
        raise FileNotFoundError(
            "Karşılaştırma için gereken dosyalar "
            "bulunamadı:\n"
            + "\n".join(lines)
        )


def _parameter_count(config):
    parameter_count = config.get(
        "total_parameters"
    )

    if parameter_count is None:
        parameter_count = config.get(
            "trainable_parameters"
        )

    if parameter_count is None:
        raise KeyError(
            "Config içinde parametre sayısı yok."
        )

    return int(parameter_count)


def _create_record(
    model_spec,
    config,
    history,
    test_summary,
):
    parameter_count = _parameter_count(config)
    training_minutes = (
        history["elapsed_seconds"].sum() / 60
    )
    peak_gpu_memory_mb = (
        history["peak_gpu_memory_mb"].max()
    )
    test_accuracy = float(
        test_summary["test_accuracy"]
    )
    test_examples = int(
        test_summary["test_examples"]
    )
    test_errors = round(
        test_examples * (1 - test_accuracy)
    )

    return {
        "model": model_spec["label"],
        "run_name": model_spec["run_name"],
        "parameters": parameter_count,
        "parameters_million": (
            parameter_count / 1_000_000
        ),
        "theoretical_fp32_size_mb": (
            parameter_count * 4 / 1024**2
        ),
        "batch_size": int(
            config["batch_size"]
        ),
        "epochs_completed": int(
            len(history)
        ),
        "best_epoch": int(
            test_summary["checkpoint_epoch"]
        ),
        "validation_loss": float(
            test_summary[
                "checkpoint_validation_loss"
            ]
        ),
        "validation_accuracy": float(
            test_summary[
                "checkpoint_validation_accuracy"
            ]
        ),
        "validation_macro_f1": float(
            test_summary[
                "checkpoint_validation_macro_f1"
            ]
        ),
        "test_examples": test_examples,
        "test_errors": int(test_errors),
        "test_loss": float(
            test_summary["test_loss"]
        ),
        "test_accuracy": test_accuracy,
        "test_macro_precision": float(
            test_summary[
                "test_macro_precision"
            ]
        ),
        "test_macro_recall": float(
            test_summary["test_macro_recall"]
        ),
        "test_macro_f1": float(
            test_summary["test_macro_f1"]
        ),
        "test_weighted_f1": float(
            test_summary["test_weighted_f1"]
        ),
        "test_top3_accuracy": float(
            test_summary["test_top3_accuracy"]
        ),
        "test_macro_roc_auc_ovr": (
            test_summary[
                "test_macro_roc_auc_ovr"
            ]
        ),
        "training_minutes": float(
            training_minutes
        ),
        "peak_gpu_memory_mb": float(
            peak_gpu_memory_mb
        ),
    }


def _load_results():
    _validate_files(MODEL_SPECS)
    records = []
    histories = {}

    for model_spec in MODEL_SPECS:
        run_name = model_spec["run_name"]
        paths = _resolve_run_files(run_name)
        config = _read_json(paths["config"])
        history = pd.read_csv(paths["history"])
        test_summary = _read_json(
            paths["test_summary"]
        )

        if (
            test_summary["run_name"]
            != run_name
        ):
            raise RuntimeError(
                "Test özeti run adı eşleşmiyor: "
                f"{run_name}"
            )

        records.append(
            _create_record(
                model_spec=model_spec,
                config=config,
                history=history,
                test_summary=test_summary,
            )
        )
        histories[model_spec["label"]] = (
            history
        )

    return pd.DataFrame(records), histories


def _add_bar_labels(axis, bars):
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _plot_test_performance(
    comparison,
    output_path,
):
    metrics = [
        (
            "test_accuracy",
            "Test Accuracy (%)",
        ),
        (
            "test_macro_f1",
            "Test Macro-F1 (%)",
        ),
        (
            "test_weighted_f1",
            "Test Weighted-F1 (%)",
        ),
    ]
    colors = [
        model_spec["color"]
        for model_spec in MODEL_SPECS
    ]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )

    for axis, (column, title) in zip(
        axes,
        metrics,
    ):
        values = comparison[column] * 100
        bars = axis.bar(
            comparison["model"],
            values,
            color=colors,
        )
        axis.set_title(title)
        axis.set_ylim(0, 105)
        axis.set_ylabel("%")
        axis.tick_params(
            axis="x",
            rotation=15,
        )
        axis.grid(
            axis="y",
            alpha=0.25,
        )
        _add_bar_labels(axis, bars)

    figure.suptitle(
        "PlantVillage Nihai Model Performansları"
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_efficiency(
    comparison,
    output_path,
):
    color_by_model = {
        model_spec["label"]: model_spec["color"]
        for model_spec in MODEL_SPECS
    }
    figure, axis = plt.subplots(
        figsize=(9, 6),
    )

    for _, row in comparison.iterrows():
        axis.scatter(
            row["parameters_million"],
            row["test_macro_f1"] * 100,
            s=180,
            color=color_by_model[row["model"]],
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )
        axis.annotate(
            (
                f"{row['model']}\n"
                f"{row['parameters_million']:.2f}M"
            ),
            (
                row["parameters_million"],
                row["test_macro_f1"] * 100,
            ),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )

    axis.set_xlabel(
        "Model parametre sayısı (milyon)"
    )
    axis.set_ylabel("Test Macro-F1 (%)")
    axis.set_title(
        "Model Boyutu ve Sınıflandırma Performansı"
    )
    axis.set_ylim(75, 101)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_validation_curves(
    histories,
    output_path,
):
    color_by_model = {
        model_spec["label"]: model_spec["color"]
        for model_spec in MODEL_SPECS
    }
    figure, axis = plt.subplots(
        figsize=(10, 6),
    )

    for model_name, history in histories.items():
        axis.plot(
            history["epoch"],
            history["validation_macro_f1"] * 100,
            marker="o",
            linewidth=2,
            label=model_name,
            color=color_by_model[model_name],
        )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation Macro-F1 (%)")
    axis.set_title(
        "Modellerin Validation Macro-F1 Eğrileri"
    )
    axis.set_ylim(0, 101)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def _percentage(value):
    return f"{value * 100:.2f}%"


def _write_markdown_report(
    comparison,
    output_path,
):
    winner = comparison.loc[
        comparison["test_macro_f1"].idxmax()
    ]
    resnet = comparison.loc[
        comparison["model"] == "ResNet18"
    ].iloc[0]
    efficientnet = comparison.loc[
        comparison["model"] == "EfficientNet-B0"
    ].iloc[0]
    parameter_reduction = (
        1
        - efficientnet["parameters"]
        / resnet["parameters"]
    )

    lines = [
        "# PlantVillage Model Comparison",
        "",
        (
            "| Model | Parameters | Best Epoch | "
            "Val Macro-F1 | Test Accuracy | "
            "Test Macro-F1 | Weighted-F1 |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for _, row in comparison.iterrows():
        lines.append(
            "| "
            f"{row['model']} | "
            f"{int(row['parameters']):,} | "
            f"{int(row['best_epoch'])} | "
            f"{_percentage(row['validation_macro_f1'])} | "
            f"{_percentage(row['test_accuracy'])} | "
            f"{_percentage(row['test_macro_f1'])} | "
            f"{_percentage(row['test_weighted_f1'])} |"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                f"The best model is **{winner['model']}** "
                f"with {_percentage(winner['test_accuracy'])} "
                "test accuracy and "
                f"{winner['test_macro_f1']:.4f} "
                "test Macro-F1."
            ),
            (
                "EfficientNet-B0 uses "
                f"{parameter_reduction * 100:.1f}% fewer "
                "parameters than ResNet18."
            ),
            (
                "The test set was not used for training, "
                "hyperparameter selection, or checkpoint "
                "selection."
            ),
            "",
            "## Limitation",
            "",
            (
                "PlantVillage contains mostly controlled "
                "background images. Performance on field "
                "images with complex backgrounds, lighting "
                "changes, and occlusion may be lower."
            ),
        ]
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _print_comparison(comparison):
    display_columns = [
        "model",
        "parameters",
        "best_epoch",
        "validation_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_errors",
    ]
    display = comparison[
        display_columns
    ].copy()

    for column in (
        "validation_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
    ):
        display[column] = (
            display[column] * 100
        ).map(lambda value: f"{value:.2f}%")

    display["parameters"] = display[
        "parameters"
    ].map(lambda value: f"{value:,}")

    print("=" * 100)
    print("NİHAİ MODEL KARŞILAŞTIRMASI")
    print("=" * 100)
    print(display.to_string(index=False))


def main():
    comparison, histories = _load_results()
    comparison_directory = (
        PROJECT_ROOT
        / "outputs"
        / "comparison"
    )
    figures_directory = (
        PROJECT_ROOT
        / "outputs"
        / "figures"
    )
    comparison_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        comparison_directory
        / "model_comparison.csv"
    )
    markdown_path = (
        comparison_directory
        / "model_comparison.md"
    )
    performance_figure_path = (
        figures_directory
        / "final_model_comparison.png"
    )
    efficiency_figure_path = (
        figures_directory
        / "final_model_efficiency.png"
    )
    validation_figure_path = (
        figures_directory
        / "final_validation_macro_f1.png"
    )

    comparison.to_csv(
        csv_path,
        index=False,
    )
    _write_markdown_report(
        comparison=comparison,
        output_path=markdown_path,
    )
    _plot_test_performance(
        comparison=comparison,
        output_path=performance_figure_path,
    )
    _plot_efficiency(
        comparison=comparison,
        output_path=efficiency_figure_path,
    )
    _plot_validation_curves(
        histories=histories,
        output_path=validation_figure_path,
    )
    _print_comparison(comparison)

    winner = comparison.loc[
        comparison["test_macro_f1"].idxmax()
    ]

    print()
    print("Kazanan model:", winner["model"])
    print(
        "Test Accuracy:",
        _percentage(winner["test_accuracy"]),
    )
    print(
        "Test Macro-F1:",
        f"{winner['test_macro_f1']:.4f}",
    )
    print()
    print("CSV:", csv_path)
    print("Markdown rapor:", markdown_path)
    print(
        "Performans grafiği:",
        performance_figure_path,
    )
    print(
        "Verimlilik grafiği:",
        efficiency_figure_path,
    )
    print(
        "Validation grafiği:",
        validation_figure_path,
    )


if __name__ == "__main__":
    main()
