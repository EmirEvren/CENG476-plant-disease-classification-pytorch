from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def display_name(class_name: str) -> str:
    return class_name.replace("___", " - ").replace("_", " ")


def run_name_from_checkpoint(checkpoint_path: Path) -> str:
    run_name = checkpoint_path.stem
    if run_name.endswith("_best"):
        run_name = run_name[:-5]
    return run_name


def extract_relative_paths(data_loader):
    dataset = data_loader.dataset
    if not hasattr(dataset, "frame"):
        raise RuntimeError(
            "Official evaluation expects ManifestImageDataset with a frame attribute."
        )
    return dataset.frame["relative_path"].astype(str).tolist()


def collect_model_probabilities(model, data_loader, criterion, device, phase="Locked official test"):
    model.eval()
    use_amp = device.type == "cuda"
    running_loss = 0.0
    processed_samples = 0
    target_batches = []
    probability_batches = []

    progress_bar = tqdm(data_loader, desc=phase, unit="batch")

    with torch.inference_mode():
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            probabilities = torch.softmax(logits.float(), dim=1)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            processed_samples += batch_size
            target_batches.append(labels.cpu().numpy())
            probability_batches.append(probabilities.cpu().numpy())

    if processed_samples == 0:
        raise RuntimeError("Evaluation DataLoader produced no samples.")

    return (
        running_loss / processed_samples,
        np.concatenate(target_batches),
        np.concatenate(probability_batches),
    )


def calculate_metrics(targets, probabilities, num_classes):
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    predictions = probabilities.argmax(axis=1)
    labels = np.arange(num_classes)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    top3_indices = np.argpartition(probabilities, kth=-3, axis=1)[:, -3:]
    top3_accuracy = np.mean(np.any(top3_indices == targets[:, None], axis=1))

    try:
        macro_auc = roc_auc_score(
            targets,
            probabilities,
            labels=labels,
            multi_class="ovr",
            average="macro",
        )
        weighted_auc = roc_auc_score(
            targets,
            probabilities,
            labels=labels,
            multi_class="ovr",
            average="weighted",
        )
        per_class_auc = roc_auc_score(
            targets,
            probabilities,
            labels=labels,
            multi_class="ovr",
            average=None,
        )
    except ValueError:
        macro_auc = None
        weighted_auc = None
        per_class_auc = np.full(num_classes, np.nan)

    return {
        "predictions": predictions,
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "top3_accuracy": float(top3_accuracy),
        "macro_roc_auc_ovr": None if macro_auc is None else float(macro_auc),
        "weighted_roc_auc_ovr": None if weighted_auc is None else float(weighted_auc),
        "per_class_roc_auc": per_class_auc,
    }


def negative_log_likelihood(targets, probabilities):
    true_probabilities = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(true_probabilities, 1e-12, 1.0)).mean())


def save_evaluation_outputs(
    *,
    run_name,
    model_name,
    class_names,
    relative_paths,
    targets,
    probabilities,
    metrics,
    test_loss,
    summary_extra=None,
    figure_prefix=None,
):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    output_dir = PROJECT_ROOT / "outputs" / "evaluation" / run_name
    figure_dir = PROJECT_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    labels = np.arange(len(class_names))
    predictions = metrics["predictions"]
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )

    classification_report = pd.DataFrame(
        {
            "class_index": labels,
            "class_name": class_names,
            "display_name": [display_name(name) for name in class_names],
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support.astype(int),
            "roc_auc_ovr": metrics["per_class_roc_auc"],
        }
    )

    confusion_counts = confusion_matrix(targets, predictions, labels=labels)
    row_totals = confusion_counts.sum(axis=1, keepdims=True)
    confusion_normalized = np.divide(
        confusion_counts,
        row_totals,
        out=np.zeros_like(confusion_counts, dtype=float),
        where=row_totals != 0,
    )

    confusion_rows = []
    for true_index in range(len(class_names)):
        for predicted_index in range(len(class_names)):
            if true_index == predicted_index:
                continue
            count = int(confusion_counts[true_index, predicted_index])
            if count == 0:
                continue
            confusion_rows.append(
                {
                    "true_index": true_index,
                    "true_class": class_names[true_index],
                    "predicted_index": predicted_index,
                    "predicted_class": class_names[predicted_index],
                    "count": count,
                    "true_class_error_rate": float(
                        confusion_normalized[true_index, predicted_index]
                    ),
                }
            )

    top_confusions = pd.DataFrame(confusion_rows)
    if not top_confusions.empty:
        top_confusions = top_confusions.sort_values(
            ["count", "true_class_error_rate"],
            ascending=False,
        ).reset_index(drop=True)

    confidence = probabilities.max(axis=1)
    true_class_probability = probabilities[np.arange(len(targets)), targets]
    prediction_frame = pd.DataFrame(
        {
            "image_path": relative_paths,
            "true_index": targets,
            "true_class": [class_names[index] for index in targets],
            "predicted_index": predictions,
            "predicted_class": [class_names[index] for index in predictions],
            "confidence": confidence,
            "true_class_probability": true_class_probability,
            "correct": predictions == targets,
        }
    )

    report_path = output_dir / "classification_report.csv"
    counts_path = output_dir / "confusion_matrix_counts.csv"
    normalized_path = output_dir / "confusion_matrix_normalized.csv"
    predictions_path = output_dir / "predictions.csv"
    top_confusions_path = output_dir / "top_confusions.csv"
    summary_path = output_dir / "test_summary.json"

    classification_report.to_csv(report_path, index=False)
    pd.DataFrame(confusion_counts, index=class_names, columns=class_names).to_csv(counts_path)
    pd.DataFrame(confusion_normalized, index=class_names, columns=class_names).to_csv(
        normalized_path
    )
    prediction_frame.to_csv(predictions_path, index=False)
    top_confusions.to_csv(top_confusions_path, index=False)

    prefix = figure_prefix or run_name
    confusion_figure_path = figure_dir / f"{prefix}_test_confusion_matrix.png"
    f1_figure_path = figure_dir / f"{prefix}_test_per_class_f1.png"

    figure, axis = plt.subplots(figsize=(15, 13))
    image = axis.imshow(confusion_normalized, vmin=0, vmax=1, aspect="auto")
    axis.set_title(f"{model_name} - Locked Official Test Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(labels)
    axis.set_yticks(labels)
    axis.set_xticklabels([display_name(name) for name in class_names], rotation=90, fontsize=6)
    axis.set_yticklabels([display_name(name) for name in class_names], fontsize=6)
    figure.colorbar(image, ax=axis, label="Rate within true class")
    figure.tight_layout()
    figure.savefig(confusion_figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    sorted_report = classification_report.sort_values("f1_score", ascending=True)
    figure, axis = plt.subplots(figsize=(12, 14))
    bars = axis.barh(sorted_report["display_name"], sorted_report["f1_score"])
    axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=7)
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("F1 score")
    axis.set_ylabel("Class")
    axis.set_title(f"{model_name} - Locked Official Test Per-Class F1")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(f1_figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "run_name": run_name,
        "model_name": model_name,
        "data_protocol": "conservative_official_leaf_preserving_test",
        "test_examples": int(len(targets)),
        "num_classes": int(len(class_names)),
        "test_loss": float(test_loss),
        "test_accuracy": metrics["accuracy"],
        "test_macro_precision": metrics["macro_precision"],
        "test_macro_recall": metrics["macro_recall"],
        "test_macro_f1": metrics["macro_f1"],
        "test_weighted_precision": metrics["weighted_precision"],
        "test_weighted_recall": metrics["weighted_recall"],
        "test_weighted_f1": metrics["weighted_f1"],
        "test_top3_accuracy": metrics["top3_accuracy"],
        "test_macro_roc_auc_ovr": metrics["macro_roc_auc_ovr"],
        "test_weighted_roc_auc_ovr": metrics["weighted_roc_auc_ovr"],
        "errors": int((predictions != targets).sum()),
        "output_files": {
            "classification_report": str(report_path.resolve()),
            "confusion_matrix_counts": str(counts_path.resolve()),
            "confusion_matrix_normalized": str(normalized_path.resolve()),
            "predictions": str(predictions_path.resolve()),
            "top_confusions": str(top_confusions_path.resolve()),
            "confusion_matrix_figure": str(confusion_figure_path.resolve()),
            "per_class_f1_figure": str(f1_figure_path.resolve()),
        },
    }
    if summary_extra:
        summary.update(summary_extra)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary, summary_path


def print_metrics(metrics, test_loss, total_examples):
    errors = int((metrics["predictions"] != metrics.pop("_targets", np.array([]))).sum()) if "_targets" in metrics else None
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
    print(f"Top-3 Accuracy: {metrics['top3_accuracy'] * 100:.2f}%")
    if metrics["macro_roc_auc_ovr"] is not None:
        print(f"Macro ROC-AUC (OvR): {metrics['macro_roc_auc_ovr']:.6f}")
    if errors is not None:
        print(f"Errors: {errors} / {total_examples}")
