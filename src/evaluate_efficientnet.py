import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from tqdm import tqdm

from data_setup import BATCH_SIZE, create_dataloaders
from efficientnet_models import (
    create_efficientnet_b0_transfer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Kaydedilmiş EfficientNet-B0 modelini "
            "kilitli "
            "PlantVillage test kümesinde değerlendirir."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Değerlendirilecek *_best.pt checkpoint yolu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Test mini-batch boyutu",
    )

    return parser.parse_args()


def _validate_arguments(arguments):
    if arguments.batch_size <= 0:
        raise ValueError(
            "Batch size pozitif olmalıdır."
        )

    if not arguments.checkpoint.is_file():
        raise FileNotFoundError(
            "Checkpoint bulunamadı: "
            f"{arguments.checkpoint.resolve()}"
        )


def _display_name(class_name):
    return (
        class_name.replace("___", " - ")
        .replace("_", " ")
    )


def _run_name_from_checkpoint(checkpoint_path):
    run_name = checkpoint_path.stem

    if run_name.endswith("_best"):
        run_name = run_name[:-5]

    return run_name


def _extract_test_paths(test_loader):
    test_subset = test_loader.dataset
    original_dataset = test_subset.dataset

    return [
        str(Path(original_dataset.samples[index][0]))
        for index in test_subset.indices
    ]


def _evaluate_model(
    model,
    test_loader,
    criterion,
    device,
):
    model.eval()
    use_amp = device.type == "cuda"
    running_loss = 0.0
    processed_samples = 0
    target_batches = []
    probability_batches = []

    progress_bar = tqdm(
        test_loader,
        desc="Test",
        unit="batch",
    )

    with torch.inference_mode():
        for images, labels in progress_bar:
            images = images.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            probabilities = torch.softmax(
                logits.float(),
                dim=1,
            )
            current_batch_size = labels.size(0)

            running_loss += (
                loss.item() * current_batch_size
            )
            processed_samples += current_batch_size
            target_batches.append(
                labels.cpu().numpy()
            )
            probability_batches.append(
                probabilities.cpu().numpy()
            )

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
            )

    if processed_samples == 0:
        raise RuntimeError(
            "Test DataLoader hiç örnek üretmedi."
        )

    targets = np.concatenate(target_batches)
    probabilities = np.concatenate(
        probability_batches
    )

    return (
        running_loss / processed_samples,
        targets,
        probabilities,
    )


def _calculate_metrics(
    targets,
    predictions,
    probabilities,
    class_indices,
):
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    accuracy = accuracy_score(
        targets,
        predictions,
    )
    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        targets,
        predictions,
        labels=class_indices,
        average="macro",
        zero_division=0,
    )
    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        targets,
        predictions,
        labels=class_indices,
        average="weighted",
        zero_division=0,
    )

    top3_indices = np.argpartition(
        probabilities,
        kth=-3,
        axis=1,
    )[:, -3:]
    top3_accuracy = np.mean(
        np.any(
            top3_indices == targets[:, None],
            axis=1,
        )
    )

    try:
        macro_roc_auc = roc_auc_score(
            targets,
            probabilities,
            labels=class_indices,
            multi_class="ovr",
            average="macro",
        )
        weighted_roc_auc = roc_auc_score(
            targets,
            probabilities,
            labels=class_indices,
            multi_class="ovr",
            average="weighted",
        )
        per_class_roc_auc = roc_auc_score(
            targets,
            probabilities,
            labels=class_indices,
            multi_class="ovr",
            average=None,
        )
    except ValueError as error:
        print(
            "Uyarı: ROC-AUC hesaplanamadı:",
            error,
        )
        macro_roc_auc = None
        weighted_roc_auc = None
        per_class_roc_auc = np.full(
            len(class_indices),
            np.nan,
        )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(weighted_f1),
        "top3_accuracy": float(top3_accuracy),
        "macro_roc_auc_ovr": (
            None
            if macro_roc_auc is None
            else float(macro_roc_auc)
        ),
        "weighted_roc_auc_ovr": (
            None
            if weighted_roc_auc is None
            else float(weighted_roc_auc)
        ),
        "per_class_roc_auc": (
            per_class_roc_auc
        ),
    }


def _create_classification_report(
    targets,
    predictions,
    class_names,
    per_class_roc_auc,
):
    from sklearn.metrics import (
        precision_recall_fscore_support,
    )

    class_indices = np.arange(len(class_names))
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            targets,
            predictions,
            labels=class_indices,
            average=None,
            zero_division=0,
        )
    )

    return pd.DataFrame(
        {
            "class_index": class_indices,
            "class_name": class_names,
            "display_name": [
                _display_name(name)
                for name in class_names
            ],
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support.astype(int),
            "roc_auc_ovr": (
                per_class_roc_auc
            ),
        }
    )


def _create_confusion_outputs(
    targets,
    predictions,
    class_names,
):
    from sklearn.metrics import confusion_matrix

    class_indices = np.arange(len(class_names))
    confusion_counts = confusion_matrix(
        targets,
        predictions,
        labels=class_indices,
    )
    row_totals = confusion_counts.sum(
        axis=1,
        keepdims=True,
    )
    confusion_normalized = np.divide(
        confusion_counts,
        row_totals,
        out=np.zeros_like(
            confusion_counts,
            dtype=float,
        ),
        where=row_totals != 0,
    )

    confusion_rows = []

    for true_index, predicted_index in zip(
        *np.where(
            ~np.eye(
                len(class_names),
                dtype=bool,
            )
        )
    ):
        count = int(
            confusion_counts[
                true_index,
                predicted_index,
            ]
        )

        if count == 0:
            continue

        confusion_rows.append(
            {
                "true_index": true_index,
                "true_class": (
                    class_names[true_index]
                ),
                "predicted_index": predicted_index,
                "predicted_class": (
                    class_names[predicted_index]
                ),
                "count": count,
                "true_class_error_rate": float(
                    confusion_normalized[
                        true_index,
                        predicted_index,
                    ]
                ),
            }
        )

    top_confusions = pd.DataFrame(
        confusion_rows
    )

    if not top_confusions.empty:
        top_confusions = (
            top_confusions.sort_values(
                ["count", "true_class_error_rate"],
                ascending=False,
            )
            .reset_index(drop=True)
        )

    return (
        confusion_counts,
        confusion_normalized,
        top_confusions,
    )


def _save_confusion_matrix_figure(
    confusion_normalized,
    class_names,
    output_path,
    title,
):
    display_names = [
        _display_name(name)
        for name in class_names
    ]

    figure, axis = plt.subplots(
        figsize=(18, 15),
    )
    sns.heatmap(
        confusion_normalized,
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=display_names,
        yticklabels=display_names,
        square=True,
        cbar_kws={
            "label": "Gerçek sınıf içindeki oran"
        },
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Tahmin edilen sınıf")
    axis.set_ylabel("Gerçek sınıf")
    axis.tick_params(
        axis="x",
        labelrotation=90,
        labelsize=6,
    )
    axis.tick_params(
        axis="y",
        labelrotation=0,
        labelsize=6,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def _save_per_class_f1_figure(
    classification_report,
    output_path,
    title,
):
    sorted_report = (
        classification_report.sort_values(
            "f1_score",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(12, 14),
    )
    bars = axis.barh(
        sorted_report["display_name"],
        sorted_report["f1_score"],
        color="#2C7FB8",
    )
    axis.bar_label(
        bars,
        fmt="%.2f",
        padding=3,
        fontsize=7,
    )
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("F1 skoru")
    axis.set_ylabel("Sınıf")
    axis.set_title(title)
    axis.grid(
        axis="x",
        alpha=0.25,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def _save_outputs(
    run_name,
    checkpoint_path,
    checkpoint,
    batch_size,
    device,
    class_names,
    test_paths,
    test_loss,
    targets,
    predictions,
    probabilities,
    metrics,
):
    evaluation_directory = (
        PROJECT_ROOT
        / "outputs"
        / "evaluation"
        / run_name
    )
    figures_directory = (
        PROJECT_ROOT
        / "outputs"
        / "figures"
    )
    evaluation_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    classification_report = (
        _create_classification_report(
            targets=targets,
            predictions=predictions,
            class_names=class_names,
            per_class_roc_auc=(
                metrics["per_class_roc_auc"]
            ),
        )
    )
    (
        confusion_counts,
        confusion_normalized,
        top_confusions,
    ) = _create_confusion_outputs(
        targets=targets,
        predictions=predictions,
        class_names=class_names,
    )

    confidence = probabilities.max(axis=1)
    true_class_probability = probabilities[
        np.arange(len(targets)),
        targets,
    ]
    predictions_frame = pd.DataFrame(
        {
            "image_path": test_paths,
            "true_index": targets,
            "true_class": [
                class_names[index]
                for index in targets
            ],
            "predicted_index": predictions,
            "predicted_class": [
                class_names[index]
                for index in predictions
            ],
            "confidence": confidence,
            "true_class_probability": (
                true_class_probability
            ),
            "correct": predictions == targets,
        }
    )

    summary_path = (
        evaluation_directory
        / "test_summary.json"
    )
    report_path = (
        evaluation_directory
        / "classification_report.csv"
    )
    confusion_counts_path = (
        evaluation_directory
        / "confusion_matrix_counts.csv"
    )
    confusion_normalized_path = (
        evaluation_directory
        / "confusion_matrix_normalized.csv"
    )
    predictions_path = (
        evaluation_directory
        / "predictions.csv"
    )
    top_confusions_path = (
        evaluation_directory
        / "top_confusions.csv"
    )
    confusion_figure_path = (
        figures_directory
        / f"{run_name}_test_confusion_matrix.png"
    )
    f1_figure_path = (
        figures_directory
        / f"{run_name}_test_per_class_f1.png"
    )

    classification_report.to_csv(
        report_path,
        index=False,
    )
    pd.DataFrame(
        confusion_counts,
        index=class_names,
        columns=class_names,
    ).to_csv(confusion_counts_path)
    pd.DataFrame(
        confusion_normalized,
        index=class_names,
        columns=class_names,
    ).to_csv(confusion_normalized_path)
    predictions_frame.to_csv(
        predictions_path,
        index=False,
    )
    top_confusions.to_csv(
        top_confusions_path,
        index=False,
    )

    summary = {
        "run_name": run_name,
        "model_name": checkpoint.get(
            "model_name",
            "efficientnet_b0_transfer",
        ),
        "checkpoint_path": str(
            checkpoint_path.resolve()
        ),
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "selection_metric": checkpoint.get(
            "selection_metric",
            "validation_macro_f1",
        ),
        "checkpoint_validation_loss": float(
            checkpoint["validation_loss"]
        ),
        "checkpoint_validation_accuracy": (
            float(
                checkpoint[
                    "validation_accuracy"
                ]
            )
        ),
        "checkpoint_validation_macro_f1": (
            float(
                checkpoint[
                    "validation_macro_f1"
                ]
            )
        ),
        "test_used_for_model_selection": False,
        "test_examples": int(len(targets)),
        "num_classes": int(len(class_names)),
        "test_loss": float(test_loss),
        "test_accuracy": metrics["accuracy"],
        "test_macro_precision": (
            metrics["macro_precision"]
        ),
        "test_macro_recall": (
            metrics["macro_recall"]
        ),
        "test_macro_f1": metrics["macro_f1"],
        "test_weighted_precision": (
            metrics["weighted_precision"]
        ),
        "test_weighted_recall": (
            metrics["weighted_recall"]
        ),
        "test_weighted_f1": (
            metrics["weighted_f1"]
        ),
        "test_top3_accuracy": (
            metrics["top3_accuracy"]
        ),
        "test_macro_roc_auc_ovr": (
            metrics["macro_roc_auc_ovr"]
        ),
        "test_weighted_roc_auc_ovr": (
            metrics["weighted_roc_auc_ovr"]
        ),
        "batch_size": int(batch_size),
        "device": str(device),
        "amp": device.type == "cuda",
        "pytorch_version": torch.__version__,
        "class_names": class_names,
        "output_files": {
            "classification_report": str(
                report_path.resolve()
            ),
            "confusion_matrix_counts": str(
                confusion_counts_path.resolve()
            ),
            "confusion_matrix_normalized": str(
                confusion_normalized_path.resolve()
            ),
            "predictions": str(
                predictions_path.resolve()
            ),
            "top_confusions": str(
                top_confusions_path.resolve()
            ),
            "confusion_matrix_figure": str(
                confusion_figure_path.resolve()
            ),
            "per_class_f1_figure": str(
                f1_figure_path.resolve()
            ),
        },
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            summary,
            summary_file,
            ensure_ascii=False,
            indent=2,
        )

    _save_confusion_matrix_figure(
        confusion_normalized=(
            confusion_normalized
        ),
        class_names=class_names,
        output_path=confusion_figure_path,
        title=(
            "EfficientNet-B0 - Normalize Test "
            "Confusion Matrix"
        ),
    )
    _save_per_class_f1_figure(
        classification_report=(
            classification_report
        ),
        output_path=f1_figure_path,
        title=(
            "EfficientNet-B0 - Test Sınıf Bazlı "
            "F1 Skorları"
        ),
    )

    return {
        "summary": summary_path,
        "classification_report": report_path,
        "confusion_matrix": (
            confusion_figure_path
        ),
        "per_class_f1": f1_figure_path,
    }


def _print_optional_metric(label, value):
    if value is None:
        print(f"{label}: Hesaplanamadı")
    else:
        print(f"{label}: {value:.4f}")


def main():
    arguments = parse_arguments()
    arguments.checkpoint = (
        arguments.checkpoint.expanduser()
    )
    _validate_arguments(arguments)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Bu dosya yalnızca kullanıcının kendi eğitiminden çıkan,
    # güvenilir yerel checkpoint için tasarlanmıştır.
    checkpoint = torch.load(
        arguments.checkpoint,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("model_name") not in (
        None,
        "efficientnet_b0_transfer",
    ):
        raise ValueError(
            "Bu değerlendirme dosyası yalnızca "
            "efficientnet_b0_transfer checkpoint'i "
            "kabul eder."
        )

    checkpoint_class_names = checkpoint.get(
        "class_names"
    )

    if not checkpoint_class_names:
        raise KeyError(
            "Checkpoint içinde class_names yok."
        )

    (
        _,
        _,
        test_loader,
        dataset_class_names,
    ) = create_dataloaders(
        batch_size=arguments.batch_size,
        num_workers=0,
    )

    if (
        list(checkpoint_class_names)
        != list(dataset_class_names)
    ):
        raise RuntimeError(
            "Checkpoint ve veri seti sınıf "
            "indeksleri eşleşmiyor."
        )

    class_names = list(checkpoint_class_names)
    dropout_rate = float(
        checkpoint.get("dropout_rate", 0.3)
    )
    model = create_efficientnet_b0_transfer(
        num_classes=len(class_names),
        dropout_rate=dropout_rate,
        pretrained=False,
    ).to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    criterion = nn.CrossEntropyLoss()
    test_paths = _extract_test_paths(
        test_loader
    )
    run_name = _run_name_from_checkpoint(
        arguments.checkpoint
    )

    print("=" * 65)
    print("EFFICIENTNET-B0 KİLİTLİ TEST DEĞERLENDİRMESİ")
    print("=" * 65)
    print("Run adı:", run_name)
    print("Checkpoint:", arguments.checkpoint)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Cihaz:", device)
    print("Test örneği:", len(test_loader.dataset))
    print("Sınıf sayısı:", len(class_names))
    print("Batch size:", arguments.batch_size)
    print("Test worker: 0")
    print("AMP:", device.type == "cuda")
    print("Test kümesi model seçiminde kullanılmadı.")
    print()

    test_loss, targets, probabilities = (
        _evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )
    )
    predictions = probabilities.argmax(axis=1)
    class_indices = np.arange(
        len(class_names)
    )
    metrics = _calculate_metrics(
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        class_indices=class_indices,
    )
    output_paths = _save_outputs(
        run_name=run_name,
        checkpoint_path=arguments.checkpoint,
        checkpoint=checkpoint,
        batch_size=arguments.batch_size,
        device=device,
        class_names=class_names,
        test_paths=test_paths,
        test_loss=test_loss,
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        metrics=metrics,
    )

    print()
    print("=" * 65)
    print("TEST SONUÇLARI")
    print("=" * 65)
    print(f"Loss: {test_loss:.4f}")
    print(
        "Accuracy:",
        f"{metrics['accuracy'] * 100:.2f}%",
    )
    print(
        "Macro Precision:",
        f"{metrics['macro_precision']:.4f}",
    )
    print(
        "Macro Recall:",
        f"{metrics['macro_recall']:.4f}",
    )
    print(
        "Macro-F1:",
        f"{metrics['macro_f1']:.4f}",
    )
    print(
        "Weighted-F1:",
        f"{metrics['weighted_f1']:.4f}",
    )
    print(
        "Top-3 Accuracy:",
        f"{metrics['top3_accuracy'] * 100:.2f}%",
    )
    _print_optional_metric(
        "Macro ROC-AUC (OvR)",
        metrics["macro_roc_auc_ovr"],
    )
    _print_optional_metric(
        "Weighted ROC-AUC (OvR)",
        metrics["weighted_roc_auc_ovr"],
    )
    print()
    print("Özet:", output_paths["summary"])
    print(
        "Sınıf raporu:",
        output_paths["classification_report"],
    )
    print(
        "Confusion matrix:",
        output_paths["confusion_matrix"],
    )
    print(
        "Sınıf F1 grafiği:",
        output_paths["per_class_f1"],
    )


if __name__ == "__main__":
    main()
