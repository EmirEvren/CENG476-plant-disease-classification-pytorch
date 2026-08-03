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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from tqdm import tqdm

from data_setup import BATCH_SIZE, create_dataloaders
from efficientnet_models import (
    create_efficientnet_b0_transfer,
)
from transfer_models import create_resnet18_transfer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESNET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "resnet18_b32_blr1e4_hlr5e4_full_best.pt"
)
DEFAULT_EFFICIENTNET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "efficientnet_b0_b32_blr1e4_hlr5e4_full_best.pt"
)
EFFICIENTNET_WEIGHT_CANDIDATES = (
    0.25,
    0.50,
    0.75,
)
RUN_NAME = "resnet18_efficientnet_soft_voting"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "ResNet18 ve EfficientNet-B0 modellerini "
            "validation ile seçilen ağırlıklı soft-voting "
            "ensemble olarak değerlendirir."
        )
    )
    parser.add_argument(
        "--resnet-checkpoint",
        type=Path,
        default=DEFAULT_RESNET_CHECKPOINT,
    )
    parser.add_argument(
        "--efficientnet-checkpoint",
        type=Path,
        default=DEFAULT_EFFICIENTNET_CHECKPOINT,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    return parser.parse_args()


def validate_arguments(arguments):
    if arguments.batch_size <= 0:
        raise ValueError(
            "Batch size pozitif olmalıdır."
        )

    for checkpoint_path in (
        arguments.resnet_checkpoint,
        arguments.efficientnet_checkpoint,
    ):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint bulunamadı: {checkpoint_path}"
            )


def display_name(class_name):
    return (
        class_name.replace("___", " - ")
        .replace("_", " ")
    )


def load_checkpoints(arguments):
    resnet_checkpoint = torch.load(
        arguments.resnet_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    efficientnet_checkpoint = torch.load(
        arguments.efficientnet_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if (
        resnet_checkpoint.get("model_name")
        != "resnet18_transfer"
    ):
        raise ValueError(
            "ResNet checkpoint model_name alanı geçersiz."
        )
    if (
        efficientnet_checkpoint.get("model_name")
        != "efficientnet_b0_transfer"
    ):
        raise ValueError(
            "EfficientNet checkpoint model_name alanı "
            "geçersiz."
        )

    resnet_classes = list(
        resnet_checkpoint["class_names"]
    )
    efficientnet_classes = list(
        efficientnet_checkpoint["class_names"]
    )

    if resnet_classes != efficientnet_classes:
        raise RuntimeError(
            "İki checkpoint'in sınıf indeksleri "
            "eşleşmiyor."
        )

    return (
        resnet_checkpoint,
        efficientnet_checkpoint,
        resnet_classes,
    )


def create_models(
    resnet_checkpoint,
    efficientnet_checkpoint,
    num_classes,
    device,
):
    resnet_model = create_resnet18_transfer(
        num_classes=num_classes,
        dropout_rate=float(
            resnet_checkpoint.get(
                "dropout_rate",
                0.3,
            )
        ),
        pretrained=False,
    )
    efficientnet_model = (
        create_efficientnet_b0_transfer(
            num_classes=num_classes,
            dropout_rate=float(
                efficientnet_checkpoint.get(
                    "dropout_rate",
                    0.3,
                )
            ),
            pretrained=False,
        )
    )

    resnet_model.load_state_dict(
        resnet_checkpoint["model_state_dict"],
        strict=True,
    )
    efficientnet_model.load_state_dict(
        efficientnet_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    return (
        resnet_model.to(device).eval(),
        efficientnet_model.to(device).eval(),
    )


def collect_probabilities(
    resnet_model,
    efficientnet_model,
    data_loader,
    device,
    phase,
):
    use_amp = device.type == "cuda"
    all_targets = []
    resnet_probabilities = []
    efficientnet_probabilities = []

    progress_bar = tqdm(
        data_loader,
        desc=phase,
        unit="batch",
    )

    with torch.inference_mode():
        for images, labels in progress_bar:
            images = images.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                resnet_logits = resnet_model(images)
                efficientnet_logits = (
                    efficientnet_model(images)
                )

            all_targets.append(labels.numpy())
            resnet_probabilities.append(
                torch.softmax(
                    resnet_logits.float(),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            efficientnet_probabilities.append(
                torch.softmax(
                    efficientnet_logits.float(),
                    dim=1,
                )
                .cpu()
                .numpy()
            )

    return (
        np.concatenate(all_targets),
        np.concatenate(resnet_probabilities),
        np.concatenate(
            efficientnet_probabilities
        ),
    )


def negative_log_likelihood(
    targets,
    probabilities,
):
    true_probabilities = probabilities[
        np.arange(len(targets)),
        targets,
    ]
    return float(
        -np.log(
            np.clip(
                true_probabilities,
                1e-12,
                1.0,
            )
        ).mean()
    )


def basic_metrics(targets, probabilities):
    predictions = probabilities.argmax(axis=1)
    return {
        "loss": negative_log_likelihood(
            targets,
            probabilities,
        ),
        "accuracy": float(
            accuracy_score(
                targets,
                predictions,
            )
        ),
        "macro_f1": float(
            precision_recall_fscore_support(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )[2]
        ),
    }


def combine_probabilities(
    resnet_probabilities,
    efficientnet_probabilities,
    efficientnet_weight,
):
    return (
        (1.0 - efficientnet_weight)
        * resnet_probabilities
        + efficientnet_weight
        * efficientnet_probabilities
    )


def select_ensemble_weight(
    targets,
    resnet_probabilities,
    efficientnet_probabilities,
):
    records = []

    for efficientnet_weight in (
        0.0,
        *EFFICIENTNET_WEIGHT_CANDIDATES,
        1.0,
    ):
        probabilities = combine_probabilities(
            resnet_probabilities,
            efficientnet_probabilities,
            efficientnet_weight,
        )
        metrics = basic_metrics(
            targets,
            probabilities,
        )
        records.append(
            {
                "efficientnet_weight": (
                    efficientnet_weight
                ),
                "resnet_weight": (
                    1.0 - efficientnet_weight
                ),
                "eligible_for_selection": (
                    efficientnet_weight
                    in EFFICIENTNET_WEIGHT_CANDIDATES
                ),
                "validation_loss": metrics["loss"],
                "validation_accuracy": (
                    metrics["accuracy"]
                ),
                "validation_macro_f1": (
                    metrics["macro_f1"]
                ),
            }
        )

    search_frame = pd.DataFrame(records)
    eligible_frame = search_frame[
        search_frame["eligible_for_selection"]
    ]
    best_row = eligible_frame.sort_values(
        [
            "validation_macro_f1",
            "validation_loss",
        ],
        ascending=[False, True],
    ).iloc[0]

    return (
        float(best_row["efficientnet_weight"]),
        search_frame,
    )


def calculate_full_metrics(
    targets,
    probabilities,
    num_classes,
):
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(num_classes)

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
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
        labels=labels,
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
            labels=labels,
            multi_class="ovr",
            average="macro",
        )
        weighted_roc_auc = roc_auc_score(
            targets,
            probabilities,
            labels=labels,
            multi_class="ovr",
            average="weighted",
        )
        per_class_roc_auc = roc_auc_score(
            targets,
            probabilities,
            labels=labels,
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
            num_classes,
            np.nan,
        )

    return {
        "loss": negative_log_likelihood(
            targets,
            probabilities,
        ),
        "accuracy": float(
            accuracy_score(
                targets,
                predictions,
            )
        ),
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
        "predictions": predictions,
    }


def extract_paths(data_loader):
    subset = data_loader.dataset
    original_dataset = subset.dataset
    return [
        str(Path(original_dataset.samples[index][0]))
        for index in subset.indices
    ]


def save_confusion_figure(
    normalized_confusion,
    class_names,
    output_path,
):
    figure, axis = plt.subplots(
        figsize=(18, 15),
    )
    sns.heatmap(
        normalized_confusion,
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=[
            display_name(name)
            for name in class_names
        ],
        yticklabels=[
            display_name(name)
            for name in class_names
        ],
        cbar_kws={
            "label": "Gerçek sınıf içindeki oran"
        },
        ax=axis,
    )
    axis.set_title(
        "ResNet18 + EfficientNet-B0 Ensemble - "
        "Normalize Test Confusion Matrix"
    )
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


def save_f1_figure(
    report_frame,
    output_path,
):
    sorted_frame = report_frame.sort_values(
        "f1_score",
        ascending=True,
    )
    figure, axis = plt.subplots(
        figsize=(12, 14),
    )
    bars = axis.barh(
        sorted_frame["display_name"],
        sorted_frame["f1_score"],
        color="#6A51A3",
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
    axis.set_title(
        "ResNet18 + EfficientNet-B0 Ensemble - "
        "Test Sınıf Bazlı F1 Skorları"
    )
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


def save_outputs(
    arguments,
    class_names,
    test_loader,
    targets,
    probabilities,
    metrics,
    efficientnet_weight,
    weight_search_frame,
    resnet_checkpoint,
    efficientnet_checkpoint,
    device,
):
    output_directory = (
        PROJECT_ROOT
        / "outputs"
        / "evaluation"
        / RUN_NAME
    )
    figures_directory = (
        PROJECT_ROOT
        / "outputs"
        / "figures"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = np.arange(len(class_names))
    predictions = metrics["predictions"]
    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    report_frame = pd.DataFrame(
        {
            "class_index": labels,
            "class_name": class_names,
            "display_name": [
                display_name(name)
                for name in class_names
            ],
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support.astype(int),
            "roc_auc_ovr": (
                metrics["per_class_roc_auc"]
            ),
        }
    )

    confusion_counts = confusion_matrix(
        targets,
        predictions,
        labels=labels,
    )
    row_totals = confusion_counts.sum(
        axis=1,
        keepdims=True,
    )
    normalized_confusion = np.divide(
        confusion_counts,
        row_totals,
        out=np.zeros_like(
            confusion_counts,
            dtype=float,
        ),
        where=row_totals != 0,
    )
    test_paths = extract_paths(test_loader)
    prediction_frame = pd.DataFrame(
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
            "confidence": probabilities.max(axis=1),
            "correct": predictions == targets,
        }
    )

    summary_path = (
        output_directory / "test_summary.json"
    )
    report_path = (
        output_directory
        / "classification_report.csv"
    )
    counts_path = (
        output_directory
        / "confusion_matrix_counts.csv"
    )
    normalized_path = (
        output_directory
        / "confusion_matrix_normalized.csv"
    )
    predictions_path = (
        output_directory / "predictions.csv"
    )
    weight_search_path = (
        output_directory
        / "validation_weight_search.csv"
    )
    confusion_figure_path = (
        figures_directory
        / f"{RUN_NAME}_test_confusion_matrix.png"
    )
    f1_figure_path = (
        figures_directory
        / f"{RUN_NAME}_test_per_class_f1.png"
    )

    report_frame.to_csv(
        report_path,
        index=False,
    )
    pd.DataFrame(
        confusion_counts,
        index=class_names,
        columns=class_names,
    ).to_csv(counts_path)
    pd.DataFrame(
        normalized_confusion,
        index=class_names,
        columns=class_names,
    ).to_csv(normalized_path)
    prediction_frame.to_csv(
        predictions_path,
        index=False,
    )
    weight_search_frame.to_csv(
        weight_search_path,
        index=False,
    )

    summary = {
        "run_name": RUN_NAME,
        "model_name": (
            "resnet18_efficientnet_b0_ensemble"
        ),
        "ensemble_type": (
            "validation_tuned_weighted_soft_voting"
        ),
        "selection_metric": (
            "validation_macro_f1"
        ),
        "test_used_for_weight_selection": False,
        "resnet_checkpoint": str(
            arguments.resnet_checkpoint.resolve()
        ),
        "resnet_checkpoint_epoch": int(
            resnet_checkpoint["epoch"]
        ),
        "efficientnet_checkpoint": str(
            arguments.efficientnet_checkpoint.resolve()
        ),
        "efficientnet_checkpoint_epoch": int(
            efficientnet_checkpoint["epoch"]
        ),
        "resnet_weight": (
            1.0 - efficientnet_weight
        ),
        "efficientnet_weight": (
            efficientnet_weight
        ),
        "test_examples": int(len(targets)),
        "num_classes": int(len(class_names)),
        "test_loss": metrics["loss"],
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
        "batch_size": arguments.batch_size,
        "device": str(device),
        "amp": device.type == "cuda",
        "pytorch_version": torch.__version__,
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

    save_confusion_figure(
        normalized_confusion,
        class_names,
        confusion_figure_path,
    )
    save_f1_figure(
        report_frame,
        f1_figure_path,
    )

    return {
        "summary": summary_path,
        "weight_search": weight_search_path,
        "confusion_figure": (
            confusion_figure_path
        ),
        "f1_figure": f1_figure_path,
    }


def print_optional_metric(label, value):
    if value is None:
        print(f"{label}: Hesaplanamadı")
    else:
        print(f"{label}: {value:.4f}")


def main():
    arguments = parse_arguments()
    arguments.resnet_checkpoint = (
        arguments.resnet_checkpoint.expanduser()
    )
    arguments.efficientnet_checkpoint = (
        arguments.efficientnet_checkpoint.expanduser()
    )
    validate_arguments(arguments)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    (
        resnet_checkpoint,
        efficientnet_checkpoint,
        checkpoint_class_names,
    ) = load_checkpoints(arguments)
    (
        _,
        validation_loader,
        test_loader,
        dataset_class_names,
    ) = create_dataloaders(
        batch_size=arguments.batch_size,
        num_workers=0,
    )

    if (
        checkpoint_class_names
        != list(dataset_class_names)
    ):
        raise RuntimeError(
            "Checkpoint ve veri seti sınıf "
            "indeksleri eşleşmiyor."
        )

    (
        resnet_model,
        efficientnet_model,
    ) = create_models(
        resnet_checkpoint,
        efficientnet_checkpoint,
        len(dataset_class_names),
        device,
    )

    print("=" * 70)
    print("RESNET18 + EFFICIENTNET-B0 SOFT-VOTING ENSEMBLE")
    print("=" * 70)
    print("Cihaz:", device)
    print("Validation örneği:", len(validation_loader.dataset))
    print("Test örneği:", len(test_loader.dataset))
    print("Batch size:", arguments.batch_size)
    print(
        "Ağırlık seçimi: Yalnızca validation Macro-F1"
    )
    print("Test kümesi ağırlık seçiminde kullanılmıyor.")
    print()

    (
        validation_targets,
        validation_resnet_probabilities,
        validation_efficientnet_probabilities,
    ) = collect_probabilities(
        resnet_model,
        efficientnet_model,
        validation_loader,
        device,
        "Validation ensemble",
    )
    (
        efficientnet_weight,
        weight_search_frame,
    ) = select_ensemble_weight(
        validation_targets,
        validation_resnet_probabilities,
        validation_efficientnet_probabilities,
    )

    print()
    print("VALIDATION AĞIRLIK ARAMASI")
    print(
        weight_search_frame.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )
    print()
    print(
        "Seçilen ağırlıklar | ResNet18:",
        f"{1.0 - efficientnet_weight:.2f}",
        "| EfficientNet-B0:",
        f"{efficientnet_weight:.2f}",
    )
    print()

    (
        test_targets,
        test_resnet_probabilities,
        test_efficientnet_probabilities,
    ) = collect_probabilities(
        resnet_model,
        efficientnet_model,
        test_loader,
        device,
        "Test ensemble",
    )
    test_probabilities = combine_probabilities(
        test_resnet_probabilities,
        test_efficientnet_probabilities,
        efficientnet_weight,
    )
    metrics = calculate_full_metrics(
        test_targets,
        test_probabilities,
        len(dataset_class_names),
    )
    output_paths = save_outputs(
        arguments=arguments,
        class_names=list(dataset_class_names),
        test_loader=test_loader,
        targets=test_targets,
        probabilities=test_probabilities,
        metrics=metrics,
        efficientnet_weight=efficientnet_weight,
        weight_search_frame=weight_search_frame,
        resnet_checkpoint=resnet_checkpoint,
        efficientnet_checkpoint=(
            efficientnet_checkpoint
        ),
        device=device,
    )

    print()
    print("=" * 70)
    print("ENSEMBLE TEST SONUÇLARI")
    print("=" * 70)
    print(f"Loss: {metrics['loss']:.4f}")
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
    print_optional_metric(
        "Macro ROC-AUC (OvR)",
        metrics["macro_roc_auc_ovr"],
    )
    print_optional_metric(
        "Weighted ROC-AUC (OvR)",
        metrics["weighted_roc_auc_ovr"],
    )
    print()
    print("Özet:", output_paths["summary"])
    print(
        "Ağırlık araması:",
        output_paths["weight_search"],
    )
    print(
        "Confusion matrix:",
        output_paths["confusion_figure"],
    )
    print(
        "Sınıf F1 grafiği:",
        output_paths["f1_figure"],
    )


if __name__ == "__main__":
    main()
