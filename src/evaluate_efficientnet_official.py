import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from official_data_setup import create_official_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "efficientnet_b0_official_leafsafe_full_best.pt"
)
RUN_NAME = "efficientnet_b0_official_leafsafe_full"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the best EfficientNet-B0 checkpoint once on the locked "
            "conservative official PlantVillage test split."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def validate_arguments(args):
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")


def evaluate(model, loader, criterion, device):
    model.eval()
    use_amp = device.type == "cuda"
    running_loss = 0.0
    processed = 0
    targets = []
    probabilities = []

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Locked official test", unit="batch"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            probs = torch.softmax(logits.float(), dim=1)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            processed += batch_size
            targets.append(labels.cpu().numpy())
            probabilities.append(probs.cpu().numpy())

    if processed == 0:
        raise RuntimeError("Locked test loader produced no samples.")

    return (
        running_loss / processed,
        np.concatenate(targets),
        np.concatenate(probabilities),
    )


def calculate_metrics(targets, probabilities, num_classes):
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    predictions = probabilities.argmax(axis=1)
    labels = np.arange(num_classes)

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            targets,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            targets,
            predictions,
            labels=labels,
            average="weighted",
            zero_division=0,
        )
    )

    top3 = np.argpartition(probabilities, kth=-3, axis=1)[:, -3:]
    top3_accuracy = float(
        np.mean(np.any(top3 == targets[:, None], axis=1))
    )

    try:
        macro_auc = float(
            roc_auc_score(
                targets,
                probabilities,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
        weighted_auc = float(
            roc_auc_score(
                targets,
                probabilities,
                labels=labels,
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError:
        macro_auc = None
        weighted_auc = None

    return {
        "predictions": predictions,
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "top3_accuracy": top3_accuracy,
        "macro_roc_auc_ovr": macro_auc,
        "weighted_roc_auc_ovr": weighted_auc,
    }


def save_outputs(
    checkpoint_path,
    checkpoint,
    class_names,
    test_dataset,
    targets,
    probabilities,
    test_loss,
    metrics,
    split_summary,
    batch_size,
    device,
):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    output_dir = (
        PROJECT_ROOT / "outputs" / "evaluation" / RUN_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = metrics["predictions"]
    labels = np.arange(len(class_names))

    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    report = pd.DataFrame(
        {
            "class_index": labels,
            "class_name": class_names,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support.astype(int),
        }
    )
    report.to_csv(output_dir / "classification_report.csv", index=False)

    confusion = confusion_matrix(targets, predictions, labels=labels)
    pd.DataFrame(
        confusion,
        index=class_names,
        columns=class_names,
    ).to_csv(output_dir / "confusion_matrix_counts.csv")

    frame = test_dataset.frame.reset_index(drop=True)
    if len(frame) != len(targets):
        raise RuntimeError("Test manifest order does not match prediction count.")

    prediction_frame = pd.DataFrame(
        {
            "relative_path": frame["relative_path"],
            "true_index": targets,
            "true_class": [class_names[index] for index in targets],
            "predicted_index": predictions,
            "predicted_class": [class_names[index] for index in predictions],
            "confidence": probabilities.max(axis=1),
            "correct": predictions == targets,
        }
    )
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)

    error_count = int((predictions != targets).sum())
    summary = {
        "run_name": RUN_NAME,
        "model_name": "efficientnet_b0_transfer",
        "evaluation_protocol": split_summary["protocol"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_loss": float(checkpoint["validation_loss"]),
        "checkpoint_validation_accuracy": float(
            checkpoint["validation_accuracy"]
        ),
        "checkpoint_validation_macro_f1": float(
            checkpoint["validation_macro_f1"]
        ),
        "checkpoint_selection_metric": checkpoint.get(
            "selection_metric", "validation_macro_f1"
        ),
        "test_used_for_training": False,
        "test_used_for_checkpoint_selection": False,
        "locked_official_test": True,
        "test_examples": int(len(targets)),
        "test_errors": error_count,
        "num_classes": len(class_names),
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
        "batch_size": batch_size,
        "device": str(device),
        "amp": device.type == "cuda",
        "pytorch_version": torch.__version__,
        "split_audit": {
            "initial_exact_cross_official_split_pairs": split_summary[
                "initial_exact_cross_official_split_pairs"
            ],
            "train_images_excluded_for_exact_test_collision": split_summary[
                "train_images_excluded_for_exact_test_collision"
            ],
            "final_exact_train_test_overlap_hashes": split_summary[
                "final_exact_train_test_overlap_hashes"
            ],
            "final_mapped_leaf_train_test_overlap_groups": split_summary[
                "final_mapped_leaf_train_test_overlap_groups"
            ],
            "final_mapped_leaf_train_validation_overlap_groups": split_summary[
                "final_mapped_leaf_train_validation_overlap_groups"
            ],
        },
    }

    with (output_dir / "test_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    return output_dir, summary


def main():
    from efficientnet_models import create_efficientnet_b0_transfer

    args = parse_arguments()
    args.checkpoint = args.checkpoint.expanduser()
    validate_arguments(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("model_name") != "efficientnet_b0_transfer":
        raise ValueError("Checkpoint is not an EfficientNet-B0 transfer model.")

    (
        _,
        _,
        test_loader,
        class_names,
        split_summary,
    ) = create_official_dataloaders(
        batch_size=args.batch_size,
        num_workers=0,
    )

    checkpoint_classes = list(checkpoint.get("class_names", []))
    if checkpoint_classes != list(class_names):
        raise RuntimeError("Checkpoint and manifest class mappings do not match.")

    checkpoint_protocol = checkpoint.get("evaluation_protocol")
    if checkpoint_protocol != split_summary["protocol"]:
        raise RuntimeError(
            "Checkpoint was not trained with the conservative official protocol."
        )

    model = create_efficientnet_b0_transfer(
        num_classes=len(class_names),
        dropout_rate=float(checkpoint.get("dropout_rate", 0.3)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    criterion = nn.CrossEntropyLoss()

    print("=" * 76)
    print("EFFICIENTNET-B0 - LOCKED OFFICIAL LEAF-SAFE TEST")
    print("=" * 76)
    print("Checkpoint:", args.checkpoint)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation Macro-F1:", f"{checkpoint['validation_macro_f1']:.4f}")
    print("Locked official test images:", len(test_loader.dataset))
    print("Classes:", len(class_names))
    print("Device:", device)
    print("Test used for training/checkpoint selection: NO")

    test_loss, targets, probabilities = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )
    metrics = calculate_metrics(
        targets,
        probabilities,
        num_classes=len(class_names),
    )

    output_dir, summary = save_outputs(
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
        class_names=class_names,
        test_dataset=test_loader.dataset,
        targets=targets,
        probabilities=probabilities,
        test_loss=test_loss,
        metrics=metrics,
        split_summary=split_summary,
        batch_size=args.batch_size,
        device=device,
    )

    print()
    print("=" * 76)
    print("LOCKED OFFICIAL TEST RESULTS")
    print("=" * 76)
    print("Loss:", f"{summary['test_loss']:.4f}")
    print("Accuracy:", f"{summary['test_accuracy'] * 100:.2f}%")
    print("Macro Precision:", f"{summary['test_macro_precision']:.4f}")
    print("Macro Recall:", f"{summary['test_macro_recall']:.4f}")
    print("Macro-F1:", f"{summary['test_macro_f1']:.4f}")
    print("Weighted-F1:", f"{summary['test_weighted_f1']:.4f}")
    print("Top-3 Accuracy:", f"{summary['test_top3_accuracy'] * 100:.2f}%")
    if summary["test_macro_roc_auc_ovr"] is not None:
        print("Macro ROC-AUC (OvR):", f"{summary['test_macro_roc_auc_ovr']:.6f}")
    print("Errors:", summary["test_errors"], "/", summary["test_examples"])
    print("Outputs:", output_dir)


if __name__ == "__main__":
    main()
