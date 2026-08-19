import argparse
from pathlib import Path

import torch
from torch import nn

from official_data_setup import create_official_dataloaders
from official_eval_common import (
    calculate_metrics,
    collect_model_probabilities,
    extract_relative_paths,
    run_name_from_checkpoint,
    save_evaluation_outputs,
)
from transfer_models import create_resnet18_transfer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ResNet18 on the locked conservative official PlantVillage test set."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if checkpoint.get("model_name") != "resnet18_transfer":
        raise ValueError("Checkpoint is not a resnet18_transfer model.")
    if checkpoint.get("data_protocol") != "conservative_official_leaf_preserving_test":
        raise ValueError("Checkpoint was not trained with the conservative official leaf-safe protocol.")
    if checkpoint.get("test_used_during_training") is not False:
        raise RuntimeError("Checkpoint metadata does not confirm a locked test protocol.")

    _, _, test_loader, class_names, split_summary = create_official_dataloaders(
        batch_size=args.batch_size,
        num_workers=0,
    )

    if list(checkpoint.get("class_names", [])) != list(class_names):
        raise RuntimeError("Checkpoint class order does not match the official manifest.")
    if len(test_loader.dataset) != 10709:
        raise RuntimeError("Locked official test set must contain 10,709 images.")

    dropout = float(checkpoint.get("dropout_rate", 0.3))
    model = create_resnet18_transfer(
        num_classes=len(class_names),
        dropout_rate=dropout,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    criterion = nn.CrossEntropyLoss()
    test_loss, targets, probabilities = collect_model_probabilities(
        model,
        test_loader,
        criterion,
        device,
    )
    metrics = calculate_metrics(targets, probabilities, len(class_names))
    relative_paths = extract_relative_paths(test_loader)

    run_name = run_name_from_checkpoint(args.checkpoint)
    summary, summary_path = save_evaluation_outputs(
        run_name=run_name,
        model_name="ResNet18",
        class_names=class_names,
        relative_paths=relative_paths,
        targets=targets,
        probabilities=probabilities,
        metrics=metrics,
        test_loss=test_loss,
        summary_extra={
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_validation_loss": float(checkpoint["validation_loss"]),
            "checkpoint_validation_accuracy": float(checkpoint["validation_accuracy"]),
            "checkpoint_validation_macro_f1": float(checkpoint["validation_macro_f1"]),
            "selection_metric": checkpoint.get("selection_metric", "validation_macro_f1"),
            "test_used_for_model_selection": False,
            "official_test_locked": True,
            "split_audit": {
                "final_exact_train_test_overlap_hashes": split_summary["final_exact_train_test_overlap_hashes"],
                "final_mapped_leaf_train_test_overlap_groups": split_summary["final_mapped_leaf_train_test_overlap_groups"],
                "final_mapped_leaf_train_validation_overlap_groups": split_summary["final_mapped_leaf_train_validation_overlap_groups"],
            },
            "batch_size": args.batch_size,
            "device": str(device),
            "pytorch_version": torch.__version__,
        },
    )

    errors = int((metrics["predictions"] != targets).sum())

    print("=" * 76)
    print("RESNET18 - LOCKED OFFICIAL LEAF-SAFE TEST")
    print("=" * 76)
    print("Checkpoint:", args.checkpoint)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print(f"Validation Macro-F1: {checkpoint['validation_macro_f1']:.4f}")
    print("Locked official test images:", len(targets))
    print("Classes:", len(class_names))
    print("Device:", device)
    print("Test used for training/checkpoint selection: NO")
    print()
    print("=" * 76)
    print("LOCKED OFFICIAL TEST RESULTS")
    print("=" * 76)
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
    print(f"Top-3 Accuracy: {metrics['top3_accuracy'] * 100:.2f}%")
    if metrics["macro_roc_auc_ovr"] is not None:
        print(f"Macro ROC-AUC (OvR): {metrics['macro_roc_auc_ovr']:.6f}")
    print(f"Errors: {errors} / {len(targets)}")
    print("Outputs:", summary_path.parent)


if __name__ == "__main__":
    main()
