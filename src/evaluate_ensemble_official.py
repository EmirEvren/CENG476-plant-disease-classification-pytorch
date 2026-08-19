import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from efficientnet_models import create_efficientnet_b0_transfer
from official_data_setup import create_official_dataloaders
from official_eval_common import (
    calculate_metrics,
    extract_relative_paths,
    negative_log_likelihood,
    save_evaluation_outputs,
)
from transfer_models import create_resnet18_transfer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESNET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "resnet18_official_leafsafe_full_best.pt"
)
DEFAULT_EFFICIENTNET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "efficientnet_b0_official_leafsafe_full_best.pt"
)
RUN_NAME = "resnet18_efficientnet_official_leafsafe_soft_voting"
ELIGIBLE_EFFICIENTNET_WEIGHTS = (0.25, 0.50, 0.75)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select a ResNet18/EfficientNet-B0 soft-voting weight only on the "
            "leaf-safe validation set, then evaluate once on the locked official test."
        )
    )
    parser.add_argument("--resnet-checkpoint", type=Path, default=DEFAULT_RESNET_CHECKPOINT)
    parser.add_argument(
        "--efficientnet-checkpoint",
        type=Path,
        default=DEFAULT_EFFICIENTNET_CHECKPOINT,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def load_checkpoint(path: Path, expected_model_name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != expected_model_name:
        raise ValueError(f"Unexpected model_name in {path}: {checkpoint.get('model_name')}")
    if checkpoint.get("test_used_during_training") is not False:
        raise RuntimeError(f"Checkpoint does not confirm locked-test training: {path}")
    protocol = checkpoint.get("data_protocol") or checkpoint.get("evaluation_protocol")
    if protocol != "conservative_official_leaf_preserving_test":
        raise ValueError(f"Checkpoint is not from the conservative official protocol: {path}")
    return checkpoint


def create_models(resnet_checkpoint, efficientnet_checkpoint, num_classes, device):
    resnet = create_resnet18_transfer(
        num_classes=num_classes,
        dropout_rate=float(resnet_checkpoint.get("dropout_rate", 0.3)),
        pretrained=False,
    )
    efficientnet = create_efficientnet_b0_transfer(
        num_classes=num_classes,
        dropout_rate=float(efficientnet_checkpoint.get("dropout_rate", 0.3)),
        pretrained=False,
    )
    resnet.load_state_dict(resnet_checkpoint["model_state_dict"], strict=True)
    efficientnet.load_state_dict(efficientnet_checkpoint["model_state_dict"], strict=True)
    return resnet.to(device).eval(), efficientnet.to(device).eval()


def collect_pair_probabilities(resnet, efficientnet, loader, device, phase):
    use_amp = device.type == "cuda"
    targets = []
    resnet_probabilities = []
    efficientnet_probabilities = []

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=phase, unit="batch"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                resnet_logits = resnet(images)
                efficientnet_logits = efficientnet(images)

            targets.append(labels.numpy())
            resnet_probabilities.append(
                torch.softmax(resnet_logits.float(), dim=1).cpu().numpy()
            )
            efficientnet_probabilities.append(
                torch.softmax(efficientnet_logits.float(), dim=1).cpu().numpy()
            )

    return (
        np.concatenate(targets),
        np.concatenate(resnet_probabilities),
        np.concatenate(efficientnet_probabilities),
    )


def combine_probabilities(resnet_probabilities, efficientnet_probabilities, efficientnet_weight):
    return (
        (1.0 - efficientnet_weight) * resnet_probabilities
        + efficientnet_weight * efficientnet_probabilities
    )


def basic_metrics(targets, probabilities, num_classes):
    metrics = calculate_metrics(targets, probabilities, num_classes)
    return {
        "loss": negative_log_likelihood(targets, probabilities),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def select_weight(targets, resnet_probabilities, efficientnet_probabilities, num_classes):
    rows = []
    for efficientnet_weight in (0.0, *ELIGIBLE_EFFICIENTNET_WEIGHTS, 1.0):
        probabilities = combine_probabilities(
            resnet_probabilities,
            efficientnet_probabilities,
            efficientnet_weight,
        )
        metrics = basic_metrics(targets, probabilities, num_classes)
        rows.append(
            {
                "resnet_weight": 1.0 - efficientnet_weight,
                "efficientnet_weight": efficientnet_weight,
                "eligible_for_selection": efficientnet_weight in ELIGIBLE_EFFICIENTNET_WEIGHTS,
                "validation_loss": metrics["loss"],
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
            }
        )

    frame = pd.DataFrame(rows)
    eligible = frame[frame["eligible_for_selection"]].copy()
    best = eligible.sort_values(
        ["validation_macro_f1", "validation_loss"],
        ascending=[False, True],
    ).iloc[0]
    return float(best["efficientnet_weight"]), frame


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")

    resnet_checkpoint = load_checkpoint(args.resnet_checkpoint, "resnet18_transfer")
    efficientnet_checkpoint = load_checkpoint(
        args.efficientnet_checkpoint,
        "efficientnet_b0_transfer",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, validation_loader, test_loader, class_names, split_summary = (
        create_official_dataloaders(batch_size=args.batch_size, num_workers=0)
    )

    if list(resnet_checkpoint.get("class_names", [])) != list(class_names):
        raise RuntimeError("ResNet18 class order does not match official manifest.")
    if list(efficientnet_checkpoint.get("class_names", [])) != list(class_names):
        raise RuntimeError("EfficientNet-B0 class order does not match official manifest.")
    if len(test_loader.dataset) != 10709:
        raise RuntimeError("Locked official test set must contain 10,709 images.")

    resnet, efficientnet = create_models(
        resnet_checkpoint,
        efficientnet_checkpoint,
        len(class_names),
        device,
    )

    print("=" * 80)
    print("RESNET18 + EFFICIENTNET-B0 - OFFICIAL LEAF-SAFE SOFT VOTING")
    print("=" * 80)
    print("Validation images:", len(validation_loader.dataset))
    print("Locked official test images:", len(test_loader.dataset))
    print("Weight selection metric: validation Macro-F1")
    print("Test used for weight selection: NO")
    print("Candidate EfficientNet weights:", ELIGIBLE_EFFICIENTNET_WEIGHTS)
    print()

    validation_targets, validation_resnet, validation_efficientnet = (
        collect_pair_probabilities(
            resnet,
            efficientnet,
            validation_loader,
            device,
            "Leaf-safe validation ensemble",
        )
    )
    efficientnet_weight, weight_search = select_weight(
        validation_targets,
        validation_resnet,
        validation_efficientnet,
        len(class_names),
    )
    resnet_weight = 1.0 - efficientnet_weight

    output_dir = PROJECT_ROOT / "outputs" / "evaluation" / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_search_path = output_dir / "validation_weight_search.csv"
    weight_search.to_csv(weight_search_path, index=False)

    print("VALIDATION WEIGHT SEARCH")
    print(weight_search.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print(f"Selected weights | ResNet18: {resnet_weight:.2f} | EfficientNet-B0: {efficientnet_weight:.2f}")
    print()

    test_targets, test_resnet, test_efficientnet = collect_pair_probabilities(
        resnet,
        efficientnet,
        test_loader,
        device,
        "Locked official test ensemble",
    )
    test_probabilities = combine_probabilities(
        test_resnet,
        test_efficientnet,
        efficientnet_weight,
    )
    test_loss = negative_log_likelihood(test_targets, test_probabilities)
    metrics = calculate_metrics(test_targets, test_probabilities, len(class_names))
    relative_paths = extract_relative_paths(test_loader)

    summary, summary_path = save_evaluation_outputs(
        run_name=RUN_NAME,
        model_name="ResNet18 + EfficientNet-B0 Ensemble",
        class_names=class_names,
        relative_paths=relative_paths,
        targets=test_targets,
        probabilities=test_probabilities,
        metrics=metrics,
        test_loss=test_loss,
        summary_extra={
            "ensemble_type": "validation_tuned_weighted_soft_voting",
            "selection_metric": "validation_macro_f1",
            "test_used_for_weight_selection": False,
            "official_test_locked": True,
            "resnet_checkpoint": str(args.resnet_checkpoint.resolve()),
            "resnet_checkpoint_epoch": int(resnet_checkpoint["epoch"]),
            "efficientnet_checkpoint": str(args.efficientnet_checkpoint.resolve()),
            "efficientnet_checkpoint_epoch": int(efficientnet_checkpoint["epoch"]),
            "resnet_weight": resnet_weight,
            "efficientnet_weight": efficientnet_weight,
            "validation_weight_search": str(weight_search_path.resolve()),
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

    errors = int((metrics["predictions"] != test_targets).sum())
    print("=" * 80)
    print("LOCKED OFFICIAL ENSEMBLE TEST RESULTS")
    print("=" * 80)
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
    print(f"Top-3 Accuracy: {metrics['top3_accuracy'] * 100:.2f}%")
    if metrics["macro_roc_auc_ovr"] is not None:
        print(f"Macro ROC-AUC (OvR): {metrics['macro_roc_auc_ovr']:.6f}")
    print(f"Errors: {errors} / {len(test_targets)}")
    print("Outputs:", summary_path.parent)


if __name__ == "__main__":
    main()
