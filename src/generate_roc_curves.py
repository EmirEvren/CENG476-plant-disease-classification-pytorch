import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from data_setup import BATCH_SIZE, create_dataloaders
from efficientnet_models import create_efficientnet_b0_transfer
from models import BaselineCNN
from transfer_models import create_resnet18_transfer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate micro- and macro-average one-vs-rest ROC curves for a saved checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _infer_model_name(checkpoint, checkpoint_path):
    model_name = str(checkpoint.get("model_name", "")).lower()
    filename = checkpoint_path.name.lower()

    if "efficientnet" in model_name or "efficientnet" in filename:
        return "efficientnet_b0"
    if "resnet" in model_name or "resnet" in filename:
        return "resnet18"
    if "baseline" in model_name or "baseline" in filename:
        return "baseline_cnn"

    raise ValueError(
        "Could not determine model type from checkpoint metadata or filename."
    )


def _create_model(model_name, num_classes, dropout_rate):
    if model_name == "baseline_cnn":
        return BaselineCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    if model_name == "resnet18":
        return create_resnet18_transfer(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained=False,
        )
    if model_name == "efficientnet_b0":
        return create_efficientnet_b0_transfer(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained=False,
        )
    raise ValueError(f"Unsupported model type: {model_name}")


def _default_dropout(model_name, checkpoint):
    if "dropout_rate" in checkpoint:
        return float(checkpoint["dropout_rate"])
    return 0.4 if model_name == "baseline_cnn" else 0.3


def _collect_probabilities(model, loader, device):
    targets = []
    probabilities = []
    use_amp = device.type == "cuda"
    model.eval()

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="ROC inference", unit="batch"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)

            probabilities.append(
                torch.softmax(logits.float(), dim=1).cpu().numpy()
            )
            targets.append(labels.numpy())

    return np.concatenate(targets), np.concatenate(probabilities)


def _calculate_average_curves(targets, probabilities, num_classes):
    binary_targets = label_binarize(
        targets,
        classes=np.arange(num_classes),
    )

    micro_fpr, micro_tpr, _ = roc_curve(
        binary_targets.ravel(),
        probabilities.ravel(),
    )
    micro_auc = auc(micro_fpr, micro_tpr)

    per_class_fpr = {}
    per_class_tpr = {}
    all_fpr = []

    for class_index in range(num_classes):
        fpr, tpr, _ = roc_curve(
            binary_targets[:, class_index],
            probabilities[:, class_index],
        )
        per_class_fpr[class_index] = fpr
        per_class_tpr[class_index] = tpr
        all_fpr.append(fpr)

    macro_fpr = np.unique(np.concatenate(all_fpr))
    mean_tpr = np.zeros_like(macro_fpr)

    for class_index in range(num_classes):
        mean_tpr += np.interp(
            macro_fpr,
            per_class_fpr[class_index],
            per_class_tpr[class_index],
        )

    mean_tpr /= num_classes
    macro_auc = auc(macro_fpr, mean_tpr)

    return micro_fpr, micro_tpr, micro_auc, macro_fpr, mean_tpr, macro_auc


def main():
    arguments = parse_arguments()
    checkpoint_path = arguments.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if arguments.batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    _, _, test_loader, class_names = create_dataloaders(
        batch_size=arguments.batch_size,
        num_workers=0,
    )

    model_name = _infer_model_name(checkpoint, checkpoint_path)
    dropout_rate = _default_dropout(model_name, checkpoint)
    model = _create_model(
        model_name=model_name,
        num_classes=len(class_names),
        dropout_rate=dropout_rate,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    targets, probabilities = _collect_probabilities(
        model,
        test_loader,
        device,
    )

    (
        micro_fpr,
        micro_tpr,
        micro_auc,
        macro_fpr,
        macro_tpr,
        macro_auc,
    ) = _calculate_average_curves(
        targets,
        probabilities,
        len(class_names),
    )

    if arguments.output is None:
        run_name = checkpoint_path.stem.replace("_best", "")
        output_path = (
            PROJECT_ROOT
            / "outputs"
            / "figures"
            / f"{run_name}_test_roc_curve.png"
        )
    else:
        output_path = arguments.output

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(
        macro_fpr,
        macro_tpr,
        linewidth=2,
        label=f"Macro-average OvR ROC (AUC={macro_auc:.4f})",
    )
    axis.plot(
        micro_fpr,
        micro_tpr,
        linewidth=2,
        label=f"Micro-average OvR ROC (AUC={micro_auc:.4f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(f"{model_name} - Test ROC Curves")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    print("Model:", model_name)
    print(f"Macro-average ROC-AUC: {macro_auc:.6f}")
    print(f"Micro-average ROC-AUC: {micro_auc:.6f}")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
