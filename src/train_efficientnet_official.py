import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_setup import NUM_WORKERS, SEED, TRAIN_EVAL_SAMPLES_PER_CLASS
from official_data_setup import (
    create_official_dataloaders,
    create_official_train_evaluation_loader,
)


DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 32
DEFAULT_BACKBONE_LEARNING_RATE = 1e-4
DEFAULT_CLASSIFIER_LEARNING_RATE = 5e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 5
DEFAULT_RUN_NAME = "efficientnet_b0_official_leafsafe_full"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune ImageNet-pretrained EfficientNet-B0 using the "
            "conservative official PlantVillage leaf-safe manifest."
        )
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=DEFAULT_BACKBONE_LEARNING_RATE,
    )
    parser.add_argument(
        "--classifier-learning-rate",
        type=float,
        default=DEFAULT_CLASSIFIER_LEARNING_RATE,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT_RATE)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    return parser.parse_args()


def validate_arguments(args):
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.backbone_learning_rate <= 0:
        raise ValueError("--backbone-learning-rate must be positive.")
    if args.classifier_learning_rate <= 0:
        raise ValueError("--classifier-learning-rate must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1).")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        raise ValueError("--run-name contains invalid characters.")


def main():
    from efficientnet_models import (
        create_efficientnet_b0_transfer,
        split_efficientnet_b0_parameters,
    )
    from training import plot_training_history, train_model

    args = parse_arguments()
    validate_arguments(args)
    set_random_seeds(SEED)

    checkpoint_path = (
        PROJECT_ROOT / "outputs" / "checkpoints" / f"{args.run_name}_best.pt"
    )
    history_path = (
        PROJECT_ROOT / "outputs" / "histories" / f"{args.run_name}_history.csv"
    )
    curves_path = (
        PROJECT_ROOT / "outputs" / "figures" / f"{args.run_name}_training_curves.png"
    )
    config_path = (
        PROJECT_ROOT / "outputs" / "experiments" / f"{args.run_name}_config.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        train_loader,
        validation_loader,
        test_loader,
        class_names,
        split_summary,
    ) = create_official_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_eval_loader = create_official_train_evaluation_loader(
        batch_size=args.batch_size,
        samples_per_class=TRAIN_EVAL_SAMPLES_PER_CLASS,
    )

    model = create_efficientnet_b0_transfer(
        num_classes=len(class_names),
        dropout_rate=args.dropout,
        pretrained=True,
    ).to(device)
    backbone_parameters, classifier_parameters = (
        split_efficientnet_b0_parameters(model)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": args.backbone_learning_rate,
                "name": "backbone",
            },
            {
                "params": classifier_parameters,
                "lr": args.classifier_learning_rate,
                "name": "classifier",
            },
        ],
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    config = {
        "run_name": args.run_name,
        "model_name": "efficientnet_b0_transfer",
        "architecture": "EfficientNet-B0",
        "pretrained": True,
        "pretrained_weights": "EfficientNet_B0_Weights.DEFAULT",
        "evaluation_protocol": split_summary["protocol"],
        "official_test_locked": True,
        "test_used_during_training": False,
        "seed": SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_num_workers": args.num_workers,
        "backbone_learning_rate": args.backbone_learning_rate,
        "classifier_learning_rate": args.classifier_learning_rate,
        "weight_decay": args.weight_decay,
        "dropout_rate": args.dropout,
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.999],
        "criterion": "CrossEntropyLoss_unweighted",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_monitor": "validation_loss",
        "selection_metric": "validation_macro_f1",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "train_examples": len(train_loader.dataset),
        "clean_train_eval_examples": len(train_eval_loader.dataset),
        "validation_examples": len(validation_loader.dataset),
        "locked_official_test_examples": len(test_loader.dataset),
        "num_classes": len(class_names),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "device": str(device),
        "amp": use_amp,
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

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print("=" * 76)
    print("EFFICIENTNET-B0 - CONSERVATIVE OFFICIAL LEAF-SAFE TRAINING")
    print("=" * 76)
    print("Run:", args.run_name)
    print("Device:", device)
    print("Train images:", len(train_loader.dataset))
    print("Clean train eval:", len(train_eval_loader.dataset))
    print("Validation images:", len(validation_loader.dataset))
    print("Locked official test images:", len(test_loader.dataset))
    print("Classes:", len(class_names))
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Backbone LR:", args.backbone_learning_rate)
    print("Classifier LR:", args.classifier_learning_rate)
    print("Dropout:", args.dropout)
    print("Weight decay:", args.weight_decay)
    print("Parameters:", f"{total_parameters:,}")
    print("Checkpoint selection: validation Macro-F1")
    print("Locked official test used during training: NO")
    print("AMP:", use_amp)

    history = train_model(
        model=model,
        train_loader=train_loader,
        train_evaluation_loader=train_eval_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        epochs=args.epochs,
        patience=EARLY_STOPPING_PATIENCE,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        checkpoint_metadata={**config, "class_names": class_names},
    )

    plot_training_history(
        history=history,
        output_path=curves_path,
        title=(
            "EfficientNet-B0 - Conservative Official Leaf-Safe Protocol - "
            f"{args.run_name}"
        ),
    )

    print()
    print("Training complete.")
    print("Best checkpoint:", checkpoint_path)
    print("History:", history_path)
    print("Curves:", curves_path)
    print("Config:", config_path)
    print()
    print("IMPORTANT: Do not evaluate or tune on the locked official test until")
    print("the best checkpoint has been selected from validation Macro-F1.")


if __name__ == "__main__":
    main()
