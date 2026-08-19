import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_setup import NUM_WORKERS, SEED, TRAIN_EVAL_SAMPLES_PER_CLASS
from models import BaselineCNN
from official_data_setup import (
    create_official_dataloaders,
    create_official_train_evaluation_loader,
)
from training import plot_training_history, train_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.4
EARLY_STOPPING_PATIENCE = 6
DEFAULT_RUN_NAME = "baseline_cnn_official_leafsafe_full"


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the custom baseline CNN on the conservative official PlantVillage leaf-safe split."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    return parser.parse_args()


def validate_args(args):
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative.")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive.")
    if args.weight_decay < 0:
        raise ValueError("weight-decay cannot be negative.")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1).")


def main():
    args = parse_args()
    validate_args(args)
    set_random_seeds(SEED)

    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"{args.run_name}_best.pt"
    history_path = PROJECT_ROOT / "outputs" / "histories" / f"{args.run_name}_history.csv"
    curves_path = PROJECT_ROOT / "outputs" / "figures" / f"{args.run_name}_training_curves.png"
    config_path = PROJECT_ROOT / "outputs" / "experiments" / f"{args.run_name}_config.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, validation_loader, test_loader, class_names, split_summary = (
        create_official_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )
    train_eval_loader = create_official_train_evaluation_loader(
        batch_size=args.batch_size,
        samples_per_class=TRAIN_EVAL_SAMPLES_PER_CLASS,
    )

    model = BaselineCNN(
        num_classes=len(class_names),
        dropout_rate=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
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
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config = {
        "run_name": args.run_name,
        "model_name": "baseline_cnn",
        "data_protocol": "conservative_official_leaf_preserving_test",
        "seed": SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
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
        "locked_test_examples": len(test_loader.dataset),
        "test_used_during_training": False,
        "num_classes": len(class_names),
        "trainable_parameters": parameter_count,
        "device": str(device),
        "amp": device.type == "cuda",
        "split_audit": {
            "exact_train_test_overlap_hashes": split_summary["final_exact_train_test_overlap_hashes"],
            "mapped_leaf_train_test_overlap_groups": split_summary["final_mapped_leaf_train_test_overlap_groups"],
            "mapped_leaf_train_validation_overlap_groups": split_summary["final_mapped_leaf_train_validation_overlap_groups"],
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print("=" * 76)
    print("BASELINE CNN - CONSERVATIVE OFFICIAL LEAF-SAFE TRAINING")
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
    print("Learning rate:", args.learning_rate)
    print("Dropout:", args.dropout)
    print("Weight decay:", args.weight_decay)
    print("Parameters:", f"{parameter_count:,}")
    print("Checkpoint selection: validation Macro-F1")
    print("Locked official test used during training: NO")
    print("AMP:", device.type == "cuda")

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
        history,
        curves_path,
        f"Baseline CNN - Official Leaf-Safe - {args.run_name}",
    )

    print()
    print("Training complete.")
    print("Best checkpoint:", checkpoint_path)
    print("History:", history_path)
    print("Curves:", curves_path)
    print("Config:", config_path)
    print("IMPORTANT: evaluate only the saved best checkpoint on the locked official test.")


if __name__ == "__main__":
    main()
