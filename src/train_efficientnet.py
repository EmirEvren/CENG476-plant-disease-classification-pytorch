import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
)

from data_setup import (
    NUM_WORKERS,
    SEED,
    TRAIN_EVAL_SAMPLES_PER_CLASS,
    create_dataloaders,
    create_train_evaluation_loader,
)


DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 32
DEFAULT_BACKBONE_LEARNING_RATE = 1e-4
DEFAULT_CLASSIFIER_LEARNING_RATE = 5e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _float_token(value):
    return (
        f"{value:.0e}"
        .replace("+", "p")
        .replace("-", "m")
    )


def _default_run_name(
    batch_size,
    backbone_learning_rate,
    classifier_learning_rate,
):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return (
        f"efficientnet_b0_b{batch_size}_"
        f"blr{_float_token(backbone_learning_rate)}_"
        f"hlr{_float_token(classifier_learning_rate)}_"
        f"{timestamp}"
    )


def _validate_run_name(run_name):
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*",
        run_name,
    ):
        raise ValueError(
            "Run adı yalnızca harf, sayı, tire ve "
            "alt çizgi içerebilir."
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "ImageNet ön-eğitimli EfficientNet-B0 "
            "modelini "
            "PlantVillage üzerinde fine-tune eder."
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Maksimum epoch sayısı",
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=(
            DEFAULT_BACKBONE_LEARNING_RATE
        ),
        help="Ön-eğitimli backbone öğrenme oranı",
    )
    parser.add_argument(
        "--classifier-learning-rate",
        type=float,
        default=(
            DEFAULT_CLASSIFIER_LEARNING_RATE
        ),
        help="Yeni classifier öğrenme oranı",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch boyutu",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Train DataLoader worker sayısı",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT_RATE,
        help="Classifier dropout oranı",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Deney dosyalarında kullanılacak ad",
    )

    return parser.parse_args()


def _validate_arguments(arguments):
    if arguments.epochs <= 0:
        raise ValueError(
            "Epoch sayısı pozitif olmalıdır."
        )
    if arguments.backbone_learning_rate <= 0:
        raise ValueError(
            "Backbone learning rate pozitif olmalıdır."
        )
    if arguments.classifier_learning_rate <= 0:
        raise ValueError(
            "Classifier learning rate pozitif olmalıdır."
        )
    if arguments.batch_size <= 0:
        raise ValueError(
            "Batch size pozitif olmalıdır."
        )
    if arguments.num_workers < 0:
        raise ValueError(
            "Num workers negatif olamaz."
        )
    if arguments.weight_decay < 0:
        raise ValueError(
            "Weight decay negatif olamaz."
        )
    if not 0 <= arguments.dropout < 1:
        raise ValueError(
            "Dropout [0, 1) aralığında olmalıdır."
        )


def main():
    from training import (
        plot_training_history,
        train_model,
    )
    from efficientnet_models import (
        create_efficientnet_b0_transfer,
        split_efficientnet_b0_parameters,
    )

    arguments = parse_arguments()
    _validate_arguments(arguments)
    set_random_seeds(SEED)

    run_name = (
        arguments.run_name
        or _default_run_name(
            batch_size=arguments.batch_size,
            backbone_learning_rate=(
                arguments.backbone_learning_rate
            ),
            classifier_learning_rate=(
                arguments.classifier_learning_rate
            ),
        )
    )
    _validate_run_name(run_name)

    checkpoint_path = (
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / f"{run_name}_best.pt"
    )
    history_path = (
        PROJECT_ROOT
        / "outputs"
        / "histories"
        / f"{run_name}_history.csv"
    )
    curves_path = (
        PROJECT_ROOT
        / "outputs"
        / "figures"
        / f"{run_name}_training_curves.png"
    )
    config_path = (
        PROJECT_ROOT
        / "outputs"
        / "experiments"
        / f"{run_name}_config.json"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    (
        train_loader,
        validation_loader,
        _,
        class_names,
    ) = create_dataloaders(
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
    )
    train_evaluation_loader = (
        create_train_evaluation_loader(
            batch_size=arguments.batch_size,
            samples_per_class=(
                TRAIN_EVAL_SAMPLES_PER_CLASS
            ),
        )
    )

    model = create_efficientnet_b0_transfer(
        num_classes=len(class_names),
        dropout_rate=arguments.dropout,
        pretrained=True,
    ).to(device)
    (
        backbone_parameters,
        classifier_parameters,
    ) = split_efficientnet_b0_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": (
                    arguments.backbone_learning_rate
                ),
                "name": "backbone",
            },
            {
                "params": classifier_parameters,
                "lr": (
                    arguments.classifier_learning_rate
                ),
                "name": "classifier",
            },
        ],
        weight_decay=arguments.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    backbone_parameter_count = sum(
        parameter.numel()
        for parameter in backbone_parameters
    )
    classifier_parameter_count = sum(
        parameter.numel()
        for parameter in classifier_parameters
    )

    config = {
        "run_name": run_name,
        "model_name": "efficientnet_b0_transfer",
        "architecture": "EfficientNet-B0",
        "pretrained": True,
        "pretrained_weights": (
            "EfficientNet_B0_Weights.DEFAULT"
        ),
        "training_strategy": (
            "full_fine_tuning_with_"
            "differential_learning_rates"
        ),
        "seed": SEED,
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "train_num_workers": arguments.num_workers,
        "evaluation_num_workers": 0,
        "backbone_learning_rate": (
            arguments.backbone_learning_rate
        ),
        "classifier_learning_rate": (
            arguments.classifier_learning_rate
        ),
        "weight_decay": arguments.weight_decay,
        "dropout_rate": arguments.dropout,
        "optimizer": "AdamW",
        "criterion": (
            "CrossEntropyLoss_unweighted"
        ),
        "scheduler": "ReduceLROnPlateau",
        "scheduler_monitor": "validation_loss",
        "selection_metric": (
            "validation_macro_f1"
        ),
        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),
        "test_used_during_training": False,
        "train_examples": len(
            train_loader.dataset
        ),
        "clean_train_eval_examples": len(
            train_evaluation_loader.dataset
        ),
        "validation_examples": len(
            validation_loader.dataset
        ),
        "num_classes": len(class_names),
        "total_parameters": total_parameters,
        "trainable_parameters": (
            trainable_parameters
        ),
        "backbone_parameters": (
            backbone_parameter_count
        ),
        "classifier_parameters": (
            classifier_parameter_count
        ),
        "device": str(device),
        "amp": use_amp,
        "pytorch_version": torch.__version__,
    }

    config_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with config_path.open(
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(
            config,
            config_file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 65)
    print("EFFICIENTNET-B0 TRANSFER LEARNING EĞİTİMİ")
    print("=" * 65)
    print("Run adı:", run_name)
    print("Cihaz:", device)
    print(
        "Ön-eğitimli ağırlıklar: "
        "EfficientNet_B0_Weights.DEFAULT"
    )
    print("Epoch:", arguments.epochs)
    print(
        "Train örneği:",
        len(train_loader.dataset),
    )
    print(
        "Temiz train değerlendirme:",
        len(train_evaluation_loader.dataset),
    )
    print(
        "Validation örneği:",
        len(validation_loader.dataset),
    )
    print(
        "Train/Evaluation worker:",
        f"{arguments.num_workers}/0",
    )
    print("Batch size:", arguments.batch_size)
    print(
        "Toplam parametre:",
        f"{total_parameters:,}",
    )
    print(
        "Eğitilebilir parametre:",
        f"{trainable_parameters:,}",
    )
    print(
        "Backbone learning rate:",
        arguments.backbone_learning_rate,
    )
    print(
        "Classifier learning rate:",
        arguments.classifier_learning_rate,
    )
    print(
        "Weight decay:",
        arguments.weight_decay,
    )
    print("Dropout:", arguments.dropout)
    print("Optimizer: AdamW")
    print(
        "Fine-tuning: Tüm katmanlar açık"
    )
    print(
        "Model seçim metriği: "
        "Validation Macro-F1"
    )
    print("Test kümesi: Kullanılmıyor")
    print("AMP:", use_amp)

    history = train_model(
        model=model,
        train_loader=train_loader,
        train_evaluation_loader=(
            train_evaluation_loader
        ),
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        epochs=arguments.epochs,
        patience=EARLY_STOPPING_PATIENCE,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        checkpoint_metadata={
            **config,
            "class_names": class_names,
        },
    )

    plot_training_history(
        history=history,
        output_path=curves_path,
        title=(
            "EfficientNet-B0 Transfer Learning "
            f"Sonuçları - {run_name}"
        ),
    )

    print()
    print("Eğitim tamamlandı.")
    print("En iyi model:", checkpoint_path)
    print("Geçmiş:", history_path)
    print("Grafik:", curves_path)
    print("Ayarlar:", config_path)


if __name__ == "__main__":
    main()
