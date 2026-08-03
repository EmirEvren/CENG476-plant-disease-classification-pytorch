import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from data_setup import SEED, TRAIN_DIR, evaluation_transform
from models import BaselineCNN


SAMPLES_PER_CLASS = 1
MAX_STEPS = 300
PRINT_EVERY = 10
TARGET_ACCURACY = 0.99
LEARNING_RATE = 1e-3


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_balanced_indices(dataset, samples_per_class):
    indices_by_class = defaultdict(list)

    for index, target in enumerate(dataset.targets):
        if len(indices_by_class[target]) < samples_per_class:
            indices_by_class[target].append(index)

    selected_indices = []

    for class_index in range(len(dataset.classes)):
        class_indices = indices_by_class[class_index]

        if len(class_indices) != samples_per_class:
            raise RuntimeError(
                f"{dataset.classes[class_index]} sınıfında yeterli "
                "görüntü bulunamadı."
            )

        selected_indices.extend(class_indices)

    return selected_indices


def evaluate_batch(model, images, labels, criterion, device, use_amp):
    model.eval()

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

    predictions = logits.argmax(dim=1)
    accuracy = (predictions == labels).float().mean().item()

    return loss.item(), accuracy


def main():
    set_random_seeds(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = device.type == "cuda"

    full_dataset = ImageFolder(
        TRAIN_DIR,
        transform=evaluation_transform,
    )
    selected_indices = choose_balanced_indices(
        full_dataset,
        samples_per_class=SAMPLES_PER_CLASS,
    )
    tiny_dataset = Subset(full_dataset, selected_indices)

    tiny_loader = DataLoader(
        tiny_dataset,
        batch_size=len(tiny_dataset),
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp,
    )

    images, labels = next(iter(tiny_loader))
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    unique_labels = torch.unique(labels).numel()

    if unique_labels != len(full_dataset.classes):
        raise RuntimeError(
            "Sanity check batch'i bütün sınıfları içermiyor."
        )

    model = BaselineCNN(
        num_classes=len(full_dataset.classes),
        dropout_rate=0.0,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    print("=" * 65)
    print("KÜÇÜK VERİ EZBERLEME TESTİ")
    print("=" * 65)
    print("Cihaz:", device)
    print("Sınıf sayısı:", len(full_dataset.classes))
    print("Görüntü sayısı:", len(tiny_dataset))
    print("Batch şekli:", images.shape)
    print("Augmentation: Kapalı")
    print("Dropout: 0.0")
    print("Weight decay: 0.0")
    print("AMP:", use_amp)
    print()

    reached_target = False

    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        should_evaluate = (
            step == 1
            or step % PRINT_EVERY == 0
            or step == MAX_STEPS
        )

        if not should_evaluate:
            continue

        evaluation_loss, evaluation_accuracy = evaluate_batch(
            model=model,
            images=images,
            labels=labels,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        print(
            f"Adım {step:03d}/{MAX_STEPS} | "
            f"Loss: {evaluation_loss:.4f} | "
            f"Accuracy: {evaluation_accuracy * 100:.2f}%"
        )

        if evaluation_accuracy >= TARGET_ACCURACY:
            reached_target = True
            break

    print()

    if reached_target:
        print(
            "SONUÇ: BAŞARILI - Model küçük veri grubunu "
            "ezberleyebildi."
        )
        return

    print(
        "SONUÇ: BAŞARISIZ - Model hedef doğruluğa ulaşamadı. "
        "Tam eğitime geçmeden eğitim hattını incelemeliyiz."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()