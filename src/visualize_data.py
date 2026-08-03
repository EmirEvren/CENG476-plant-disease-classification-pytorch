from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch

from data_setup import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    create_dataloaders,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DISTRIBUTION_FILE = PROJECT_ROOT / "outputs" / "class_distribution.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def readable_name(class_name):
    if "___" in class_name:
        plant, condition = class_name.split("___", maxsplit=1)
        plant = plant.replace("_", " ")
        condition = condition.replace("_", " ")
        return f"{plant} - {condition}"

    return class_name.replace("_", " ")


def save_sample_images():
    train_loader, _, _, class_names = create_dataloaders()
    images, labels = next(iter(train_loader))

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    images = images * std + mean
    images = images.clamp(0, 1)

    figure, axes = plt.subplots(3, 4, figsize=(14, 10))

    for index, axis in enumerate(axes.flat):
        image = images[index].permute(1, 2, 0).numpy()
        class_name = class_names[labels[index].item()]

        axis.imshow(image)
        axis.set_title(
            readable_name(class_name),
            fontsize=8,
        )
        axis.axis("off")

    figure.suptitle(
        "Augmentation Sonrası Eğitim Görüntüleri",
        fontsize=16,
    )
    figure.tight_layout()

    output_path = FIGURES_DIR / "sample_images.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path


def save_class_distribution():
    distribution = pd.read_csv(DISTRIBUTION_FILE)
    distribution["readable_name"] = distribution["class_name"].apply(
        readable_name
    )

    distribution = distribution.sort_values("total_count")

    figure, axis = plt.subplots(figsize=(14, 14))

    bars = axis.barh(
        distribution["readable_name"],
        distribution["total_count"],
    )

    axis.bar_label(
        bars,
        padding=3,
        fontsize=7,
    )

    axis.set_title("PlantVillage Sınıf Dağılımı")
    axis.set_xlabel("Görüntü Sayısı")
    axis.set_ylabel("Sınıf")
    axis.grid(axis="x", alpha=0.25)

    figure.tight_layout()

    output_path = FIGURES_DIR / "class_distribution.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path


if __name__ == "__main__":
    sample_path = save_sample_images()
    distribution_path = save_class_distribution()

    print("Grafikler oluşturuldu:")
    print("Örnek görüntüler:", sample_path)
    print("Sınıf dağılımı:", distribution_path)