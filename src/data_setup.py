from pathlib import Path
from random import Random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode


SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 2
TRAIN_EVAL_SAMPLES_PER_CLASS = 20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "PlantVillage"

TRAIN_DIR = DATASET_ROOT / "train"
ORIGINAL_VAL_DIR = DATASET_ROOT / "val"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.80, 1.0),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=15,
            interpolation=InterpolationMode.BILINEAR,
            fill=(128, 128, 128),
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


evaluation_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def _validate_dataset_directories():
    missing_directories = [
        directory
        for directory in (TRAIN_DIR, ORIGINAL_VAL_DIR)
        if not directory.exists()
    ]

    if missing_directories:
        missing_text = "\n".join(
            str(directory) for directory in missing_directories
        )
        raise FileNotFoundError(
            "PlantVillage klasörleri bulunamadı:\n"
            f"{missing_text}"
        )


def _loader_worker_options(
    num_workers,
    persistent_workers,
):
    if num_workers <= 0:
        return {}

    return {
        "persistent_workers": persistent_workers,
        "prefetch_factor": 2,
    }


def _validate_class_mapping(first_dataset, second_dataset):
    if first_dataset.class_to_idx != second_dataset.class_to_idx:
        raise RuntimeError(
            "Train ve validation sınıf indeksleri eşleşmiyor."
        )


def _create_validation_test_subsets(original_val_dataset):
    from sklearn.model_selection import train_test_split

    all_indices = list(range(len(original_val_dataset)))

    validation_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.5,
        random_state=SEED,
        stratify=original_val_dataset.targets,
    )

    validation_set = set(validation_indices)
    test_set = set(test_indices)

    if validation_set & test_set:
        raise RuntimeError(
            "Validation ve test kümelerinde ortak görüntü bulundu."
        )

    if validation_set | test_set != set(all_indices):
        raise RuntimeError(
            "Validation/test bölünmesinde eksik görüntü bulundu."
        )

    return (
        Subset(original_val_dataset, validation_indices),
        Subset(original_val_dataset, test_indices),
    )


def _choose_balanced_indices(
    targets,
    num_classes,
    samples_per_class,
):
    indices_by_class = [[] for _ in range(num_classes)]

    for index, target in enumerate(targets):
        indices_by_class[target].append(index)

    random_generator = Random(SEED)
    selected_indices = []

    for class_index, class_indices in enumerate(indices_by_class):
        if len(class_indices) < samples_per_class:
            raise RuntimeError(
                f"{class_index} indeksli sınıfta temiz train "
                "değerlendirmesi için yeterli görüntü yok."
            )

        random_generator.shuffle(class_indices)
        selected_indices.extend(
            class_indices[:samples_per_class]
        )

    random_generator.shuffle(selected_indices)
    return selected_indices


def create_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
):
    _validate_dataset_directories()

    train_dataset = ImageFolder(
        TRAIN_DIR,
        transform=train_transform,
    )
    original_val_dataset = ImageFolder(
        ORIGINAL_VAL_DIR,
        transform=evaluation_transform,
    )

    _validate_class_mapping(
        train_dataset,
        original_val_dataset,
    )

    validation_dataset, test_dataset = (
        _create_validation_test_subsets(
            original_val_dataset,
        )
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        **_loader_worker_options(
            num_workers,
            persistent_workers=True,
        ),
    }
    evaluation_options = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **train_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **evaluation_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **evaluation_options,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_dataset.classes,
    )


def create_train_evaluation_loader(
    batch_size=BATCH_SIZE,
    samples_per_class=TRAIN_EVAL_SAMPLES_PER_CLASS,
):
    _validate_dataset_directories()

    clean_train_dataset = ImageFolder(
        TRAIN_DIR,
        transform=evaluation_transform,
    )
    selected_indices = _choose_balanced_indices(
        targets=clean_train_dataset.targets,
        num_classes=len(clean_train_dataset.classes),
        samples_per_class=samples_per_class,
    )
    train_evaluation_dataset = Subset(
        clean_train_dataset,
        selected_indices,
    )

    train_evaluation_loader = DataLoader(
        train_evaluation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_evaluation_loader


if __name__ == "__main__":
    (
        train_loader,
        validation_loader,
        test_loader,
        class_names,
    ) = create_dataloaders()
    train_evaluation_loader = (
        create_train_evaluation_loader()
    )

    images, labels = next(iter(train_loader))

    print("=" * 50)
    print("DATA LOADER KONTROLÜ")
    print("=" * 50)
    print("Train:", len(train_loader.dataset))
    print(
        "Temiz train değerlendirme:",
        len(train_evaluation_loader.dataset),
    )
    print("Validation:", len(validation_loader.dataset))
    print("Test:", len(test_loader.dataset))
    print("Sınıf sayısı:", len(class_names))
    print("Görüntü batch boyutu:", images.shape)
    print("Etiket batch boyutu:", labels.shape)
    print("Sınıf indeksleri: Eşleşiyor")
    print("Validation/test çakışması: Yok")
