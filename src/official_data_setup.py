from __future__ import annotations

import json
from pathlib import Path
from random import Random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from data_setup import (
    NUM_WORKERS,
    SEED,
    TRAIN_EVAL_SAMPLES_PER_CLASS,
    evaluation_transform,
    train_transform,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "PlantVillage"
MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "audit"
    / "official_leaf_safe_split_manifest.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "audit"
    / "official_leaf_safe_split_summary.json"
)

REQUIRED_COLUMNS = {
    "split",
    "relative_path",
    "class_index",
    "class_name",
    "sha256",
    "leaf_id",
}


class ManifestImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform
        self.targets = self.frame["class_index"].astype(int).tolist()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = DATASET_ROOT / str(row["relative_path"])
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(row["class_index"])


def _validate_summary():
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            "Leaf-safe summary not found. Run "
            "src/build_official_leaf_safe_manifest.py first."
        )

    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    required_zero_checks = (
        "final_exact_train_test_overlap_hashes",
        "final_mapped_leaf_train_test_overlap_groups",
        "final_mapped_leaf_train_validation_overlap_groups",
    )
    for key in required_zero_checks:
        if int(summary.get(key, -1)) != 0:
            raise RuntimeError(
                f"Leaf-safe summary check failed: {key}={summary.get(key)}"
            )

    if int(summary.get("test_examples", -1)) != 10709:
        raise RuntimeError(
            "Locked official test set must contain 10,709 images."
        )

    return summary


def _load_manifest():
    summary = _validate_summary()

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "Leaf-safe manifest not found. Run "
            "src/build_official_leaf_safe_manifest.py first."
        )

    frame = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise RuntimeError(
            "Manifest is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    expected_splits = {"train", "validation", "test"}
    actual_splits = set(frame["split"].unique())
    if actual_splits != expected_splits:
        raise RuntimeError(
            f"Unexpected manifest splits: {sorted(actual_splits)}"
        )

    path_duplicates = int(frame["relative_path"].duplicated().sum())
    if path_duplicates != 0:
        raise RuntimeError(
            f"Manifest contains {path_duplicates} duplicated local paths."
        )

    missing_files = [
        relative_path
        for relative_path in frame["relative_path"]
        if not (DATASET_ROOT / str(relative_path)).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Manifest references {len(missing_files)} missing image files."
        )

    class_table = (
        frame[["class_index", "class_name"]]
        .drop_duplicates()
        .sort_values("class_index")
    )
    if len(class_table) != 38:
        raise RuntimeError(
            f"Expected 38 classes, found {len(class_table)}."
        )

    expected_indices = list(range(38))
    actual_indices = class_table["class_index"].astype(int).tolist()
    if actual_indices != expected_indices:
        raise RuntimeError("Class indices are not contiguous 0..37.")

    class_names = class_table["class_name"].tolist()

    counts = frame["split"].value_counts().to_dict()
    for split_name in expected_splits:
        summary_key = f"{split_name}_examples"
        if int(counts.get(split_name, 0)) != int(summary.get(summary_key, -1)):
            raise RuntimeError(
                f"Manifest/summary count mismatch for {split_name}."
            )

    return frame, class_names, summary


def _loader_worker_options(num_workers):
    if num_workers <= 0:
        return {}
    return {
        "persistent_workers": True,
        "prefetch_factor": 2,
    }


def create_official_dataloaders(
    batch_size=32,
    num_workers=NUM_WORKERS,
):
    frame, class_names, summary = _load_manifest()

    train_frame = frame[frame["split"] == "train"].copy()
    validation_frame = frame[frame["split"] == "validation"].copy()
    test_frame = frame[frame["split"] == "test"].copy()

    train_dataset = ManifestImageDataset(
        train_frame,
        transform=train_transform,
    )
    validation_dataset = ManifestImageDataset(
        validation_frame,
        transform=evaluation_transform,
    )
    test_dataset = ManifestImageDataset(
        test_frame,
        transform=evaluation_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    common_train_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        **_loader_worker_options(num_workers),
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
        **common_train_options,
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
        class_names,
        summary,
    )


def _balanced_indices(targets, num_classes, samples_per_class):
    indices_by_class = [[] for _ in range(num_classes)]
    for index, target in enumerate(targets):
        indices_by_class[int(target)].append(index)

    rng = Random(SEED)
    selected = []
    for class_index, class_indices in enumerate(indices_by_class):
        if len(class_indices) < samples_per_class:
            raise RuntimeError(
                f"Class {class_index} has only {len(class_indices)} train images; "
                f"cannot select {samples_per_class} clean-eval samples."
            )
        rng.shuffle(class_indices)
        selected.extend(class_indices[:samples_per_class])

    rng.shuffle(selected)
    return selected


def create_official_train_evaluation_loader(
    batch_size=32,
    samples_per_class=TRAIN_EVAL_SAMPLES_PER_CLASS,
):
    frame, class_names, _ = _load_manifest()
    train_frame = frame[frame["split"] == "train"].copy()

    clean_dataset = ManifestImageDataset(
        train_frame,
        transform=evaluation_transform,
    )
    selected_indices = _balanced_indices(
        clean_dataset.targets,
        num_classes=len(class_names),
        samples_per_class=samples_per_class,
    )
    subset = Subset(clean_dataset, selected_indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    (
        train_loader,
        validation_loader,
        test_loader,
        class_names,
        summary,
    ) = create_official_dataloaders()
    clean_loader = create_official_train_evaluation_loader()

    images, labels = next(iter(train_loader))

    print("=" * 72)
    print("OFFICIAL LEAF-SAFE DATA LOADER CHECK")
    print("=" * 72)
    print("Protocol:", summary["protocol"])
    print("Train:", len(train_loader.dataset))
    print("Clean train eval:", len(clean_loader.dataset))
    print("Validation:", len(validation_loader.dataset))
    print("Locked official test:", len(test_loader.dataset))
    print("Classes:", len(class_names))
    print("Batch image shape:", tuple(images.shape))
    print("Batch label shape:", tuple(labels.shape))
    print("Final exact train/test overlap hashes: 0")
    print("Final mapped leaf train/test overlap groups: 0")
    print("Final mapped leaf train/validation overlap groups: 0")
