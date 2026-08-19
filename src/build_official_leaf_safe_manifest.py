from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.datasets import ImageFolder


SEED = 42
VALIDATION_FRACTION_OF_OFFICIAL_TRAIN = 0.10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "PlantVillage"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audit"
CACHE_DIR = PROJECT_ROOT / ".cache" / "plantvillage_official"

HF_SPLIT_BASE = (
    "https://huggingface.co/datasets/mohanty/PlantVillage/"
    "resolve/main/splits"
)
OFFICIAL_TRAIN_URL = f"{HF_SPLIT_BASE}/color_train.txt"
OFFICIAL_TEST_URL = f"{HF_SPLIT_BASE}/color_test.txt"

EXPECTED_OFFICIAL_TRAIN = 43596
EXPECTED_OFFICIAL_TEST = 10709
EXPECTED_TOTAL = 54305

MANIFEST_PATH = OUTPUT_DIR / "official_leaf_safe_split_manifest.csv"
SUMMARY_PATH = OUTPUT_DIR / "official_leaf_safe_split_summary.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a local PlantVillage manifest from the official predefined "
            "leaf-preserving color train/test split. The official test split "
            "remains locked. A deterministic stratified validation subset is "
            "created only inside the official training split."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=VALIDATION_FRACTION_OF_OFFICIAL_TRAIN,
        help="Fraction of the official train split used for validation.",
    )
    parser.add_argument(
        "--skip-hash-audit",
        action="store_true",
        help="Skip SHA-256 exact-duplicate audit across official train/test.",
    )
    return parser.parse_args()


def validate_args(args):
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5.")


def download_text(url: str, destination: Path) -> Path:
    if destination.is_file():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading:", url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CENG476-OfficialPlantVillageSplit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())
    return destination


def read_split_file(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def parse_official_path(value: str):
    normalized = value.replace("\\", "/").strip("/")
    parts = normalized.split("/")
    if len(parts) < 2:
        raise ValueError(f"Unexpected official split path: {value}")
    class_name = parts[-2]
    filename = parts[-1]
    return class_name, filename


def local_key(class_name: str, filename: str):
    return class_name, filename.lower()


def build_local_index(dataset_root: Path):
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    for directory in (train_dir, val_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")

    train_dataset = ImageFolder(train_dir)
    val_dataset = ImageFolder(val_dir)

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise RuntimeError("Local train and val class mappings do not match.")

    by_key = defaultdict(list)
    for source_partition, dataset in (
        ("kaggle_train", train_dataset),
        ("kaggle_val", val_dataset),
    ):
        for path_string, class_index in dataset.samples:
            path = Path(path_string)
            class_name = dataset.classes[class_index]
            key = local_key(class_name, path.name)
            by_key[key].append(
                {
                    "path": path,
                    "relative_path": str(path.relative_to(dataset_root)),
                    "source_partition": source_partition,
                    "class_index": class_index,
                    "class_name": class_name,
                }
            )

    duplicate_keys = {
        key: rows for key, rows in by_key.items() if len(rows) != 1
    }
    return by_key, duplicate_keys, train_dataset.classes


def official_rows(split_name: str, entries, local_index):
    rows = []
    missing = []

    for official_path in entries:
        class_name, filename = parse_official_path(official_path)
        key = local_key(class_name, filename)
        matches = local_index.get(key, [])

        if len(matches) != 1:
            missing.append(
                {
                    "official_split": split_name,
                    "official_path": official_path,
                    "class_name": class_name,
                    "filename": filename,
                    "local_match_count": len(matches),
                }
            )
            continue

        local = matches[0]
        rows.append(
            {
                **local,
                "official_split": split_name,
                "official_path": official_path,
            }
        )

    return rows, missing


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def exact_duplicate_audit(official_train_rows, official_test_rows):
    train_hashes = defaultdict(list)
    test_hashes = defaultdict(list)

    for row in tqdm(
        official_train_rows,
        desc="Hashing official train",
        unit="img",
    ):
        train_hashes[sha256_file(row["path"])].append(row)

    for row in tqdm(
        official_test_rows,
        desc="Hashing official test",
        unit="img",
    ):
        test_hashes[sha256_file(row["path"])].append(row)

    shared_hashes = sorted(set(train_hashes) & set(test_hashes))
    pairs = []
    for digest in shared_hashes:
        for left in train_hashes[digest]:
            for right in test_hashes[digest]:
                pairs.append(
                    {
                        "sha256": digest,
                        "train_relative_path": left["relative_path"],
                        "test_relative_path": right["relative_path"],
                        "train_class": left["class_name"],
                        "test_class": right["class_name"],
                    }
                )
    return pairs


def create_internal_validation(official_train_rows, validation_fraction):
    indices = list(range(len(official_train_rows)))
    labels = [row["class_index"] for row in official_train_rows]

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=SEED,
        stratify=labels,
    )

    train_set = set(train_indices)
    validation_set = set(validation_indices)
    if train_set & validation_set:
        raise RuntimeError("Internal train/validation overlap detected.")

    output = []
    for index, row in enumerate(official_train_rows):
        split = "validation" if index in validation_set else "train"
        output.append({**row, "split": split})
    return output


def serialize_manifest_rows(rows):
    serializable = []
    for row in rows:
        serializable.append(
            {
                "split": row["split"],
                "official_split": row["official_split"],
                "relative_path": row["relative_path"],
                "source_partition": row["source_partition"],
                "class_index": row["class_index"],
                "class_name": row["class_name"],
                "official_path": row["official_path"],
            }
        )
    return serializable


def main():
    args = parse_args()
    validate_args(args)

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("OFFICIAL PLANTVILLAGE LEAF-SAFE SPLIT BUILDER")
    print("=" * 78)
    print("Dataset root:", dataset_root)
    print("Seed:", SEED)
    print(
        "Protocol: official leaf-preserving train/test; internal validation "
        "only inside official train"
    )
    print()

    official_train_path = download_text(
        OFFICIAL_TRAIN_URL,
        CACHE_DIR / "color_train.txt",
    )
    official_test_path = download_text(
        OFFICIAL_TEST_URL,
        CACHE_DIR / "color_test.txt",
    )

    official_train_entries = read_split_file(official_train_path)
    official_test_entries = read_split_file(official_test_path)

    print("Official train entries:", len(official_train_entries))
    print("Official test entries:", len(official_test_entries))
    print("Official total entries:", len(official_train_entries) + len(official_test_entries))

    if len(official_train_entries) != EXPECTED_OFFICIAL_TRAIN:
        raise RuntimeError(
            f"Unexpected official train size: {len(official_train_entries)} "
            f"(expected {EXPECTED_OFFICIAL_TRAIN})."
        )
    if len(official_test_entries) != EXPECTED_OFFICIAL_TEST:
        raise RuntimeError(
            f"Unexpected official test size: {len(official_test_entries)} "
            f"(expected {EXPECTED_OFFICIAL_TEST})."
        )
    if len(official_train_entries) + len(official_test_entries) != EXPECTED_TOTAL:
        raise RuntimeError("Unexpected official PlantVillage total size.")

    local_index, duplicate_local_keys, class_names = build_local_index(dataset_root)

    duplicate_rows = []
    for (class_name, filename), matches in duplicate_local_keys.items():
        duplicate_rows.append(
            {
                "class_name": class_name,
                "filename": filename,
                "match_count": len(matches),
                "relative_paths": "|".join(row["relative_path"] for row in matches),
            }
        )
    pd.DataFrame(duplicate_rows).to_csv(
        output_dir / "official_split_duplicate_local_keys.csv",
        index=False,
    )

    if duplicate_local_keys:
        raise RuntimeError(
            "Duplicate local (class, filename) keys detected. See "
            "official_split_duplicate_local_keys.csv."
        )

    official_train_rows, train_missing = official_rows(
        "official_train",
        official_train_entries,
        local_index,
    )
    official_test_rows, test_missing = official_rows(
        "official_test",
        official_test_entries,
        local_index,
    )

    missing_rows = train_missing + test_missing
    pd.DataFrame(missing_rows).to_csv(
        output_dir / "official_split_unmatched_files.csv",
        index=False,
    )

    if missing_rows:
        raise RuntimeError(
            f"Could not uniquely match {len(missing_rows)} official files to "
            "the local Kaggle copy. See official_split_unmatched_files.csv."
        )

    local_paths_train = {row["relative_path"] for row in official_train_rows}
    local_paths_test = {row["relative_path"] for row in official_test_rows}
    if local_paths_train & local_paths_test:
        raise RuntimeError("A local image path appears in both official splits.")

    if len(local_paths_train | local_paths_test) != EXPECTED_TOTAL:
        raise RuntimeError("Official split does not resolve to 54,305 unique local images.")

    if args.skip_hash_audit:
        exact_pairs = None
    else:
        exact_pairs = exact_duplicate_audit(
            official_train_rows,
            official_test_rows,
        )
        pd.DataFrame(exact_pairs).to_csv(
            output_dir / "official_split_exact_cross_split_duplicates.csv",
            index=False,
        )
        if exact_pairs:
            raise RuntimeError(
                f"Found {len(exact_pairs)} byte-identical image pairs across "
                "the official train/test split. Review the audit CSV."
            )

    internal_rows = create_internal_validation(
        official_train_rows,
        args.validation_fraction,
    )
    final_test_rows = [
        {**row, "split": "test"} for row in official_test_rows
    ]

    manifest_rows = serialize_manifest_rows(internal_rows + final_test_rows)
    manifest_frame = pd.DataFrame(manifest_rows).sort_values(
        ["split", "class_index", "relative_path"]
    )
    manifest_path = output_dir / MANIFEST_PATH.name
    manifest_frame.to_csv(manifest_path, index=False)

    split_counts = manifest_frame["split"].value_counts().to_dict()
    class_split_counts = (
        manifest_frame.groupby(["class_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    class_split_counts.to_csv(
        output_dir / "official_leaf_safe_class_distribution.csv",
        index=False,
    )

    missing_class_splits = []
    for _, row in class_split_counts.iterrows():
        for split_name in ("train", "validation", "test"):
            if int(row.get(split_name, 0)) == 0:
                missing_class_splits.append(
                    {
                        "class_name": row["class_name"],
                        "split": split_name,
                    }
                )
    if missing_class_splits:
        pd.DataFrame(missing_class_splits).to_csv(
            output_dir / "official_leaf_safe_missing_class_splits.csv",
            index=False,
        )
        raise RuntimeError(
            "At least one class is absent from train/validation/test. See "
            "official_leaf_safe_missing_class_splits.csv."
        )

    summary = {
        "protocol": "official_leaf_preserving_test_with_internal_validation",
        "seed": SEED,
        "validation_fraction_of_official_train": args.validation_fraction,
        "official_train_examples": len(official_train_rows),
        "official_test_examples": len(official_test_rows),
        "resolved_local_images": len(local_paths_train | local_paths_test),
        "local_duplicate_keys": len(duplicate_local_keys),
        "unmatched_official_files": len(missing_rows),
        "exact_cross_official_split_duplicate_pairs": (
            None if exact_pairs is None else len(exact_pairs)
        ),
        "train_examples": int(split_counts.get("train", 0)),
        "validation_examples": int(split_counts.get("validation", 0)),
        "test_examples": int(split_counts.get("test", 0)),
        "num_classes": len(class_names),
        "official_test_locked": True,
        "official_test_leaf_preserving": True,
        "validation_note": (
            "Validation is a deterministic stratified image-level subset of "
            "the official training split. The final test split is the official "
            "leaf-preserving test split and is never used for training, "
            "checkpoint selection, or hyperparameter selection."
        ),
        "manifest_path": str(manifest_path),
    }

    summary_path = output_dir / SUMMARY_PATH.name
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("=" * 78)
    print("OFFICIAL SPLIT SUMMARY")
    print("=" * 78)
    print("Official train matched:", len(official_train_rows))
    print("Official test matched:", len(official_test_rows))
    print("Unmatched official files:", len(missing_rows))
    print("Duplicate local keys:", len(duplicate_local_keys))
    if exact_pairs is None:
        print("Exact cross-split hash audit: SKIPPED")
    else:
        print("Exact cross official train/test duplicate pairs:", len(exact_pairs))
    print("Internal train images:", split_counts.get("train", 0))
    print("Internal validation images:", split_counts.get("validation", 0))
    print("Locked official test images:", split_counts.get("test", 0))
    print("Classes:", len(class_names))
    print()
    print("PASS: official leaf-safe test manifest created.")
    print("Manifest:", manifest_path)
    print("Summary:", summary_path)
    print()
    print(
        "Important: the final test split is the official PlantVillage "
        "leaf-preserving benchmark split. The internal validation split is "
        "created only from official training images."
    )


if __name__ == "__main__":
    main()
