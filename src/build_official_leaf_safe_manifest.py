from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd
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
LEAF_MAP_URL = (
    "https://raw.githubusercontent.com/spMohanty/"
    "PlantVillage-Dataset/master/leaf_grouping/leaf-map.json"
)

EXPECTED_OFFICIAL_TRAIN = 43596
EXPECTED_OFFICIAL_TEST = 10709
EXPECTED_TOTAL = 54305

MANIFEST_PATH = OUTPUT_DIR / "official_leaf_safe_split_manifest.csv"
SUMMARY_PATH = OUTPUT_DIR / "official_leaf_safe_split_summary.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a conservative local PlantVillage manifest from the official "
            "leaf-preserving color train/test split. Byte-identical official "
            "train images that collide with the locked official test are removed "
            "from training. Internal validation is group-aware for every image "
            "that can be mapped to an official physical leaf ID."
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
        help="Target fraction of cleaned official train used for validation.",
    )
    return parser.parse_args()


def validate_args(args):
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5.")


def download_file(url: str, destination: Path) -> Path:
    if destination.is_file():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading:", url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CENG476-OfficialPlantVillageSplit/2.0"},
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
    return parts[-2], parts[-1]


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
            by_key[local_key(class_name, path.name)].append(
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
        matches = local_index.get(local_key(class_name, filename), [])

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

        rows.append(
            {
                **matches[0],
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


def attach_hashes(rows, description: str):
    output = []
    for row in tqdm(rows, desc=description, unit="img"):
        output.append({**row, "sha256": sha256_file(row["path"])})
    return output


def cross_hash_pairs(train_rows, test_rows):
    train_by_hash = defaultdict(list)
    test_by_hash = defaultdict(list)

    for row in train_rows:
        train_by_hash[row["sha256"]].append(row)
    for row in test_rows:
        test_by_hash[row["sha256"]].append(row)

    pairs = []
    for digest in sorted(set(train_by_hash) & set(test_by_hash)):
        for train_row in train_by_hash[digest]:
            for test_row in test_by_hash[digest]:
                pairs.append(
                    {
                        "sha256": digest,
                        "train_relative_path": train_row["relative_path"],
                        "test_relative_path": test_row["relative_path"],
                        "train_class": train_row["class_name"],
                        "test_class": test_row["class_name"],
                    }
                )
    return pairs


def image_identifier(filename: str) -> str:
    value = filename.replace("_final_masked", "")
    if "___" in value:
        value = value.split("___")[-1]
    value = value.split("copy")[0]
    for extension in (
        ".jpg",
        ".JPG",
        ".jpeg",
        ".JPEG",
        ".png",
        ".PNG",
    ):
        value = value.replace(extension, "")
    return value.strip().lower()


def load_leaf_map():
    path = download_file(
        LEAF_MAP_URL,
        CACHE_DIR / "leaf-map.json",
    )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle), path


def resolve_leaf_id(row, leaf_map):
    key = image_identifier(Path(row["relative_path"]).name)
    suggestions = leaf_map.get(key)
    if not suggestions:
        return None
    if isinstance(suggestions, str):
        suggestions = [suggestions]
    if len(suggestions) == 1:
        return str(suggestions[0])
    for suggestion in suggestions:
        if row["class_name"] in str(suggestion):
            return str(suggestion)
    return None


def attach_leaf_ids(rows, leaf_map):
    return [
        {**row, "leaf_id": resolve_leaf_id(row, leaf_map)}
        for row in rows
    ]


def mapped_leaf_overlap(train_rows, test_rows):
    train_leaf_ids = {
        row["leaf_id"] for row in train_rows if row["leaf_id"] is not None
    }
    test_leaf_ids = {
        row["leaf_id"] for row in test_rows if row["leaf_id"] is not None
    }
    return train_leaf_ids & test_leaf_ids


def remove_test_collisions(train_rows, test_rows, output_dir: Path):
    initial_pairs = cross_hash_pairs(train_rows, test_rows)
    pd.DataFrame(initial_pairs).to_csv(
        output_dir / "official_split_exact_cross_split_duplicates.csv",
        index=False,
    )

    test_hashes = {row["sha256"] for row in test_rows}
    excluded = [row for row in train_rows if row["sha256"] in test_hashes]
    cleaned = [row for row in train_rows if row["sha256"] not in test_hashes]

    pd.DataFrame(
        [
            {
                "reason": "byte_identical_to_locked_official_test",
                "relative_path": row["relative_path"],
                "class_name": row["class_name"],
                "sha256": row["sha256"],
            }
            for row in excluded
        ]
    ).to_csv(
        output_dir / "official_split_excluded_train_exact_duplicates.csv",
        index=False,
    )

    remaining_pairs = cross_hash_pairs(cleaned, test_rows)
    if remaining_pairs:
        raise RuntimeError(
            "Internal error: exact duplicate collisions remain after cleanup."
        )

    return cleaned, excluded, initial_pairs


def remove_mapped_leaf_collisions(train_rows, test_rows, output_dir: Path):
    overlaps = mapped_leaf_overlap(train_rows, test_rows)

    excluded = [
        row
        for row in train_rows
        if row["leaf_id"] is not None and row["leaf_id"] in overlaps
    ]
    cleaned = [
        row
        for row in train_rows
        if not (row["leaf_id"] is not None and row["leaf_id"] in overlaps)
    ]

    pd.DataFrame(
        [
            {
                "reason": "mapped_physical_leaf_also_present_in_locked_test",
                "relative_path": row["relative_path"],
                "class_name": row["class_name"],
                "leaf_id": row["leaf_id"],
            }
            for row in excluded
        ]
    ).to_csv(
        output_dir / "official_split_excluded_train_leaf_collisions.csv",
        index=False,
    )

    if mapped_leaf_overlap(cleaned, test_rows):
        raise RuntimeError(
            "Internal error: mapped leaf collisions remain after cleanup."
        )

    return cleaned, excluded, overlaps


def group_key_for_validation(row):
    if row["leaf_id"] is not None:
        return f"leaf::{row['leaf_id']}"
    return f"unmapped_hash::{row['sha256']}"


def create_group_aware_validation(train_rows, validation_fraction, class_names):
    groups_by_class = defaultdict(lambda: defaultdict(list))
    for row in train_rows:
        groups_by_class[row["class_index"]][
            group_key_for_validation(row)
        ].append(row)

    output = []
    class_summary = []

    for class_index, class_name in enumerate(class_names):
        groups = list(groups_by_class[class_index].items())
        if len(groups) < 2:
            raise RuntimeError(
                f"Class {class_name} has fewer than two independent validation "
                "groups after cleanup."
            )

        rng = random.Random(SEED + class_index)
        rng.shuffle(groups)

        total_images = sum(len(rows) for _, rows in groups)
        target_validation = max(1, round(total_images * validation_fraction))

        validation_groups = []
        validation_count = 0
        remaining_group_count = len(groups)

        for group_key, rows in groups:
            remaining_group_count -= 1
            if remaining_group_count == 0:
                break
            if validation_count >= target_validation:
                break
            validation_groups.append(group_key)
            validation_count += len(rows)

        validation_group_set = set(validation_groups)
        if not validation_group_set:
            validation_group_set.add(groups[0][0])

        train_count = 0
        validation_count = 0
        for group_key, rows in groups:
            split = "validation" if group_key in validation_group_set else "train"
            for row in rows:
                output.append({**row, "split": split})
            if split == "validation":
                validation_count += len(rows)
            else:
                train_count += len(rows)

        if train_count == 0 or validation_count == 0:
            raise RuntimeError(
                f"Could not create non-empty train/validation groups for {class_name}."
            )

        class_summary.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "clean_official_train_images": total_images,
                "validation_groups": len(validation_group_set),
                "train_images": train_count,
                "validation_images": validation_count,
            }
        )

    return output, class_summary


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
                "leaf_id": "" if row["leaf_id"] is None else row["leaf_id"],
                "sha256": row["sha256"],
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
    print("CONSERVATIVE OFFICIAL PLANTVILLAGE SPLIT BUILDER")
    print("=" * 78)
    print("Dataset root:", dataset_root)
    print("Seed:", SEED)
    print(
        "Protocol: official leaf-preserving train/test + removal of exact "
        "train/test collisions + mapped-leaf-safe internal validation"
    )
    print()

    official_train_path = download_file(
        OFFICIAL_TRAIN_URL,
        CACHE_DIR / "color_train.txt",
    )
    official_test_path = download_file(
        OFFICIAL_TEST_URL,
        CACHE_DIR / "color_test.txt",
    )
    leaf_map, leaf_map_path = load_leaf_map()

    official_train_entries = read_split_file(official_train_path)
    official_test_entries = read_split_file(official_test_path)

    print("Official train entries:", len(official_train_entries))
    print("Official test entries:", len(official_test_entries))
    print(
        "Official total entries:",
        len(official_train_entries) + len(official_test_entries),
    )

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
                "relative_paths": "|".join(
                    row["relative_path"] for row in matches
                ),
            }
        )
    pd.DataFrame(duplicate_rows).to_csv(
        output_dir / "official_split_duplicate_local_keys.csv",
        index=False,
    )
    if duplicate_local_keys:
        raise RuntimeError(
            "Duplicate local (class, filename) keys detected. See audit CSV."
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
            "the local Kaggle copy. See audit CSV."
        )

    local_paths_train = {row["relative_path"] for row in official_train_rows}
    local_paths_test = {row["relative_path"] for row in official_test_rows}
    if local_paths_train & local_paths_test:
        raise RuntimeError("A local image path appears in both official splits.")
    if len(local_paths_train | local_paths_test) != EXPECTED_TOTAL:
        raise RuntimeError(
            "Official split does not resolve to 54,305 unique local paths."
        )

    official_train_rows = attach_hashes(
        official_train_rows,
        "Hashing official train",
    )
    official_test_rows = attach_hashes(
        official_test_rows,
        "Hashing official test",
    )

    official_train_rows, exact_excluded, initial_exact_pairs = (
        remove_test_collisions(
            official_train_rows,
            official_test_rows,
            output_dir,
        )
    )

    official_train_rows = attach_leaf_ids(official_train_rows, leaf_map)
    official_test_rows = attach_leaf_ids(official_test_rows, leaf_map)

    train_mapped = sum(row["leaf_id"] is not None for row in official_train_rows)
    test_mapped = sum(row["leaf_id"] is not None for row in official_test_rows)

    official_train_rows, leaf_excluded, initial_leaf_overlaps = (
        remove_mapped_leaf_collisions(
            official_train_rows,
            official_test_rows,
            output_dir,
        )
    )

    internal_rows, class_summary = create_group_aware_validation(
        official_train_rows,
        args.validation_fraction,
        class_names,
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

    pd.DataFrame(class_summary).to_csv(
        output_dir / "official_leaf_safe_class_distribution.csv",
        index=False,
    )

    split_counts = manifest_frame["split"].value_counts().to_dict()
    class_split_counts = (
        manifest_frame.groupby(["class_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
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
            "At least one class is absent from train/validation/test."
        )

    manifest_train_hashes = set(
        manifest_frame.loc[
            manifest_frame["split"].isin(["train", "validation"]),
            "sha256",
        ]
    )
    manifest_test_hashes = set(
        manifest_frame.loc[manifest_frame["split"] == "test", "sha256"]
    )
    if manifest_train_hashes & manifest_test_hashes:
        raise RuntimeError("Final manifest still has exact train/test leakage.")

    train_val_leaf = set(
        manifest_frame.loc[
            manifest_frame["split"].isin(["train", "validation"])
            & (manifest_frame["leaf_id"] != ""),
            "leaf_id",
        ]
    )
    test_leaf = set(
        manifest_frame.loc[
            (manifest_frame["split"] == "test")
            & (manifest_frame["leaf_id"] != ""),
            "leaf_id",
        ]
    )
    if train_val_leaf & test_leaf:
        raise RuntimeError("Final manifest still has mapped leaf train/test leakage.")

    mapped_train_validation_overlap = set(
        manifest_frame.loc[
            (manifest_frame["split"] == "train")
            & (manifest_frame["leaf_id"] != ""),
            "leaf_id",
        ]
    ) & set(
        manifest_frame.loc[
            (manifest_frame["split"] == "validation")
            & (manifest_frame["leaf_id"] != ""),
            "leaf_id",
        ]
    )
    if mapped_train_validation_overlap:
        raise RuntimeError("Mapped physical leaf crosses train/validation.")

    summary = {
        "protocol": "conservative_official_leaf_preserving_test",
        "seed": SEED,
        "validation_fraction_target": args.validation_fraction,
        "official_train_entries": len(official_train_entries),
        "official_test_entries": len(official_test_entries),
        "resolved_local_images_before_cleanup": EXPECTED_TOTAL,
        "local_duplicate_keys": len(duplicate_local_keys),
        "unmatched_official_files": len(missing_rows),
        "initial_exact_cross_official_split_pairs": len(initial_exact_pairs),
        "train_images_excluded_for_exact_test_collision": len(exact_excluded),
        "initial_mapped_leaf_cross_official_split_groups": len(initial_leaf_overlaps),
        "train_images_excluded_for_mapped_leaf_test_collision": len(leaf_excluded),
        "official_train_leaf_map_coverage_after_exact_cleanup": (
            train_mapped / max(len(official_train_rows) + len(leaf_excluded), 1)
        ),
        "official_test_leaf_map_coverage": (
            test_mapped / max(len(official_test_rows), 1)
        ),
        "final_exact_train_test_overlap_hashes": 0,
        "final_mapped_leaf_train_test_overlap_groups": 0,
        "final_mapped_leaf_train_validation_overlap_groups": 0,
        "train_examples": int(split_counts.get("train", 0)),
        "validation_examples": int(split_counts.get("validation", 0)),
        "test_examples": int(split_counts.get("test", 0)),
        "num_classes": len(class_names),
        "official_test_locked": True,
        "official_test_documented_leaf_preserving": True,
        "leaf_map_path": str(leaf_map_path),
        "validation_note": (
            "Validation is created only inside the cleaned official training "
            "partition. Images with a resolved official leaf ID are grouped by "
            "physical leaf; unresolved images are grouped at least by exact "
            "SHA-256 identity. The locked official test is never used for "
            "training, scheduling, checkpoint selection, or hyperparameter "
            "selection."
        ),
        "manifest_path": str(manifest_path),
    }

    summary_path = output_dir / SUMMARY_PATH.name
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("=" * 78)
    print("CONSERVATIVE OFFICIAL SPLIT SUMMARY")
    print("=" * 78)
    print("Official train matched:", len(official_train_entries))
    print("Official test matched:", len(official_test_entries))
    print("Unmatched official files:", len(missing_rows))
    print("Duplicate local keys:", len(duplicate_local_keys))
    print("Initial exact official train/test pairs:", len(initial_exact_pairs))
    print("Train images removed for exact test collision:", len(exact_excluded))
    print("Initial mapped physical-leaf overlaps:", len(initial_leaf_overlaps))
    print("Train images removed for mapped-leaf test collision:", len(leaf_excluded))
    print(
        "Official train leaf-map coverage:",
        f"{summary['official_train_leaf_map_coverage_after_exact_cleanup'] * 100:.2f}%",
    )
    print(
        "Official test leaf-map coverage:",
        f"{summary['official_test_leaf_map_coverage'] * 100:.2f}%",
    )
    print("Final exact train/test overlap hashes: 0")
    print("Final mapped leaf train/test overlap groups: 0")
    print("Final mapped leaf train/validation overlap groups: 0")
    print("Internal train images:", split_counts.get("train", 0))
    print("Internal validation images:", split_counts.get("validation", 0))
    print("Locked official test images:", split_counts.get("test", 0))
    print("Classes:", len(class_names))
    print()
    print("PASS: conservative official manifest created.")
    print("Manifest:", manifest_path)
    print("Summary:", summary_path)
    print()
    print(
        "Interpretation: the official test remains locked. Any byte-identical "
        "official-train collision is removed from training, and every physical "
        "leaf that can be resolved from the official leaf map is prevented from "
        "crossing the train/test and train/validation boundaries."
    )


if __name__ == "__main__":
    main()
