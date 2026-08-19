from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import ImageFolder


SEED = 42
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10
NEAR_DUPLICATE_THRESHOLD = 4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "PlantVillage"
TRAIN_DIR = DATASET_ROOT / "train"
ORIGINAL_VAL_DIR = DATASET_ROOT / "val"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audit"
MANIFEST_PATH = OUTPUT_DIR / "leaf_grouped_split_manifest.csv"
SUMMARY_PATH = OUTPUT_DIR / "leaf_grouped_split_summary.json"
LEAF_MAP_CACHE = PROJECT_ROOT / ".cache" / "plantvillage" / "leaf-map.json"
LEAF_MAP_URL = (
    "https://raw.githubusercontent.com/spMohanty/"
    "PlantVillage-Dataset/master/leaf_grouping/leaf-map.json"
)


@dataclass(frozen=True)
class Record:
    index: int
    source_partition: str
    path: Path
    relative_path: str
    class_index: int
    class_name: str
    leaf_id: str


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a conservative PlantVillage split in which known physical "
            "leaf groups, byte-identical duplicates, and dHash near-duplicates "
            "cannot cross train/validation/test boundaries. Only images with "
            "an official leaf-map match are included by default."
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
        "--leaf-map",
        type=Path,
        default=None,
        help="Optional local official PlantVillage leaf-map.json.",
    )
    parser.add_argument(
        "--near-threshold",
        type=int,
        default=NEAR_DUPLICATE_THRESHOLD,
        help="64-bit dHash Hamming threshold. Supported range: 0..4.",
    )
    parser.add_argument(
        "--skip-near",
        action="store_true",
        help="Skip near-duplicate clustering. Exact duplicate and leaf grouping remain enabled.",
    )
    return parser.parse_args()


def validate_args(args):
    if not 0 <= args.near_threshold <= 4:
        raise ValueError("--near-threshold must be in the range 0..4.")


def load_leaf_map(explicit_path: Path | None):
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Leaf map not found: {path}")
    else:
        path = LEAF_MAP_CACHE
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading official PlantVillage leaf map...")
            request = urllib.request.Request(
                LEAF_MAP_URL,
                headers={"User-Agent": "CENG476-LeafGroupedSplit/1.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                path.write_bytes(response.read())

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle), path


def image_identifier(filename: str) -> str:
    value = filename.replace("_final_masked", "")
    if "___" in value:
        value = value.split("___")[-1]
    value = value.split("copy")[0]
    for extension in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        value = value.replace(extension, "")
    return value.strip().lower()


def resolve_leaf_id(path: Path, class_name: str, leaf_map) -> str | None:
    key = image_identifier(path.name)
    suggestions = leaf_map.get(key)
    if not suggestions:
        return None
    if isinstance(suggestions, str):
        suggestions = [suggestions]
    if len(suggestions) == 1:
        return str(suggestions[0])
    for suggestion in suggestions:
        if class_name in str(suggestion):
            return str(suggestion)
    return None


def load_records(dataset_root: Path, leaf_map):
    train_dir = dataset_root / "train"
    original_val_dir = dataset_root / "val"
    for directory in (train_dir, original_val_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")

    train_dataset = ImageFolder(train_dir)
    original_val_dataset = ImageFolder(original_val_dir)
    if train_dataset.class_to_idx != original_val_dataset.class_to_idx:
        raise RuntimeError("Train and val class mappings do not match.")

    raw_records = []
    for source_partition, dataset in (
        ("original_train", train_dataset),
        ("original_val", original_val_dataset),
    ):
        for path_string, class_index in dataset.samples:
            path = Path(path_string)
            class_name = dataset.classes[class_index]
            leaf_id = resolve_leaf_id(path, class_name, leaf_map)
            raw_records.append(
                (
                    source_partition,
                    path,
                    class_index,
                    class_name,
                    leaf_id,
                )
            )

    mapped_records = []
    unmapped_rows = []
    for source_partition, path, class_index, class_name, leaf_id in raw_records:
        relative_path = str(path.relative_to(dataset_root))
        if leaf_id is None:
            unmapped_rows.append(
                {
                    "source_partition": source_partition,
                    "relative_path": relative_path,
                    "class_index": class_index,
                    "class_name": class_name,
                }
            )
            continue
        mapped_records.append(
            Record(
                index=len(mapped_records),
                source_partition=source_partition,
                path=path,
                relative_path=relative_path,
                class_index=class_index,
                class_name=class_name,
                leaf_id=leaf_id,
            )
        )

    return mapped_records, unmapped_rows, train_dataset.classes, len(raw_records)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(path: Path) -> int:
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.int16)
    differences = pixels[:, :-1] > pixels[:, 1:]
    value = 0
    for bit, flag in enumerate(differences.reshape(-1)):
        if bool(flag):
            value |= 1 << bit
    return value


def hamming64(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def lsh_bands(value: int):
    sizes = (13, 13, 13, 13, 12)
    shift = 0
    result = []
    for band_index, size in enumerate(sizes):
        mask = (1 << size) - 1
        result.append((band_index, (value >> shift) & mask))
        shift += size
    return result


def union_known_leaf_groups(records, union_find):
    first_by_leaf = {}
    for record in records:
        previous = first_by_leaf.get(record.leaf_id)
        if previous is None:
            first_by_leaf[record.leaf_id] = record.index
        else:
            union_find.union(previous, record.index)


def union_exact_duplicates(records, union_find):
    first_by_hash = {}
    duplicate_pairs = 0
    for record in tqdm(records, desc="Exact duplicate clustering", unit="img"):
        digest = sha256_file(record.path)
        previous = first_by_hash.get(digest)
        if previous is None:
            first_by_hash[digest] = record.index
        else:
            union_find.union(previous, record.index)
            duplicate_pairs += 1
    return duplicate_pairs


def union_near_duplicates(records, union_find, threshold):
    hashes = [0] * len(records)
    for record in tqdm(records, desc="dHash calculation", unit="img"):
        hashes[record.index] = dhash64(record.path)

    buckets = defaultdict(list)
    seen_pairs = set()
    near_pairs = 0

    for record in tqdm(records, desc="Near-duplicate clustering", unit="img"):
        current_hash = hashes[record.index]
        candidates = set()
        for band_index, band_value in lsh_bands(current_hash):
            candidates.update(
                buckets[(record.class_index, band_index, band_value)]
            )

        for other_index in candidates:
            pair_key = (min(other_index, record.index), max(other_index, record.index))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            if hamming64(hashes[other_index], current_hash) <= threshold:
                union_find.union(other_index, record.index)
                near_pairs += 1

        for band_index, band_value in lsh_bands(current_hash):
            buckets[(record.class_index, band_index, band_value)].append(record.index)

    return near_pairs


def create_components(records, union_find):
    components = defaultdict(list)
    for record in records:
        components[union_find.find(record.index)].append(record)

    result = []
    for component_index, (_, members) in enumerate(sorted(components.items())):
        class_indices = {member.class_index for member in members}
        if len(class_indices) != 1:
            raise RuntimeError(
                "A strict duplicate/leaf component spans multiple classes, which "
                "indicates a dataset-label inconsistency."
            )
        result.append(
            {
                "group_id": f"strict_group_{component_index:06d}",
                "class_index": members[0].class_index,
                "class_name": members[0].class_name,
                "members": members,
                "size": len(members),
            }
        )
    return result


def choose_split_for_group(current_counts, target_counts, rng):
    split_names = ["train", "validation", "test"]
    scores = []
    for split_name in split_names:
        target = target_counts[split_name]
        current = current_counts[split_name]
        deficit_ratio = (target - current) / max(target, 1.0)
        scores.append((deficit_ratio, rng.random(), split_name))
    scores.sort(reverse=True)
    return scores[0][2]


def split_components(components, class_names):
    groups_by_class = defaultdict(list)
    for component in components:
        groups_by_class[component["class_index"]].append(component)

    assignment = {}
    class_summaries = []

    for class_index, class_name in enumerate(class_names):
        groups = groups_by_class[class_index]
        if len(groups) < 3:
            raise RuntimeError(
                f"Class {class_name} has fewer than three strict groups; "
                "cannot create train/validation/test splits safely."
            )

        total_images = sum(group["size"] for group in groups)
        targets = {
            "train": total_images * TRAIN_RATIO,
            "validation": total_images * VALIDATION_RATIO,
            "test": total_images * TEST_RATIO,
        }
        current = {"train": 0, "validation": 0, "test": 0}

        rng = random.Random(SEED + class_index)
        rng.shuffle(groups)
        groups.sort(key=lambda group: group["size"], reverse=True)

        # Seed every split with one group so each class is represented everywhere.
        seeded_order = ["train", "validation", "test"]
        for split_name, group in zip(seeded_order, groups[:3]):
            assignment[group["group_id"]] = split_name
            current[split_name] += group["size"]

        for group in groups[3:]:
            split_name = choose_split_for_group(current, targets, rng)
            assignment[group["group_id"]] = split_name
            current[split_name] += group["size"]

        class_summaries.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "mapped_images": total_images,
                "strict_groups": len(groups),
                "train_images": current["train"],
                "validation_images": current["validation"],
                "test_images": current["test"],
            }
        )

    return assignment, class_summaries


def write_manifest(records, components, assignment, output_dir):
    group_by_record_index = {}
    for component in components:
        for member in component["members"]:
            group_by_record_index[member.index] = component["group_id"]

    rows = []
    for record in records:
        group_id = group_by_record_index[record.index]
        rows.append(
            {
                "split": assignment[group_id],
                "relative_path": record.relative_path,
                "source_partition": record.source_partition,
                "class_index": record.class_index,
                "class_name": record.class_name,
                "leaf_id": record.leaf_id,
                "strict_group_id": group_id,
            }
        )

    frame = pd.DataFrame(rows).sort_values(
        ["split", "class_index", "strict_group_id", "relative_path"]
    )
    manifest_path = output_dir / MANIFEST_PATH.name
    frame.to_csv(manifest_path, index=False)
    return frame, manifest_path


def validate_manifest(frame):
    split_sets = {
        split_name: set(frame.loc[frame["split"] == split_name, "strict_group_id"])
        for split_name in ("train", "validation", "test")
    }
    if split_sets["train"] & split_sets["validation"]:
        raise RuntimeError("Strict group overlap: train/validation.")
    if split_sets["train"] & split_sets["test"]:
        raise RuntimeError("Strict group overlap: train/test.")
    if split_sets["validation"] & split_sets["test"]:
        raise RuntimeError("Strict group overlap: validation/test.")

    leaf_splits = frame.groupby("leaf_id")["split"].nunique()
    if int((leaf_splits > 1).sum()) != 0:
        raise RuntimeError("Known physical leaf leakage remains in the manifest.")

    return {
        split_name: int((frame["split"] == split_name).sum())
        for split_name in ("train", "validation", "test")
    }


def main():
    args = parse_args()
    validate_args(args)

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("STRICT LEAF-GROUPED PLANTVILLAGE SPLIT BUILDER")
    print("=" * 76)
    print("Dataset root:", dataset_root)
    print("Protocol: mapped physical leaves + exact duplicates + near duplicates")
    print("Ratios: 80% train / 10% validation / 10% test")
    print("Seed:", SEED)
    print()

    leaf_map, leaf_map_path = load_leaf_map(args.leaf_map)
    records, unmapped_rows, class_names, total_images = load_records(
        dataset_root,
        leaf_map,
    )

    mapped_images = len(records)
    coverage = mapped_images / total_images if total_images else 0.0
    print(f"Official leaf-map coverage: {coverage * 100:.2f}% ({mapped_images}/{total_images})")
    print(f"Excluded unmapped images: {len(unmapped_rows)}")
    print()

    pd.DataFrame(unmapped_rows).to_csv(
        output_dir / "leaf_grouped_unmapped_images.csv",
        index=False,
    )

    union_find = UnionFind(len(records))
    union_known_leaf_groups(records, union_find)
    exact_duplicate_pairs = union_exact_duplicates(records, union_find)

    if args.skip_near:
        near_duplicate_pairs = None
    else:
        near_duplicate_pairs = union_near_duplicates(
            records,
            union_find,
            args.near_threshold,
        )

    components = create_components(records, union_find)
    assignment, class_summaries = split_components(components, class_names)
    frame, manifest_path = write_manifest(
        records,
        components,
        assignment,
        output_dir,
    )
    split_counts = validate_manifest(frame)

    pd.DataFrame(class_summaries).to_csv(
        output_dir / "leaf_grouped_class_distribution.csv",
        index=False,
    )

    summary = {
        "protocol": "strict_leaf_grouped_mapped_only",
        "seed": SEED,
        "dataset_total_images": total_images,
        "mapped_images_used": mapped_images,
        "unmapped_images_excluded": len(unmapped_rows),
        "leaf_map_coverage": coverage,
        "leaf_map_path": str(leaf_map_path),
        "num_classes": len(class_names),
        "strict_groups": len(components),
        "exact_duplicate_links_merged": exact_duplicate_pairs,
        "near_duplicate_threshold": None if args.skip_near else args.near_threshold,
        "near_duplicate_links_merged": near_duplicate_pairs,
        "split_counts": split_counts,
        "known_leaf_cross_split_groups": 0,
        "strict_group_cross_split_groups": 0,
        "manifest_path": str(manifest_path),
    }
    summary_path = output_dir / SUMMARY_PATH.name
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("=" * 76)
    print("STRICT SPLIT SUMMARY")
    print("=" * 76)
    print("Mapped images used:", mapped_images)
    print("Unmapped images excluded:", len(unmapped_rows))
    print("Strict groups:", len(components))
    print("Exact duplicate links merged:", exact_duplicate_pairs)
    if near_duplicate_pairs is None:
        print("Near-duplicate clustering: SKIPPED")
    else:
        print(
            f"Near-duplicate links merged (dHash <= {args.near_threshold}):",
            near_duplicate_pairs,
        )
    print("Train images:", split_counts["train"])
    print("Validation images:", split_counts["validation"])
    print("Test images:", split_counts["test"])
    print("Known physical-leaf overlap: 0")
    print("Strict-group overlap: 0")
    print()
    print("PASS: strict grouped manifest created.")
    print("Manifest:", manifest_path)
    print("Summary:", summary_path)
    print()
    print(
        "Important: this is a conservative mapped-only evaluation protocol. "
        "Unmapped images are intentionally excluded rather than treated as "
        "independent leaves."
    )


if __name__ == "__main__":
    main()
