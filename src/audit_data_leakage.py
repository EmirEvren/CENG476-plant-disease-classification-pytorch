from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.datasets import ImageFolder


SEED = 42
DEFAULT_DATASET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "PlantVillage"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "audit"
)
DEFAULT_LEAF_MAP_URL = (
    "https://huggingface.co/datasets/mohanty/PlantVillage/"
    "resolve/main/leaf_grouping/leaf-map.json"
)


@dataclass(frozen=True)
class ImageRecord:
    split: str
    class_name: str
    path: Path
    relative_path: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit the PlantVillage train/validation/test protocol for "
            "exact duplicates, near-duplicates, and same-physical-leaf "
            "leakage."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path containing PlantVillage/train and PlantVillage/val.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for audit JSON/CSV outputs.",
    )
    parser.add_argument(
        "--near-threshold",
        type=int,
        default=4,
        help=(
            "Maximum 64-bit dHash Hamming distance for a near-duplicate. "
            "Supported range is 0..4."
        ),
    )
    parser.add_argument(
        "--skip-near",
        action="store_true",
        help="Skip perceptual near-duplicate analysis.",
    )
    parser.add_argument(
        "--leaf-map",
        type=Path,
        default=None,
        help="Optional local official PlantVillage leaf-map.json.",
    )
    parser.add_argument(
        "--no-leaf-map-download",
        action="store_true",
        help="Do not download the official leaf map if --leaf-map is absent.",
    )
    parser.add_argument(
        "--max-near-pairs",
        type=int,
        default=5000,
        help="Maximum near-duplicate pairs saved to CSV.",
    )
    return parser.parse_args()


def validate_args(args):
    if not 0 <= args.near_threshold <= 4:
        raise ValueError(
            "--near-threshold must be in 0..4. "
            "The LSH candidate search is guaranteed complete only up to 4."
        )
    if args.max_near_pairs <= 0:
        raise ValueError("--max-near-pairs must be positive.")


def build_split_records(dataset_root: Path):
    train_dir = dataset_root / "train"
    original_val_dir = dataset_root / "val"

    for directory in (train_dir, original_val_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {directory}")

    train_dataset = ImageFolder(train_dir)
    original_val_dataset = ImageFolder(original_val_dir)

    if train_dataset.class_to_idx != original_val_dataset.class_to_idx:
        raise RuntimeError("Train and val class mappings do not match.")

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
        raise RuntimeError("Validation/test index overlap detected.")

    records = []

    for path_str, class_idx in train_dataset.samples:
        path = Path(path_str)
        records.append(
            ImageRecord(
                split="train",
                class_name=train_dataset.classes[class_idx],
                path=path,
                relative_path=str(path.relative_to(dataset_root)),
            )
        )

    for split_name, indices in (
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        for index in indices:
            path_str, class_idx = original_val_dataset.samples[index]
            path = Path(path_str)
            records.append(
                ImageRecord(
                    split=split_name,
                    class_name=original_val_dataset.classes[class_idx],
                    path=path,
                    relative_path=str(path.relative_to(dataset_root)),
                )
            )

    return records, train_dataset.classes


def sha256_file(path: Path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def audit_exact_duplicates(records):
    by_hash = defaultdict(list)

    for record in tqdm(records, desc="SHA-256 exact duplicate audit", unit="img"):
        by_hash[sha256_file(record.path)].append(record)

    cross_split_groups = []
    pair_rows = []

    for digest, group in by_hash.items():
        splits = sorted({r.split for r in group})
        if len(splits) <= 1:
            continue

        cross_split_groups.append(
            {
                "sha256": digest,
                "splits": "|".join(splits),
                "count": len(group),
                "classes": "|".join(sorted({r.class_name for r in group})),
                "paths": "|".join(r.relative_path for r in group),
            }
        )

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                left, right = group[i], group[j]
                if left.split == right.split:
                    continue
                pair_rows.append(
                    {
                        "sha256": digest,
                        "split_a": left.split,
                        "split_b": right.split,
                        "class_a": left.class_name,
                        "class_b": right.class_name,
                        "path_a": left.relative_path,
                        "path_b": right.relative_path,
                    }
                )

    return cross_split_groups, pair_rows


def dhash64(path: Path):
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())

    value = 0
    bit = 0
    for row in range(8):
        base = row * 9
        for col in range(8):
            if pixels[base + col] > pixels[base + col + 1]:
                value |= 1 << bit
            bit += 1
    return value


def hamming64(a: int, b: int):
    return (a ^ b).bit_count()


def lsh_bands(value: int):
    # Five bands: 13 + 13 + 13 + 13 + 12 = 64 bits.
    # If Hamming distance <= 4, at least one band must be unchanged.
    sizes = (13, 13, 13, 13, 12)
    shift = 0
    result = []
    for band_index, size in enumerate(sizes):
        mask = (1 << size) - 1
        result.append((band_index, (value >> shift) & mask))
        shift += size
    return result


def audit_near_duplicates(records, threshold: int, max_pairs: int):
    hashes = []
    for record in tqdm(records, desc="dHash perceptual audit", unit="img"):
        try:
            hashes.append(dhash64(record.path))
        except Exception as exc:
            raise RuntimeError(f"Could not read image: {record.path}") from exc

    buckets = defaultdict(list)
    seen_pairs = set()
    near_pairs = []
    total_pairs = 0

    for idx, record in enumerate(tqdm(records, desc="Near-duplicate matching", unit="img")):
        current_hash = hashes[idx]
        candidates = set()

        for band_index, band_value in lsh_bands(current_hash):
            candidates.update(
                buckets[(record.class_name, band_index, band_value)]
            )

        for other_idx in candidates:
            other = records[other_idx]
            if other.split == record.split:
                continue

            pair_key = (
                min(other.relative_path, record.relative_path),
                max(other.relative_path, record.relative_path),
            )
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            distance = hamming64(hashes[other_idx], current_hash)
            if distance <= threshold:
                total_pairs += 1
                if len(near_pairs) < max_pairs:
                    near_pairs.append(
                        {
                            "hamming_distance": distance,
                            "split_a": other.split,
                            "split_b": record.split,
                            "class_name": record.class_name,
                            "path_a": other.relative_path,
                            "path_b": record.relative_path,
                        }
                    )

        for band_index, band_value in lsh_bands(current_hash):
            buckets[(record.class_name, band_index, band_value)].append(idx)

    near_pairs.sort(
        key=lambda row: (
            row["hamming_distance"],
            row["class_name"],
            row["path_a"],
            row["path_b"],
        )
    )
    return total_pairs, near_pairs


def filename_to_identifier(filename: str):
    value = filename.replace("_final_masked", "")
    if "___" in value:
        value = value.split("___")[-1]
    value = value.split("copy")[0]
    value = (
        value.replace(".jpg", "")
        .replace(".JPG", "")
        .replace(".jpeg", "")
        .replace(".JPEG", "")
        .replace(".png", "")
        .replace(".PNG", "")
        .strip()
    )
    return value.lower()


def get_leaf_map(args, output_dir: Path):
    if args.leaf_map is not None:
        if not args.leaf_map.is_file():
            raise FileNotFoundError(f"Leaf map not found: {args.leaf_map}")
        source = args.leaf_map
    elif args.no_leaf_map_download:
        return None, "disabled"
    else:
        cache_dir = output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        source = cache_dir / "official_leaf-map.json"

        if not source.is_file():
            print("Downloading official PlantVillage leaf map...")
            try:
                request = urllib.request.Request(
                    DEFAULT_LEAF_MAP_URL,
                    headers={"User-Agent": "CENG476-PlantVillage-Audit/1.0"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    source.write_bytes(response.read())
            except Exception as exc:
                print(
                    "WARNING: official leaf-map download failed. "
                    "Same-leaf audit will be skipped."
                )
                print(f"Reason: {exc}")
                return None, "download_failed"

    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle), str(source)


def resolve_leaf_id(record: ImageRecord, leaf_map):
    identifier = filename_to_identifier(record.path.name)
    suggestions = leaf_map.get(identifier)

    if not suggestions:
        return None

    if isinstance(suggestions, str):
        suggestions = [suggestions]

    if len(suggestions) == 1:
        return suggestions[0]

    for suggestion in suggestions:
        if record.class_name in suggestion:
            return suggestion

    return None


def audit_leaf_leakage(records, leaf_map):
    if leaf_map is None:
        return {
            "mapped_images": 0,
            "coverage": 0.0,
            "cross_split_leaf_groups": [],
        }

    by_leaf = defaultdict(list)
    mapped = 0

    for record in records:
        leaf_id = resolve_leaf_id(record, leaf_map)
        if leaf_id is None:
            continue
        mapped += 1
        by_leaf[leaf_id].append(record)

    leaks = []
    for leaf_id, group in by_leaf.items():
        splits = sorted({r.split for r in group})
        if len(splits) <= 1:
            continue
        leaks.append(
            {
                "leaf_id": leaf_id,
                "splits": "|".join(splits),
                "image_count": len(group),
                "classes": "|".join(sorted({r.class_name for r in group})),
                "paths": "|".join(r.relative_path for r in group),
            }
        )

    leaks.sort(key=lambda row: (-row["image_count"], row["leaf_id"]))
    return {
        "mapped_images": mapped,
        "coverage": mapped / len(records) if records else 0.0,
        "cross_split_leaf_groups": leaks,
    }


def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    args = parse_args()
    validate_args(args)

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PLANTVILLAGE DATA LEAKAGE AUDIT")
    print("=" * 72)
    print("Dataset root:", dataset_root)
    print("Seed:", SEED)
    print()

    records, class_names = build_split_records(dataset_root)

    split_counts = pd.Series([r.split for r in records]).value_counts().to_dict()
    print("Split counts:", split_counts)
    print("Classes:", len(class_names))
    print()

    exact_groups, exact_pairs = audit_exact_duplicates(records)
    save_csv(output_dir / "exact_cross_split_groups.csv", exact_groups)
    save_csv(output_dir / "exact_cross_split_pairs.csv", exact_pairs)

    if args.skip_near:
        near_total = None
        near_pairs = []
    else:
        near_total, near_pairs = audit_near_duplicates(
            records,
            threshold=args.near_threshold,
            max_pairs=args.max_near_pairs,
        )
        save_csv(output_dir / "near_duplicate_pairs.csv", near_pairs)

    leaf_map, leaf_map_source = get_leaf_map(args, output_dir)
    leaf_result = audit_leaf_leakage(records, leaf_map)
    save_csv(
        output_dir / "same_leaf_cross_split_groups.csv",
        leaf_result["cross_split_leaf_groups"],
    )

    summary = {
        "dataset_root": str(dataset_root),
        "seed": SEED,
        "split_counts": split_counts,
        "num_classes": len(class_names),
        "exact_cross_split_hash_groups": len(exact_groups),
        "exact_cross_split_pairs": len(exact_pairs),
        "near_duplicate_threshold": None if args.skip_near else args.near_threshold,
        "near_duplicate_cross_split_pairs": near_total,
        "near_duplicate_pairs_saved": len(near_pairs),
        "leaf_map_source": leaf_map_source,
        "leaf_map_mapped_images": leaf_result["mapped_images"],
        "leaf_map_coverage": leaf_result["coverage"],
        "same_leaf_cross_split_groups": len(
            leaf_result["cross_split_leaf_groups"]
        ),
    }

    with (output_dir / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("=" * 72)
    print("AUDIT SUMMARY")
    print("=" * 72)
    print(
        "Exact cross-split duplicate hash groups:",
        summary["exact_cross_split_hash_groups"],
    )
    print(
        "Exact cross-split duplicate pairs:",
        summary["exact_cross_split_pairs"],
    )

    if near_total is None:
        print("Near-duplicate audit: SKIPPED")
    else:
        print(
            f"Near-duplicate cross-split pairs (dHash <= {args.near_threshold}):",
            near_total,
        )

    if leaf_map is None:
        print("Same-physical-leaf audit: NOT AVAILABLE")
    else:
        print(
            "Leaf-map coverage:",
            f'{summary["leaf_map_coverage"] * 100:.2f}%',
            f'({summary["leaf_map_mapped_images"]}/{len(records)})',
        )
        print(
            "Same-leaf cross-split groups:",
            summary["same_leaf_cross_split_groups"],
        )

    print()
    if summary["exact_cross_split_pairs"] > 0:
        print("FAIL: exact duplicate images occur in different splits.")
    else:
        print("PASS: no byte-identical cross-split images found.")

    if near_total is not None:
        if near_total > 0:
            print(
                "WARN: perceptually near-identical cross-split image pairs "
                "were found. Review near_duplicate_pairs.csv."
            )
        else:
            print(
                "PASS: no near-duplicate pairs were found at the selected "
                "dHash threshold."
            )

    if leaf_map is not None:
        if summary["same_leaf_cross_split_groups"] > 0:
            print(
                "FAIL: images mapped to the same physical leaf occur in "
                "different splits."
            )
        else:
            print(
                "PASS on mapped images: no same-physical-leaf cross-split "
                "groups found."
            )
        if summary["leaf_map_coverage"] < 0.95:
            print(
                "NOTE: leaf-map coverage is below 95%, so the leaf-level "
                "audit is partial."
            )

    print()
    print("Reports saved to:", output_dir)


if __name__ == "__main__":
    main()
