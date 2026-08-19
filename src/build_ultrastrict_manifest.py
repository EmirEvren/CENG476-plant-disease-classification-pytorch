from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"
SOURCE_MANIFEST = AUDIT_DIR / "official_leaf_safe_split_manifest.csv"
SOURCE_SUMMARY = AUDIT_DIR / "official_leaf_safe_split_summary.json"
STRICT_PAIRS = (
    AUDIT_DIR
    / "final_leafsafe_protocol_audit"
    / "dhash_strict_cross_split_pairs.csv"
)
OUTPUT_MANIFEST = AUDIT_DIR / "official_leaf_safe_split_manifest_ultrastrict.csv"
OUTPUT_SUMMARY = AUDIT_DIR / "official_leaf_safe_split_summary_ultrastrict.json"
OUTPUT_QUARANTINE = AUDIT_DIR / "official_leaf_safe_ultrastrict_quarantine.csv"

# Higher-priority held-out splits are protected. For every strict cross-split
# dHash pair, the lower-priority side is quarantined instead of modifying the
# locked official test. This makes the protocol conservative and deterministic.
SPLIT_PRIORITY = {
    "train": 0,
    "validation": 1,
    "test": 2,
}


def load_inputs():
    for path in (SOURCE_MANIFEST, SOURCE_SUMMARY, STRICT_PAIRS):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    manifest = pd.read_csv(SOURCE_MANIFEST, keep_default_na=False)
    pairs = pd.read_csv(STRICT_PAIRS, keep_default_na=False)
    with SOURCE_SUMMARY.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    required_manifest = {
        "split",
        "relative_path",
        "class_index",
        "class_name",
        "leaf_id",
        "sha256",
    }
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise RuntimeError(
            "Source manifest missing columns: "
            + ", ".join(sorted(missing_manifest))
        )

    required_pairs = {
        "hamming_distance",
        "path_a",
        "path_b",
    }
    missing_pairs = required_pairs - set(pairs.columns)
    if missing_pairs:
        raise RuntimeError(
            "Strict-pair CSV missing columns: "
            + ", ".join(sorted(missing_pairs))
        )

    return manifest, pairs, summary


def choose_quarantine_path(path_a, path_b, split_by_path):
    split_a = split_by_path[path_a]
    split_b = split_by_path[path_b]

    if split_a == split_b:
        raise RuntimeError(
            f"Strict pair unexpectedly lies in one split: {path_a} / {path_b}"
        )

    priority_a = SPLIT_PRIORITY[split_a]
    priority_b = SPLIT_PRIORITY[split_b]

    if priority_a == priority_b:
        raise RuntimeError("Split priorities must be unique.")

    # Remove the lower-priority side. Thus the official test is never altered;
    # validation is also protected relative to train.
    if priority_a < priority_b:
        return path_a, split_a, path_b, split_b
    return path_b, split_b, path_a, split_a


def main():
    manifest, pairs, source_summary = load_inputs()

    split_by_path = dict(
        zip(
            manifest["relative_path"].astype(str),
            manifest["split"].astype(str),
        )
    )

    if len(split_by_path) != len(manifest):
        raise RuntimeError("Source manifest contains duplicated paths.")

    quarantine_rows = []
    quarantine_paths = set()

    for _, pair in pairs.iterrows():
        path_a = str(pair["path_a"])
        path_b = str(pair["path_b"])

        if path_a not in split_by_path or path_b not in split_by_path:
            raise RuntimeError(
                "Strict-pair CSV references a path not present in the source manifest."
            )

        remove_path, remove_split, keep_path, keep_split = choose_quarantine_path(
            path_a,
            path_b,
            split_by_path,
        )
        quarantine_paths.add(remove_path)
        quarantine_rows.append(
            {
                "hamming_distance": int(pair["hamming_distance"]),
                "removed_path": remove_path,
                "removed_split": remove_split,
                "protected_path": keep_path,
                "protected_split": keep_split,
                "reason": "strict_cross_split_dhash_quarantine",
            }
        )

    quarantine_frame = pd.DataFrame(quarantine_rows).sort_values(
        ["removed_split", "removed_path", "protected_split", "protected_path"]
    )
    quarantine_frame.to_csv(OUTPUT_QUARANTINE, index=False)

    cleaned = manifest[
        ~manifest["relative_path"].astype(str).isin(quarantine_paths)
    ].copy()

    # The locked official test must be untouched.
    source_test = set(
        manifest.loc[manifest["split"] == "test", "relative_path"].astype(str)
    )
    cleaned_test = set(
        cleaned.loc[cleaned["split"] == "test", "relative_path"].astype(str)
    )
    if source_test != cleaned_test:
        raise RuntimeError("Ultra-strict quarantine modified the locked test set.")
    if len(cleaned_test) != 10709:
        raise RuntimeError("Locked official test must remain exactly 10,709 images.")

    # Every original strict dHash cross-split pair must be broken by quarantine.
    remaining_paths = set(cleaned["relative_path"].astype(str))
    surviving_pairs = []
    for _, pair in pairs.iterrows():
        a = str(pair["path_a"])
        b = str(pair["path_b"])
        if a in remaining_paths and b in remaining_paths:
            surviving_pairs.append((a, b))
    if surviving_pairs:
        raise RuntimeError(
            f"{len(surviving_pairs)} strict dHash pairs survived quarantine."
        )

    # Retain all 38 classes in every split.
    class_counts = (
        cleaned.groupby(["class_index", "class_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for split_name in ("train", "validation", "test"):
        if split_name not in class_counts.columns:
            raise RuntimeError(f"No examples remain in split: {split_name}")
        missing = class_counts[class_counts[split_name] == 0]
        if not missing.empty:
            raise RuntimeError(
                f"At least one class has no images in {split_name} after quarantine."
            )

    cleaned = cleaned.sort_values(
        ["split", "class_index", "relative_path"]
    ).reset_index(drop=True)
    cleaned.to_csv(OUTPUT_MANIFEST, index=False)

    counts = cleaned["split"].value_counts().to_dict()
    unique_removed_by_split = (
        manifest[
            manifest["relative_path"].astype(str).isin(quarantine_paths)
        ]["split"]
        .value_counts()
        .to_dict()
    )

    summary = dict(source_summary)
    summary.update(
        {
            # Keep the base protocol identifier for backward-compatible
            # evaluation code; the fields below record the stricter layer.
            "protocol": "conservative_official_leaf_preserving_test",
            "ultrastrict_dhash_quarantine": True,
            "ultrastrict_dhash_threshold": 4,
            "source_manifest": str(SOURCE_MANIFEST),
            "source_strict_pair_csv": str(STRICT_PAIRS),
            "strict_dhash_pairs_before_quarantine": int(len(pairs)),
            "strict_dhash_pairs_after_quarantine": 0,
            "quarantined_unique_images": int(len(quarantine_paths)),
            "quarantined_train_images": int(unique_removed_by_split.get("train", 0)),
            "quarantined_validation_images": int(
                unique_removed_by_split.get("validation", 0)
            ),
            "quarantined_test_images": int(unique_removed_by_split.get("test", 0)),
            "train_examples": int(counts.get("train", 0)),
            "validation_examples": int(counts.get("validation", 0)),
            "test_examples": int(counts.get("test", 0)),
            "official_test_locked": True,
            "ultrastrict_quarantine_policy": (
                "For every dHash<=4 cross-split pair, remove the lower-priority "
                "side with priority test > validation > train. The 10,709-image "
                "official test is never modified."
            ),
            "manifest_path": str(OUTPUT_MANIFEST),
        }
    )

    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 80)
    print("ULTRA-STRICT dHASH QUARANTINE MANIFEST")
    print("=" * 80)
    print("Strict cross-split dHash pairs before quarantine:", len(pairs))
    print("Unique images quarantined:", len(quarantine_paths))
    print("  train:", unique_removed_by_split.get("train", 0))
    print("  validation:", unique_removed_by_split.get("validation", 0))
    print("  test:", unique_removed_by_split.get("test", 0))
    print("Strict pairs remaining:", 0)
    print("Train images:", counts.get("train", 0))
    print("Validation images:", counts.get("validation", 0))
    print("Locked official test images:", counts.get("test", 0))
    print("Classes in all splits: 38")
    print()
    print("PASS: ultra-strict manifest created without modifying locked test.")
    print("Manifest:", OUTPUT_MANIFEST)
    print("Summary:", OUTPUT_SUMMARY)
    print("Quarantine log:", OUTPUT_QUARANTINE)


if __name__ == "__main__":
    main()
