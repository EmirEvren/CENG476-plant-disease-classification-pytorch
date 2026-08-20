from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = PROJECT_ROOT / "data" / "external" / "PlantDoc-Windows" / "test"
COMPLETION_FILE = ".download_complete.json"
API_ROOT = "https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/contents/test"
RAW_ROOT = "https://raw.githubusercontent.com/pratikkayal/PlantDoc-Dataset/master"
USER_AGENT = "CENG476-plant-disease-full-control/1.0"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Only folders that have an explicit semantic mapping to PlantVillage are needed
# for the OOD experiment. Downloading just these folders avoids the repository's
# Windows-incompatible filenames during git checkout and avoids pulling train data.
PLANTDOC_FOLDERS = [
    "Apple Scab Leaf",
    "Apple leaf",
    "Apple rust leaf",
    "Bell_pepper leaf spot",
    "Bell_pepper leaf",
    "Blueberry leaf",
    "Cherry leaf",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Peach leaf",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Raspberry leaf",
    "Soyabean leaf",
    "Squash Powdery mildew leaf",
    "Strawberry leaf",
    "Tomato Early blight leaf",
    "Tomato Septoria leaf spot",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato leaf",
    "Tomato mold leaf",
    "grape leaf black rot",
    "grape leaf",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download only the mapped PlantDoc test folders through the GitHub "
            "Contents API while sanitizing Windows-incompatible filenames."
        )
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def request_json(url: str):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def safe_filename(name: str, source_path: str) -> str:
    # Windows disallows < > : " / \\ | ? * and trailing spaces/dots.
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name).rstrip(" .")
    if not cleaned:
        cleaned = "image"
    suffix = Path(cleaned).suffix
    stem = Path(cleaned).stem
    token = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:10]
    return f"{stem}__{token}{suffix}"


def raw_url_from_repo_path(repo_path: str) -> str:
    """Build a raw GitHub URL from the literal repository path.

    PlantDoc contains filenames whose literal names include percent sequences
    such as ``%20`` and characters such as ``?``. GitHub's ``download_url``
    field can be ambiguous for those names because a literal ``%20`` may be
    interpreted as a space and a literal ``?`` as a query delimiter. Quoting
    the repository path ourselves fixes this: percent signs become ``%25``,
    spaces become ``%20``, question marks become ``%3F``, while path slashes
    remain separators.
    """
    encoded_path = urllib.parse.quote(repo_path, safe="/")
    return f"{RAW_ROOT}/{encoded_path}"


def download_binary(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return "cached"

    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            temporary.replace(destination)
            return "downloaded"
        except Exception:
            if temporary.exists():
                temporary.unlink()
            if attempt == 3:
                raise
            time.sleep(2 * attempt)
    raise RuntimeError("unreachable")


def list_folder(folder_name: str):
    encoded = urllib.parse.quote(folder_name, safe="")
    url = f"{API_ROOT}/{encoded}?ref=master"
    payload = request_json(url)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected GitHub API payload for {folder_name}")
    return payload


def collect_files(items):
    files = []
    stack = list(items)
    while stack:
        item = stack.pop()
        item_type = item.get("type")
        if item_type == "dir":
            nested = request_json(item["url"])
            if isinstance(nested, list):
                stack.extend(nested)
            continue
        if item_type != "file":
            continue
        original_path = str(item.get("path", ""))
        suffix = Path(urllib.parse.unquote(original_path)).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            continue
        # Keep the repository path and SHA. Do not trust download_url for this
        # dataset because several literal filenames contain URL metacharacters.
        if original_path:
            files.append(item)
    return files


def main():
    args = parse_args()
    target = args.target.expanduser().resolve()
    marker = target.parent / COMPLETION_FILE

    if marker.is_file() and not args.force:
        with marker.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        expected = int(summary.get("downloaded_images", 0))
        actual = sum(
            1
            for path in target.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if expected > 0 and actual >= expected:
            print("PlantDoc Windows-safe test set already complete.")
            print("Images:", actual)
            print("Path:", target)
            return
        print("Completion marker is stale; resuming download.")

    target.mkdir(parents=True, exist_ok=True)
    total = 0
    downloaded = 0
    cached = 0
    folder_counts = {}

    print("=" * 88)
    print("WINDOWS-SAFE PLANTDOC TEST DOWNLOADER")
    print("=" * 88)
    print("Source: pratikkayal/PlantDoc-Dataset test folders via GitHub Contents API")
    print("Target:", target)
    print("Mapped folders:", len(PLANTDOC_FOLDERS))
    print()

    for folder_index, folder_name in enumerate(PLANTDOC_FOLDERS, start=1):
        print(f"[{folder_index:02d}/{len(PLANTDOC_FOLDERS):02d}] {folder_name}")
        items = collect_files(list_folder(folder_name))
        folder_target = target / folder_name
        count = 0
        for item in sorted(items, key=lambda row: str(row.get("path", ""))):
            original_path = str(item["path"])
            original_name = urllib.parse.unquote(original_path.rsplit("/", 1)[-1])
            filename = safe_filename(original_name, original_path)
            source_url = raw_url_from_repo_path(original_path)
            state = download_binary(source_url, folder_target / filename)
            count += 1
            total += 1
            if state == "downloaded":
                downloaded += 1
            else:
                cached += 1
        folder_counts[folder_name] = count
        print(f"    images: {count}")

    if total == 0:
        raise RuntimeError("PlantDoc downloader found zero mapped test images.")

    summary = {
        "source_repository": "pratikkayal/PlantDoc-Dataset",
        "source_split": "test",
        "download_method": (
            "GitHub Contents API listing + self-encoded raw URLs with "
            "Windows-safe local filenames"
        ),
        "mapped_folders": len(PLANTDOC_FOLDERS),
        "downloaded_images": total,
        "new_downloads_this_run": downloaded,
        "cached_images_this_run": cached,
        "folder_counts": folder_counts,
        "target": str(target),
        "note": (
            "Local filenames are sanitized only for Windows compatibility. "
            "Raw URLs are reconstructed from literal Git repository paths so "
            "filenames containing %, ?, spaces, and similar URL metacharacters "
            "are fetched without changing image bytes or class folders."
        ),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("=" * 88)
    print("PLANTDOC DOWNLOAD COMPLETE")
    print("=" * 88)
    print("Images:", total)
    print("New downloads:", downloaded)
    print("Cached:", cached)
    print("Completion marker:", marker)
    print("Test root:", target)


if __name__ == "__main__":
    main()
