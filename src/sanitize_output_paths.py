import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"


def _portable_path(value):
    if not isinstance(value, str):
        return value

    normalized = value.replace("\\", "/")
    marker = "/outputs/"
    lower = normalized.lower()
    marker_index = lower.find(marker)
    if marker_index >= 0:
        return normalized[marker_index + 1 :]

    if normalized.lower().startswith("outputs/"):
        return normalized

    return value


def _sanitize(value):
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return _portable_path(value)


def main():
    changed = []

    for json_path in sorted(OUTPUTS_ROOT.rglob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            original = json.load(handle)

        sanitized = _sanitize(original)
        if sanitized == original:
            continue

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                sanitized,
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")

        changed.append(json_path.relative_to(PROJECT_ROOT))

    if not changed:
        print("No absolute output paths found.")
        return

    print("Sanitized files:")
    for path in changed:
        print(" -", path)


if __name__ == "__main__":
    main()
