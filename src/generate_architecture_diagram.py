from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "baseline_cnn_architecture.png"
)


def main():
    blocks = [
        "Input\n3 x 224 x 224",
        "Conv 3x3, 32\nBatchNorm + ReLU\nMaxPool 2x2",
        "Conv 3x3, 64\nBatchNorm + ReLU\nMaxPool 2x2",
        "Conv 3x3, 128\nBatchNorm + ReLU\nMaxPool 2x2",
        "Conv 3x3, 256\nBatchNorm + ReLU\nMaxPool 2x2",
        "AdaptiveAvgPool\n1 x 1",
        "Dropout\np = 0.40",
        "Linear\n256 -> 38",
        "Output logits\n38 classes",
    ]

    figure, axis = plt.subplots(figsize=(16, 4.8))
    axis.set_xlim(0, len(blocks) * 2.1)
    axis.set_ylim(0, 4)
    axis.axis("off")

    width = 1.65
    height = 1.55
    y = 1.25

    for index, label in enumerate(blocks):
        x = 0.25 + index * 2.05
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.2,
            fill=False,
        )
        axis.add_patch(box)
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
        )

        if index < len(blocks) - 1:
            axis.annotate(
                "",
                xy=(x + 2.0, y + height / 2),
                xytext=(x + width, y + height / 2),
                arrowprops={"arrowstyle": "->", "linewidth": 1.2},
            )

    axis.set_title(
        "Baseline CNN Architecture (399,142 trainable parameters)",
        fontsize=14,
        pad=16,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
