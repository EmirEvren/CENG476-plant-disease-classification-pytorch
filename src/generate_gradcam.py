import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from tqdm import tqdm

from data_setup import (
    BATCH_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    create_dataloaders,
)
from efficientnet_models import (
    create_efficientnet_b0_transfer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "efficientnet_b0_b32_blr1e4_hlr5e4_full_best.pt"
)
RUN_NAME = "efficientnet_b0_gradcam"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "EfficientNet-B0 için doğru ve hatalı "
            "tahminlerden Grad-CAM görselleri üretir."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--num-correct",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--num-errors",
        type=int,
        default=6,
    )
    return parser.parse_args()


def validate_arguments(arguments):
    if not arguments.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {arguments.checkpoint}"
        )
    if arguments.batch_size <= 0:
        raise ValueError(
            "Batch size pozitif olmalıdır."
        )
    if arguments.num_correct <= 0:
        raise ValueError(
            "num-correct pozitif olmalıdır."
        )
    if arguments.num_errors <= 0:
        raise ValueError(
            "num-errors pozitif olmalıdır."
        )


def display_name(class_name):
    return (
        class_name.replace("___", " - ")
        .replace("_", " ")
    )


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        checkpoint.get("model_name")
        != "efficientnet_b0_transfer"
    ):
        raise ValueError(
            "Checkpoint EfficientNet-B0 modeline ait değil."
        )

    class_names = list(checkpoint["class_names"])
    model = create_efficientnet_b0_transfer(
        num_classes=len(class_names),
        dropout_rate=float(
            checkpoint.get("dropout_rate", 0.3)
        ),
        pretrained=False,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    return (
        model.to(device).eval(),
        checkpoint,
        class_names,
    )


def dataset_path(test_dataset, subset_index):
    original_dataset = test_dataset.dataset
    original_index = test_dataset.indices[
        subset_index
    ]
    return str(
        Path(
            original_dataset.samples[
                original_index
            ][0]
        )
    )


def select_examples(
    model,
    test_loader,
    class_names,
    device,
    num_correct,
    num_errors,
):
    use_amp = device.type == "cuda"
    unique_correct = []
    unique_errors = []
    all_correct = []
    all_errors = []
    seen_correct_classes = set()
    seen_error_pairs = set()
    subset_offset = 0

    progress_bar = tqdm(
        test_loader,
        desc="Grad-CAM örnek seçimi",
        unit="batch",
    )

    with torch.inference_mode():
        for images, labels in progress_bar:
            images = images.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)

            probabilities = torch.softmax(
                logits.float(),
                dim=1,
            ).cpu()
            predictions = probabilities.argmax(
                dim=1
            )

            for local_index in range(len(labels)):
                subset_index = (
                    subset_offset + local_index
                )
                true_index = int(
                    labels[local_index]
                )
                predicted_index = int(
                    predictions[local_index]
                )
                record = {
                    "subset_index": subset_index,
                    "image_path": dataset_path(
                        test_loader.dataset,
                        subset_index,
                    ),
                    "true_index": true_index,
                    "true_class": (
                        class_names[true_index]
                    ),
                    "predicted_index": (
                        predicted_index
                    ),
                    "predicted_class": (
                        class_names[predicted_index]
                    ),
                    "confidence": float(
                        probabilities[
                            local_index,
                            predicted_index,
                        ]
                    ),
                    "correct": (
                        true_index == predicted_index
                    ),
                }

                if record["correct"]:
                    all_correct.append(record)
                    if (
                        true_index
                        not in seen_correct_classes
                    ):
                        seen_correct_classes.add(
                            true_index
                        )
                        unique_correct.append(record)
                else:
                    all_errors.append(record)
                    error_pair = (
                        true_index,
                        predicted_index,
                    )
                    if error_pair not in seen_error_pairs:
                        seen_error_pairs.add(error_pair)
                        unique_errors.append(record)

            subset_offset += len(labels)

    selected_correct = unique_correct[
        :num_correct
    ]
    selected_errors = unique_errors[
        :num_errors
    ]

    selected_correct_indices = {
        item["subset_index"]
        for item in selected_correct
    }
    selected_error_indices = {
        item["subset_index"]
        for item in selected_errors
    }

    for record in all_correct:
        if len(selected_correct) >= num_correct:
            break
        if (
            record["subset_index"]
            not in selected_correct_indices
        ):
            selected_correct.append(record)
            selected_correct_indices.add(
                record["subset_index"]
            )

    for record in all_errors:
        if len(selected_errors) >= num_errors:
            break
        if (
            record["subset_index"]
            not in selected_error_indices
        ):
            selected_errors.append(record)
            selected_error_indices.add(
                record["subset_index"]
            )

    if len(selected_correct) < num_correct:
        raise RuntimeError(
            "Yeterli doğru tahmin örneği bulunamadı."
        )
    if len(selected_errors) < num_errors:
        raise RuntimeError(
            "Yeterli hatalı tahmin örneği bulunamadı."
        )

    return selected_correct, selected_errors


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.hook_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        module_inputs,
        output,
    ):
        del module, module_inputs
        self.activations = output
        output.register_hook(
            self._gradient_hook
        )

    def _gradient_hook(self, gradients):
        self.gradients = gradients

    def generate(self, image, target_index):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        target_score = logits[
            0,
            target_index,
        ]
        target_score.backward()

        if (
            self.activations is None
            or self.gradients is None
        ):
            raise RuntimeError(
                "Grad-CAM aktivasyonları alınamadı."
            )

        channel_weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )
        heatmap = (
            channel_weights * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )
        heatmap = torch.relu(heatmap)
        heatmap = functional.interpolate(
            heatmap,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        heatmap = heatmap[0, 0]
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        heatmap = (
            (heatmap - heatmap_min)
            / (heatmap_max - heatmap_min + 1e-8)
        )

        return heatmap.detach().cpu().numpy()

    def close(self):
        self.hook_handle.remove()


def unnormalize_image(image_tensor):
    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=image_tensor.dtype,
    ).view(3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD,
        dtype=image_tensor.dtype,
    ).view(3, 1, 1)
    image = image_tensor.cpu() * std + mean
    return (
        image.clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )


def overlay_heatmap(image, heatmap):
    heatmap_colors = plt.get_cmap("jet")(
        heatmap
    )[..., :3]
    return np.clip(
        0.58 * image + 0.42 * heatmap_colors,
        0,
        1,
    )


def create_correct_figure(
    model,
    gradcam,
    test_dataset,
    records,
    class_names,
    device,
    output_path,
):
    figure, axes = plt.subplots(
        len(records),
        2,
        figsize=(10, 4 * len(records)),
        squeeze=False,
    )

    for row_index, record in enumerate(records):
        image_tensor, _ = test_dataset[
            record["subset_index"]
        ]
        image = unnormalize_image(image_tensor)
        heatmap = gradcam.generate(
            image_tensor.unsqueeze(0).to(device),
            record["predicted_index"],
        )
        overlay = overlay_heatmap(
            image,
            heatmap,
        )
        class_label = display_name(
            class_names[
                record["predicted_index"]
            ]
        )

        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(
            f"Orijinal - {class_label}",
            fontsize=9,
        )
        axes[row_index, 1].imshow(overlay)
        axes[row_index, 1].set_title(
            "Grad-CAM - Doğru tahmin\n"
            f"Güven: {record['confidence']:.3f}",
            fontsize=9,
        )

        for column_index in range(2):
            axes[
                row_index,
                column_index,
            ].axis("off")

    figure.suptitle(
        "EfficientNet-B0 Grad-CAM - Doğru Tahminler",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_error_figure(
    model,
    gradcam,
    test_dataset,
    records,
    class_names,
    device,
    output_path,
):
    figure, axes = plt.subplots(
        len(records),
        3,
        figsize=(15, 4 * len(records)),
        squeeze=False,
    )

    for row_index, record in enumerate(records):
        image_tensor, _ = test_dataset[
            record["subset_index"]
        ]
        image = unnormalize_image(image_tensor)
        predicted_heatmap = gradcam.generate(
            image_tensor.unsqueeze(0).to(device),
            record["predicted_index"],
        )
        true_heatmap = gradcam.generate(
            image_tensor.unsqueeze(0).to(device),
            record["true_index"],
        )
        predicted_overlay = overlay_heatmap(
            image,
            predicted_heatmap,
        )
        true_overlay = overlay_heatmap(
            image,
            true_heatmap,
        )
        true_label = display_name(
            class_names[record["true_index"]]
        )
        predicted_label = display_name(
            class_names[
                record["predicted_index"]
            ]
        )

        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(
            f"Gerçek: {true_label}\n"
            f"Tahmin: {predicted_label}",
            fontsize=8,
        )
        axes[row_index, 1].imshow(
            predicted_overlay
        )
        axes[row_index, 1].set_title(
            "Tahmin edilen sınıf için Grad-CAM\n"
            f"Güven: {record['confidence']:.3f}",
            fontsize=8,
        )
        axes[row_index, 2].imshow(true_overlay)
        axes[row_index, 2].set_title(
            "Gerçek sınıf için Grad-CAM",
            fontsize=8,
        )

        for column_index in range(3):
            axes[
                row_index,
                column_index,
            ].axis("off")

    figure.suptitle(
        "EfficientNet-B0 Grad-CAM - Hatalı Tahminler",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    arguments = parse_arguments()
    arguments.checkpoint = (
        arguments.checkpoint.expanduser()
    )
    validate_arguments(arguments)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, checkpoint, checkpoint_class_names = (
        load_model(
            arguments.checkpoint,
            device,
        )
    )
    (
        _,
        _,
        test_loader,
        dataset_class_names,
    ) = create_dataloaders(
        batch_size=arguments.batch_size,
        num_workers=0,
    )

    if (
        checkpoint_class_names
        != list(dataset_class_names)
    ):
        raise RuntimeError(
            "Checkpoint ve veri seti sınıf "
            "indeksleri eşleşmiyor."
        )

    print("=" * 65)
    print("EFFICIENTNET-B0 GRAD-CAM ANALİZİ")
    print("=" * 65)
    print("Cihaz:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Test örneği:", len(test_loader.dataset))
    print(
        "Hedef katman: model.features[-1]"
    )
    print()

    (
        correct_records,
        error_records,
    ) = select_examples(
        model=model,
        test_loader=test_loader,
        class_names=list(dataset_class_names),
        device=device,
        num_correct=arguments.num_correct,
        num_errors=arguments.num_errors,
    )

    output_directory = (
        PROJECT_ROOT
        / "outputs"
        / "gradcam"
        / RUN_NAME
    )
    figures_directory = (
        PROJECT_ROOT
        / "outputs"
        / "figures"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    correct_figure_path = (
        figures_directory
        / f"{RUN_NAME}_correct_examples.png"
    )
    error_figure_path = (
        figures_directory
        / f"{RUN_NAME}_error_examples.png"
    )
    metadata_path = (
        output_directory
        / "gradcam_samples.csv"
    )
    summary_path = (
        output_directory
        / "gradcam_summary.json"
    )

    gradcam = GradCAM(
        model=model,
        target_layer=model.features[-1],
    )
    try:
        create_correct_figure(
            model=model,
            gradcam=gradcam,
            test_dataset=test_loader.dataset,
            records=correct_records,
            class_names=list(dataset_class_names),
            device=device,
            output_path=correct_figure_path,
        )
        create_error_figure(
            model=model,
            gradcam=gradcam,
            test_dataset=test_loader.dataset,
            records=error_records,
            class_names=list(dataset_class_names),
            device=device,
            output_path=error_figure_path,
        )
    finally:
        gradcam.close()

    metadata_frame = pd.DataFrame(
        correct_records + error_records
    )
    metadata_frame["sample_type"] = (
        ["correct"] * len(correct_records)
        + ["error"] * len(error_records)
    )
    metadata_frame.to_csv(
        metadata_path,
        index=False,
    )

    summary = {
        "run_name": RUN_NAME,
        "model_name": "efficientnet_b0_transfer",
        "checkpoint": str(
            arguments.checkpoint.resolve()
        ),
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "gradcam_target_layer": (
            "model.features[-1]"
        ),
        "correct_examples": len(
            correct_records
        ),
        "error_examples": len(error_records),
        "test_used_for_training": False,
        "device": str(device),
        "output_files": {
            "correct_figure": str(
                correct_figure_path.resolve()
            ),
            "error_figure": str(
                error_figure_path.resolve()
            ),
            "sample_metadata": str(
                metadata_path.resolve()
            ),
        },
    }
    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            summary,
            summary_file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Grad-CAM analizi tamamlandı.")
    print(
        "Doğru tahmin görseli:",
        correct_figure_path,
    )
    print(
        "Hatalı tahmin görseli:",
        error_figure_path,
    )
    print("Örnek bilgileri:", metadata_path)
    print("Özet:", summary_path)


if __name__ == "__main__":
    main()
