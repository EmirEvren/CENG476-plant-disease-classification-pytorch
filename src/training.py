from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm


def _calculate_macro_f1(
    all_targets,
    all_predictions,
):
    from sklearn.metrics import f1_score

    return f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )


def run_epoch(
    model,
    data_loader,
    criterion,
    device,
    use_amp,
    phase,
    optimizer=None,
    scaler=None,
):
    is_training = optimizer is not None
    model.train(is_training)

    if is_training and scaler is None:
        raise ValueError(
            "Eğitim aşamasında scaler belirtilmelidir."
        )

    running_loss = 0.0
    correct_predictions = 0
    processed_samples = 0
    all_targets = []
    all_predictions = []

    progress_bar = tqdm(
        data_loader,
        desc=phase,
        unit="batch",
    )

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        predictions = logits.argmax(dim=1)
        current_batch_size = labels.size(0)

        running_loss += loss.item() * current_batch_size
        correct_predictions += (
            predictions == labels
        ).sum().item()
        processed_samples += current_batch_size

        all_targets.extend(
            labels.detach().cpu().tolist()
        )
        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
        )

    if processed_samples == 0:
        raise RuntimeError(
            f"{phase} DataLoader hiç örnek üretmedi."
        )

    return {
        "loss": running_loss / processed_samples,
        "accuracy": (
            correct_predictions / processed_samples
        ),
        "macro_f1": _calculate_macro_f1(
            all_targets,
            all_predictions,
        ),
    }


def _print_metrics(label, metrics):
    print(
        f"{label:<16} | "
        f"Loss: {metrics['loss']:.4f} | "
        f"Accuracy: {metrics['accuracy'] * 100:.2f}% | "
        f"Macro-F1: {metrics['macro_f1']:.4f}"
    )


def train_model(
    model,
    train_loader,
    train_evaluation_loader,
    validation_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    epochs,
    patience,
    checkpoint_path,
    history_path,
    checkpoint_metadata=None,
    minimum_improvement=1e-4,
):
    checkpoint_path = Path(checkpoint_path)
    history_path = Path(history_path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    use_amp = device.type == "cuda"
    best_validation_macro_f1 = float("-inf")
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        print()
        print("=" * 65)
        print(f"Epoch {epoch}/{epochs}")
        print("=" * 65)

        learning_rate = optimizer.param_groups[0]["lr"]

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        start_time = perf_counter()

        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            phase="Train",
            optimizer=optimizer,
            scaler=scaler,
        )
        train_evaluation_metrics = run_epoch(
            model=model,
            data_loader=train_evaluation_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            phase="Clean Train Eval",
        )
        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            phase="Validation",
        )

        elapsed_seconds = perf_counter() - start_time

        if device.type == "cuda":
            peak_memory_mb = (
                torch.cuda.max_memory_allocated()
                / 1024**2
            )
        else:
            peak_memory_mb = 0.0

        scheduler.step(validation_metrics["loss"])
        next_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        epoch_record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_eval_loss": (
                train_evaluation_metrics["loss"]
            ),
            "train_eval_accuracy": (
                train_evaluation_metrics["accuracy"]
            ),
            "train_eval_macro_f1": (
                train_evaluation_metrics["macro_f1"]
            ),
            "validation_loss": (
                validation_metrics["loss"]
            ),
            "validation_accuracy": (
                validation_metrics["accuracy"]
            ),
            "validation_macro_f1": (
                validation_metrics["macro_f1"]
            ),
            "elapsed_seconds": elapsed_seconds,
            "peak_gpu_memory_mb": peak_memory_mb,
        }
        history.append(epoch_record)

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        _print_metrics(
            "Train (aug.)",
            train_metrics,
        )
        _print_metrics(
            "Clean Train Eval",
            train_evaluation_metrics,
        )
        _print_metrics(
            "Validation",
            validation_metrics,
        )
        print(
            f"Süre: {elapsed_seconds:.1f} saniye | "
            f"Tepe GPU belleği: "
            f"{peak_memory_mb:.1f} MB | "
            f"LR: {learning_rate:.2e}"
        )

        if next_learning_rate != learning_rate:
            print(
                "Scheduler öğrenme oranını "
                f"{next_learning_rate:.2e} "
                "değerine düşürdü."
            )

        current_macro_f1 = (
            validation_metrics["macro_f1"]
        )
        current_validation_loss = (
            validation_metrics["loss"]
        )
        f1_improved = (
            current_macro_f1
            > best_validation_macro_f1
            + minimum_improvement
        )
        f1_tied = (
            abs(
                current_macro_f1
                - best_validation_macro_f1
            )
            <= minimum_improvement
        )
        loss_improved_on_tie = (
            f1_tied
            and current_validation_loss
            < best_validation_loss
        )
        is_best = (
            f1_improved or loss_improved_on_tie
        )

        if is_best:
            best_validation_macro_f1 = (
                current_macro_f1
            )
            best_validation_loss = (
                current_validation_loss
            )
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "scheduler_state_dict": (
                    scheduler.state_dict()
                ),
                "scaler_state_dict": (
                    scaler.state_dict()
                ),
                "validation_loss": (
                    validation_metrics["loss"]
                ),
                "validation_accuracy": (
                    validation_metrics["accuracy"]
                ),
                "validation_macro_f1": (
                    validation_metrics["macro_f1"]
                ),
                "selection_metric": (
                    "validation_macro_f1"
                ),
            }

            if checkpoint_metadata:
                checkpoint.update(
                    checkpoint_metadata
                )

            torch.save(checkpoint, checkpoint_path)
            print(
                "Yeni en iyi model kaydedildi "
                "(Validation Macro-F1):",
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            print(
                "Validation Macro-F1 iyileşmedi. "
                "Early stopping sayacı: "
                f"{epochs_without_improvement}/"
                f"{patience}"
            )

        if epochs_without_improvement >= patience:
            print(
                "Early stopping eğitimi durdurdu."
            )
            break

    return history


def _plot_available_series(
    axis,
    history_frame,
    series,
):
    for column_name, label, multiplier in series:
        if column_name not in history_frame:
            continue

        values = history_frame[column_name]

        if not values.notna().any():
            continue

        axis.plot(
            history_frame["epoch"],
            values * multiplier,
            marker="o",
            label=label,
        )


def plot_training_history(
    history,
    output_path,
    title,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_frame = pd.DataFrame(history)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    _plot_available_series(
        axes[0],
        history_frame,
        [
            ("train_loss", "Train (aug.)", 1),
            (
                "train_eval_loss",
                "Clean Train Eval",
                1,
            ),
            (
                "validation_loss",
                "Validation",
                1,
            ),
        ],
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    _plot_available_series(
        axes[1],
        history_frame,
        [
            (
                "train_accuracy",
                "Train (aug.)",
                100,
            ),
            (
                "train_eval_accuracy",
                "Clean Train Eval",
                100,
            ),
            (
                "validation_accuracy",
                "Validation",
                100,
            ),
        ],
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("%")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    _plot_available_series(
        axes[2],
        history_frame,
        [
            (
                "train_macro_f1",
                "Train (aug.)",
                1,
            ),
            (
                "train_eval_macro_f1",
                "Clean Train Eval",
                1,
            ),
            (
                "validation_macro_f1",
                "Validation",
                1,
            ),
        ],
    )
    axes[2].set_title("Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)
