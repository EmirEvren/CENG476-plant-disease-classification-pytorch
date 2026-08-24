from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs" / "audit" / "full_control"
HIST = ROOT / "outputs" / "histories" / "efficientnet_b0_b32_blr1e4_hlr5e4_full_history.csv"
OUT = ROOT / "outputs" / "figures" / "presentation_metrics"
OUT.mkdir(parents=True, exist_ok=True)

per_class = pd.read_csv(AUDIT / "efficientnet_per_class.csv").sort_values("class_index").reset_index(drop=True)
cm_df = pd.read_csv(AUDIT / "efficientnet_confusion_matrix.csv", index_col=0)
cm = cm_df.to_numpy(dtype=int)
history = pd.read_csv(HIST)

classes = per_class["class_name"].tolist()
ids = [f"C{i:02d}" for i in range(len(classes))]


def pretty_name(name: str) -> str:
    s = name.replace("___", " — ").replace("_", " ")
    s = s.replace("Corn (maize)", "Corn").replace("Cherry (including sour)", "Cherry")
    s = s.replace("Pepper, bell", "Pepper")
    s = s.replace("Haunglongbing (Citrus greening)", "Citrus greening")
    s = s.replace("Spider mites Two-spotted spider mite", "Spider mites")
    return s


pretty = [pretty_name(c) for c in classes]
N = cm.sum()
tp = np.diag(cm).astype(float)
support = cm.sum(axis=1).astype(float)
pred_support = cm.sum(axis=0).astype(float)
fn = support - tp
fp = pred_support - tp
tn = N - tp - fn - fp
precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
specificity = np.divide(tn, tn + fp, out=np.zeros_like(tp), where=(tn + fp) != 0)

# 1. Loss vs validation loss.
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(history["epoch"], history["train_loss"], marker="o", label="Train Loss")
ax.plot(history["epoch"], history["validation_loss"], marker="o", label="Validation Loss")
ax.set_title("EfficientNet-B0 — Training Loss vs Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xticks(history["epoch"])
fig.tight_layout()
fig.savefig(OUT / "01_efficientnet_loss_vs_validation_loss.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 2. Accuracy vs validation accuracy.
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(history["epoch"], history["train_accuracy"] * 100, marker="o", label="Train Accuracy")
ax.plot(history["epoch"], history["validation_accuracy"] * 100, marker="o", label="Validation Accuracy")
ax.set_title("EfficientNet-B0 — Training Accuracy vs Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xticks(history["epoch"])
fig.tight_layout()
fig.savefig(OUT / "02_efficientnet_accuracy_vs_validation_accuracy.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 3. Validation Macro-F1.
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(history["epoch"], history["validation_macro_f1"], marker="o", label="Validation Macro-F1")
best_idx = history["validation_macro_f1"].idxmax()
best_epoch = int(history.loc[best_idx, "epoch"])
best_f1 = float(history.loc[best_idx, "validation_macro_f1"])
ax.scatter([best_epoch], [best_f1], s=80, label=f"Best: epoch {best_epoch}, F1={best_f1:.4f}")
ax.set_title("EfficientNet-B0 — Validation Macro-F1")
ax.set_xlabel("Epoch")
ax.set_ylabel("Macro-F1")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xticks(history["epoch"])
fig.tight_layout()
fig.savefig(OUT / "03_efficientnet_validation_macro_f1.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 4. Full class-wise F1, sensitivity and specificity.
y = np.arange(len(classes))
h = 0.24
fig, ax = plt.subplots(figsize=(14, 18))
ax.barh(y - h, f1, height=h, label="F1-Score")
ax.barh(y, recall, height=h, label="Sensitivity (Recall)")
ax.barh(y + h, specificity, height=h, label="Specificity")
ax.set_yticks(y)
ax.set_yticklabels([f"{ids[i]}  {pretty[i]}" for i in range(len(classes))], fontsize=8)
ax.set_xlim(0.88, 1.005)
ax.set_xlabel("Score")
ax.set_title("EfficientNet-B0 — Class-wise F1, Sensitivity and Specificity\nFinal Ultra-Strict PlantVillage Test (n=10,709)")
ax.grid(True, axis="x", alpha=0.3)
ax.legend(loc="lower right")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT / "04_efficientnet_classwise_f1_sensitivity_specificity.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 5. Raw confusion matrix.
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cm)
ax.set_title("EfficientNet-B0 — Confusion Matrix (Raw Counts)\nFinal Ultra-Strict PlantVillage Test")
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(ids, rotation=90, fontsize=7)
ax.set_yticklabels(ids, fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUT / "05_efficientnet_confusion_matrix_raw.png", dpi=240, bbox_inches="tight")
plt.close(fig)

# 6. Row-normalized confusion matrix.
cm_norm = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cm_norm)
ax.set_title("EfficientNet-B0 — Row-Normalized Confusion Matrix")
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(ids, rotation=90, fontsize=7)
ax.set_yticklabels(ids, fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUT / "06_efficientnet_confusion_matrix_normalized.png", dpi=240, bbox_inches="tight")
plt.close(fig)

# 7. Class-ID legend for use next to confusion matrices.
fig, ax = plt.subplots(figsize=(14, 12))
ax.axis("off")
legend_rows = [[ids[i], pretty[i], int(support[i])] for i in range(len(classes))]
table = ax.table(cellText=legend_rows, colLabels=["ID", "Class", "Support"], loc="center", cellLoc="left", colLoc="left", colWidths=[0.08, 0.75, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.35)
ax.set_title("PlantVillage Final Test — Class ID Legend", pad=20)
fig.tight_layout()
fig.savefig(OUT / "07_class_id_legend.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 8-9. Detailed classification report in two readable halves.
def draw_report(rows, filename, title):
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.axis("off")
    cells = []
    for i in rows:
        cells.append([ids[i], pretty[i], f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}", f"{specificity[i]:.4f}", f"{int(support[i])}"])
    table = ax.table(cellText=cells, colLabels=["ID", "Class", "Precision", "Recall", "F1", "Specificity", "Support"], loc="center", cellLoc="left", colLoc="left", colWidths=[0.06, 0.46, 0.10, 0.10, 0.10, 0.10, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)
    ax.set_title(title, pad=18)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


draw_report(range(0, 19), "08_classification_report_classes_00_18.png", "EfficientNet-B0 — Classification Report (Classes C00–C18)")
draw_report(range(19, 38), "09_classification_report_classes_19_37.png", "EfficientNet-B0 — Classification Report (Classes C19–C37)")

# 10. Overall summary.
accuracy = tp.sum() / N
macro_precision = precision.mean()
macro_recall = recall.mean()
macro_f1 = f1.mean()
weighted_precision = np.average(precision, weights=support)
weighted_recall = np.average(recall, weights=support)
weighted_f1 = np.average(f1, weights=support)
summary = [
    ["Accuracy", "", "", "", f"{accuracy:.4f}", str(N)],
    ["Macro avg", f"{macro_precision:.4f}", f"{macro_recall:.4f}", f"{macro_f1:.4f}", "", str(N)],
    ["Weighted avg", f"{weighted_precision:.4f}", f"{weighted_recall:.4f}", f"{weighted_f1:.4f}", "", str(N)],
]
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.axis("off")
table = ax.table(cellText=summary, colLabels=["Summary", "Precision", "Recall", "F1-Score", "Accuracy", "Support"], loc="center", cellLoc="center", colLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.0)
ax.set_title("EfficientNet-B0 — Final Classification Summary", pad=18)
fig.tight_layout()
fig.savefig(OUT / "10_efficientnet_classification_summary.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 11. Lowest class-wise F1 scores: useful for discussion instead of showing only near-perfect classes.
worst_idx = np.argsort(f1)[:10]
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.barh(np.arange(len(worst_idx)), f1[worst_idx])
ax.set_yticks(np.arange(len(worst_idx)))
ax.set_yticklabels([f"{ids[i]}  {pretty[i]}" for i in worst_idx], fontsize=9)
ax.set_xlim(0.90, 1.0)
ax.set_xlabel("F1-Score")
ax.set_title("EfficientNet-B0 — 10 Lowest Class-wise F1 Scores")
ax.grid(True, axis="x", alpha=0.3)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT / "11_efficientnet_lowest_class_f1.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Save a reusable CSV including specificity derived from the final confusion matrix.
report_df = pd.DataFrame({
    "class_id": ids,
    "class_name": classes,
    "precision": precision,
    "sensitivity_recall": recall,
    "specificity": specificity,
    "f1_score": f1,
    "support": support.astype(int),
    "errors": fn.astype(int),
})
report_df.to_csv(OUT / "efficientnet_final_classification_metrics.csv", index=False)

print(f"Generated presentation metrics in: {OUT}")
print(f"Final EfficientNet accuracy: {accuracy * 100:.6f}%")
print(f"Macro-F1: {macro_f1:.6f}")
print(f"Errors: {int(N - tp.sum())}")
