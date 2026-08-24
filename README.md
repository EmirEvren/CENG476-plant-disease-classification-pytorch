# Plant Disease Classification with PyTorch

**CENG 476 — Introduction to Deep Learning**  
**Student:** Emir Evren — **ID:** 210444038  
**Task:** 38-class PlantVillage image classification  
**Framework:** PyTorch + CUDA

## Project Focus

This project is presented primarily as a **model-development and training-methodology study**, rather than as a claim of architectural novelty. The central questions are:

1. How does a transparent CNN trained from scratch compare with transfer learning?
2. How do preprocessing, data augmentation, Batch Normalization, dropout, weight decay, learning-rate strategy, scheduling, checkpoint selection and early stopping affect training behavior?
3. Can unusually high results be trusted after explicit leakage and generalization checks?

The final audited benchmark is **84.62%** for the custom CNN, **97.66%** for ResNet18, **99.01%** for EfficientNet-B0 and **99.14%** for the validation-selected 50/50 ensemble.

## Model Development and Training Methodology

| Stage | Method / setting | Why it was used | Evidence / observed effect |
|---|---|---|---|
| Pipeline sanity | 38-image intentional overfit | Verify labels, loss, gradients and optimizer before long runs | 71.05% at step 30, **100% at step 40** |
| Preprocessing | RGB `224×224`, ImageNet normalization | Standardize input and match pretrained-model expectations | Used consistently across all final models |
| Data augmentation | RandomResizedCrop, HFlip, ±15° rotation, ColorJitter | Add moderate training variation and reduce dependence on exact pixels | Preventive regularization; its isolated causal effect was **not** measured in a separate on/off ablation |
| Batch Normalization | After each baseline convolution, before ReLU | Stabilize intermediate activation scales and optimization | Four BN layers in the custom CNN; no isolated BN ablation |
| Dropout | Baseline 0.40, transfer heads 0.30 | Reduce excessive co-adaptation | Pilot Macro-F1: 0.20→0.4990, 0.40→0.4953, 0.60→0.4311; **0.60 was too aggressive** |
| Weight decay | AdamW `1e-4` | Explicit regularization of parameter growth | Used in all final runs; not isolated as a single-variable ablation |
| Baseline LR tuning | `1e-3`, `5e-4`, `3e-4` pilots | Compare convergence behavior | Final baseline LR: **`5e-4`** |
| Transfer LR strategy | Backbone `1e-4`, classifier `5e-4` | Preserve pretrained features while adapting the new head faster | Used for full fine-tuning of ResNet18 and EfficientNet-B0 |
| LR scheduler | ReduceLROnPlateau, factor 0.5, patience 2, min `1e-6` | Reduce step size when validation loss plateaus | Scheduler monitors **validation loss** |
| Checkpoint selection | Validation Macro-F1; validation loss tie-break | Give all 38 classes equal importance during model selection | Test set not used for checkpoint selection |
| Early stopping | Patience 6 baseline, 5 transfer | Safeguard against unnecessary training and overfitting | Implemented; selected final runs reached max epoch before early stopping terminated them |
| Transfer learning | ImageNet ResNet18 + EfficientNet-B0, full fine-tuning | Test the value of pretrained visual representations | 84.62% scratch → 97.66% ResNet18 → **99.01% EfficientNet** |
| Ensemble | Validation-only soft voting | Combine complementary errors | 50/50 selected on validation Macro-F1; final **99.14%**, 92 errors |
| Integrity audit | exact hash, physical-leaf mapping, `dHash≤4` | Investigate the historical 99.76% result | Original split found optimistic; final benchmark recomputed under ultra-strict protocol |

## Data and Evaluation Strategy

The project uses a fixed **train / validation / test** design. **K-fold cross-validation was not used.** Repeated full deep-model fine-tuning would substantially increase compute cost, while a separate validation set and locked test set were already maintained. Reproducibility was additionally checked with three random seeds.

Final ultra-strict split:

| Split | Images | Role |
|---|---:|---|
| Train | **39,091** | Gradient-based optimization + augmentation |
| Validation | **4,462** | Scheduler, checkpoint, hyperparameter and ensemble decisions |
| Locked test | **10,709** | Final evaluation |
| Total | **54,262** | 38 classes |

The locked test set was not used for model training, checkpoint selection, hyperparameter tuning or ensemble-weight selection. Test images were used only for deterministic, model-independent integrity auditing before final evaluation.

## Preprocessing and Data Augmentation

Training transform:

```python
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.80, 1.00)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])
```

Validation/test transform:

```python
Resize(256) → CenterCrop(224) → ToTensor() → ImageNet Normalize
```

Random augmentation is applied only to training data so that validation and test measurements remain deterministic. Augmentation is intentionally moderate because plant-disease classification depends strongly on lesion color and fine texture.

## Models

### Custom CNN — scratch baseline

```text
3×224×224
→ Conv(3→32) + BN + ReLU + MaxPool
→ Conv(32→64) + BN + ReLU + MaxPool
→ Conv(64→128) + BN + ReLU + MaxPool
→ Conv(128→256) + BN + ReLU + MaxPool
→ AdaptiveAvgPool(1×1)
→ Dropout(0.40)
→ Linear(256→38)
→ raw logits
```

**399,142 trainable parameters.** This model provides a transparent from-scratch reference rather than an architectural novelty claim.

### ResNet18

ImageNet pretrained, full end-to-end fine-tuning, `Dropout(0.30) → Linear(512,38)`, **11,196,006 parameters**.

### EfficientNet-B0

ImageNet pretrained, full end-to-end fine-tuning, `Dropout(0.30) → Linear(1280,38)`, **4,056,226 parameters**. It provides the best individual accuracy/parameter trade-off.

## Loss, Optimizer and Training Step

All models use unweighted `nn.CrossEntropyLoss()` with **raw logits**. Softmax is intentionally not applied before the loss because CrossEntropyLoss performs the stable LogSoftmax/NLL computation internally. Softmax is used later only when class probabilities are needed for ensemble voting, confidence, ROC and calibration analysis.

Optimizer: **AdamW**, `betas=(0.9,0.999)`, `weight_decay=1e-4`.

Mini-batch update order:

```text
zero_grad → forward → CrossEntropyLoss → backward → optimizer.step
```

AMP is enabled on CUDA.

## Final Training Configuration

| Setting | Custom CNN | ResNet18 | EfficientNet-B0 |
|---|---:|---:|---:|
| Batch size | 64 | 32 | 32 |
| Maximum epochs | 15 | 12 | 12 |
| Backbone/main LR | `5e-4` | `1e-4` | `1e-4` |
| Classifier LR | — | `5e-4` | `5e-4` |
| Dropout | 0.40 | 0.30 | 0.30 |
| Weight decay | `1e-4` | `1e-4` | `1e-4` |
| Early-stop patience | 6 | 5 | 5 |

`ReduceLROnPlateau` monitors validation loss with `factor=0.5`, `patience=2`, `min_lr=1e-6`. The best checkpoint is selected by **validation Macro-F1**, with validation loss as a tie-breaker.

## Training Dynamics and Overfitting Check

A deterministic clean-train subset of **20 images/class = 760 images** is used to compare training, validation and test behavior without random training augmentation.

| Model | Clean-train accuracy | Validation accuracy | Test accuracy |
|---|---:|---:|---:|
| Custom CNN | 77.89% | 83.55% | 84.62% |
| ResNet18 | 98.16% | 98.81% | 97.66% |
| EfficientNet-B0 | 99.74% | 98.68% | 99.01% |

The final transfer models do not show the large train-to-validation collapse expected from severe conventional memorization. Their larger limitation is cross-domain generalization, discussed below.

Representative training curves are stored in `outputs/figures/`, including:

- `baseline_b64_lr5e4_full_training_curves.png`
- `resnet18_b32_blr1e4_hlr5e4_full_training_curves.png`
- `efficientnet_b0_b32_blr1e4_hlr5e4_full_training_curves.png`
- dropout and LR pilot curves

## Why the Original 99.76% Was Rejected as the Final Benchmark

The historical image-level split produced a **99.76%** ensemble result. Instead of reporting it directly, the split was audited.

Original audit findings:

- exact cross-split duplicate groups/pairs: **10**
- perceptual near-duplicate pairs: **68**
- mapped same-physical-leaf cross-split groups: **4,956**

The final strict review found **39** `dHash≤4` cross-split pairs. With priority `test > validation > train`, **34 train + 4 validation + 0 test** images were quarantined. No detected exact, mapped-leaf or strict `dHash≤4` overlap remains under the implemented audit protocol.

This does not mathematically prove that every unmapped image represents a unique physical specimen; available physical-leaf mapping covers about 75.7% of the images.

## Final Ultra-Strict Results

| Model | Parameters | Accuracy | Macro-F1 | Weighted-F1 | Macro ROC-AUC | Errors |
|---|---:|---:|---:|---:|---:|---:|
| Custom CNN | 399,142 | **84.62%** | 0.7813 | 0.8361 | 0.995589 | 1,647 |
| ResNet18 | 11,196,006 | **97.66%** | 0.9686 | 0.9763 | 0.999929 | 251 |
| EfficientNet-B0 | 4,056,226 | **99.01%** | **0.9874** | 0.9901 | 0.999955 | 106 |
| 50/50 ensemble | 15,252,232 | **99.14%** | **0.9897** | 0.9914 | 0.999976 | **92** |

The 50/50 ensemble weight was selected **only on validation Macro-F1**.

## Reproducibility and Additional Validation

EfficientNet-B0 seed stability on the same fixed ultra-strict manifest:

| Seed | Test accuracy | Macro-F1 |
|---:|---:|---:|
| 42 | 99.010% | 0.987373 |
| 123 | 98.767% | 0.983668 |
| 777 | 99.225% | 0.990149 |

Mean accuracy: **99.001%**, standard deviation: **0.229 percentage points**. The best seed is not cherry-picked as the headline result.

Other checks include random-label sanity control, calibration (ECE/NLL/Brier), 1,000-sample bootstrap confidence intervals, corruption/occlusion stress tests and Grad-CAM qualitative inspection.

## External Generalization: PlantDoc

The final models were evaluated **without retraining** on 236 manually mapped Cropped-PlantDoc test images from 27 mapped source classes.

| Model | Accuracy | Mapped Macro-F1 |
|---|---:|---:|
| EfficientNet-B0 | **23.31%** | 0.2183 |
| 50/50 ensemble | **25.00%** | 0.2349 |

This is an out-of-domain stress test, not a directly comparable benchmark. The result shows **strong PlantVillage-domain performance but weak cross-domain transfer**. It should not be interpreted as proof that individual training images were simply memorized.

## Notebook and Reproduction

The main notebook is:

`notebooks/CENG476_Plant_Disease_Main_Experiments_Emir_Evren.ipynb`

It contains code cells, recorded outputs, methodology explanations, model-development experiments and the final audited evaluation.

Environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Main pipelines:

```bat
call .\run_ultrastrict_all.bat
call .\run_full_control_all.bat
```

## Repository Structure

| Path | Purpose |
|---|---|
| `src/` | preprocessing, models, training, evaluation and audit code |
| `notebooks/` | main experiment notebook |
| `outputs/figures/` | training curves, confusion matrices, Grad-CAM and robustness figures |
| `outputs/audit/` | split-integrity and full-control evidence |
| `report/` | final report and validation summary |
| `run_ultrastrict_all.bat` | ultra-strict final pipeline |
| `run_full_control_all.bat` | extended validation pipeline |

## Final Interpretation

The main contribution of this project is the **controlled development and evaluation process**:

> Moderate augmentation, Batch Normalization, dropout/weight decay regularization, validation-driven learning-rate control, full transfer learning and validation-only model selection produced strong PlantVillage performance. The historical 99.76% result was rejected after leakage auditing; after the stricter protocol the best single model still achieved 99.01% and the ensemble 99.14%. However, the PlantDoc result shows that this same-domain performance does not directly transfer to field-like imagery.
