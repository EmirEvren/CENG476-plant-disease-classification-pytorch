# Plant Disease Classification with PyTorch

**CENG 476 — Introduction to Deep Learning**  
**Student:** Emir Evren — **ID:** 210444038  
**Task:** 38-class PlantVillage image classification  
**Framework:** PyTorch + CUDA

## Project Focus

This project is presented primarily as a **model-development and training-methodology study**, not as a claim of architectural novelty. In line with the presentation guidance, the emphasis is on what was applied during training, why it was applied, and what was observed:

- preprocessing and deterministic evaluation transforms,
- training-only data augmentation,
- Batch Normalization and activation functions,
- dropout and AdamW weight-decay regularization,
- CrossEntropyLoss with raw logits,
- learning-rate tuning and differential transfer-learning rates,
- ReduceLROnPlateau scheduling,
- validation Macro-F1 checkpoint selection,
- early-stopping safeguards,
- train/validation dynamics and overfitting analysis,
- transfer-learning development,
- leakage auditing and external generalization checks.

**Final audited benchmark:** Custom CNN **84.62%**, ResNet18 **97.66%**, EfficientNet-B0 **99.01%**, validation-selected 50/50 ensemble **99.14%**.

## Methodology at a Glance

| Stage | Method / value | Why it was used | Observed evidence |
|---|---|---|---|
| Input | RGB `3×224×224` | consistent input size | used by all final models |
| Training augmentation | crop, HFlip, ±15° rotation, ColorJitter 0.2 | controlled variation; reduce exact-pixel dependence | retained in final pipeline; no isolated on/off causal gain claimed |
| Evaluation preprocessing | Resize 256 → CenterCrop 224 → ImageNet Normalize | deterministic comparison and pretrained-model compatibility | same evaluation transform for validation/test |
| Batch Normalization | after each custom-CNN convolution | stabilize activation scales / optimization | 4 BN layers; no isolated BN ablation |
| Dropout | 0.40 baseline; 0.30 transfer heads | reduce excessive co-adaptation | pilot: 0.20→0.4990, 0.40→0.4953, 0.60→0.4311 Val Macro-F1 |
| Weight decay | AdamW `1e-4` | explicit regularization | used in all final runs; no isolated WD ablation |
| Loss | unweighted CrossEntropyLoss | 38-class single-label classification | raw logits; Softmax not placed before loss |
| Baseline LR | `1e-3`, `5e-4`, `3e-4` pilots | avoid arbitrary LR choice | final baseline `5e-4` |
| Transfer LR | backbone `1e-4`, classifier `5e-4` | preserve pretrained features while adapting new head faster | full fine-tuning for ResNet18/EfficientNet-B0 |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 2, min `1e-6` | reduce LR after validation-loss plateau | scheduler monitors validation loss |
| Checkpoint | validation Macro-F1, val-loss tie-break | balanced 38-class model selection | test set not used for checkpoint selection |
| Early stopping | patience 6 baseline / 5 transfer | safeguard against unnecessary over-training | implemented; selected final runs reached max epoch first |
| Transfer learning | ResNet18 + EfficientNet-B0 | evaluate pretrained visual representations | 84.62% scratch → 97.66% ResNet18 → 99.01% EfficientNet |
| Ensemble | validation-only 50/50 soft voting | combine model probabilities | final 99.14%, 92 errors |

## Data and Evaluation Strategy

A fixed **train / validation / locked-test** design is used. **K-fold cross-validation was not used.** Repeated full deep-network fine-tuning across folds would substantially increase compute cost, while a dedicated validation set and locked test were already maintained. Reproducibility was separately examined across three random seeds.

Final ultra-strict split:

| Split | Images | Role |
|---|---:|---|
| Train | **39,091** | gradient-based optimization + augmentation |
| Validation | **4,462** | scheduler, checkpoint, hyperparameter and ensemble decisions |
| Locked test | **10,709** | final evaluation |
| Total | **54,262** | 38 classes |

The locked test set was not used for model training, checkpoint selection, hyperparameter tuning or ensemble-weight selection. Test images were included only in deterministic, model-independent duplicate/near-duplicate integrity auditing.

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

Validation/test:

```text
Resize(256) → CenterCrop(224) → ToTensor() → ImageNet Normalize
```

Random transforms are training-only. The augmentation is intentionally moderate because plant-disease classification depends on lesion color and fine texture. An augmentation-on/off single-variable ablation was not performed, so the repository does not claim an isolated percentage improvement caused only by augmentation.

## Models

### Custom CNN — scratch baseline

```text
3×224×224
→ Conv(3→32) + BatchNorm + ReLU + MaxPool
→ Conv(32→64) + BatchNorm + ReLU + MaxPool
→ Conv(64→128) + BatchNorm + ReLU + MaxPool
→ Conv(128→256) + BatchNorm + ReLU + MaxPool
→ AdaptiveAvgPool(1×1)
→ Dropout(0.40)
→ Linear(256→38)
→ raw logits
```

**399,142 trainable parameters.** It provides a transparent from-scratch reference.

### ResNet18

ImageNet pretrained, full end-to-end fine-tuning, `Dropout(0.30) → Linear(512,38)`, **11,196,006 parameters**.

### EfficientNet-B0

ImageNet pretrained, full end-to-end fine-tuning, `Dropout(0.30) → Linear(1280,38)`, **4,056,226 parameters**. It provides the best individual accuracy/parameter trade-off.

## Loss and Optimization

All models use **unweighted `nn.CrossEntropyLoss()` with raw logits**. Softmax is intentionally not applied before the loss because CrossEntropyLoss performs the stable LogSoftmax/NLL computation internally. Softmax is used later when probabilities are needed for ensemble voting, confidence, ROC and calibration analysis.

Optimizer: **AdamW**, `betas=(0.9,0.999)`, `weight_decay=1e-4`.

Mini-batch order:

```text
zero_grad → forward → CrossEntropyLoss → backward → optimizer.step
```

AMP is enabled on CUDA.

## Final Training Configuration

| Setting | Custom CNN | ResNet18 | EfficientNet-B0 |
|---|---:|---:|---:|
| Batch size | 64 | 32 | 32 |
| Max epochs | 15 | 12 | 12 |
| Main/backbone LR | `5e-4` | `1e-4` | `1e-4` |
| Classifier LR | — | `5e-4` | `5e-4` |
| Dropout | 0.40 | 0.30 | 0.30 |
| Weight decay | `1e-4` | `1e-4` | `1e-4` |
| Early-stop patience | 6 | 5 | 5 |

`ReduceLROnPlateau` monitors validation loss with `factor=0.5`, `patience=2`, `min_lr=1e-6`. The best checkpoint is selected by **validation Macro-F1**, with validation loss as a tie-breaker.

## Training Dynamics and Overfitting Check

A deterministic clean-train subset of **20 images/class = 760 images** is used for train-vs-validation analysis without random augmentation.

| Model | Clean-train accuracy | Validation accuracy | Test accuracy |
|---|---:|---:|---:|
| Custom CNN | 77.89% | 83.55% | 84.62% |
| ResNet18 | 98.16% | 98.81% | 97.66% |
| EfficientNet-B0 | 99.74% | 98.68% | 99.01% |

Representative curves:

- `outputs/figures/baseline_b64_lr5e4_full_training_curves.png`
- `outputs/figures/resnet18_b32_blr1e4_hlr5e4_full_training_curves.png`
- `outputs/figures/efficientnet_b0_b32_blr1e4_hlr5e4_full_training_curves.png`
- dropout and LR pilot curves in `outputs/figures/`

The transfer models do not show the large same-domain train-to-validation collapse expected from severe conventional memorization. Their larger limitation is cross-domain generalization.

## Final Ultra-Strict Results

| Model | Parameters | Accuracy | Macro-F1 | Weighted-F1 | Macro ROC-AUC | Errors |
|---|---:|---:|---:|---:|---:|---:|
| Custom CNN | 399,142 | **84.62%** | 0.7813 | 0.8361 | 0.995589 | 1,647 |
| ResNet18 | 11,196,006 | **97.66%** | 0.9686 | 0.9763 | 0.999929 | 251 |
| EfficientNet-B0 | 4,056,226 | **99.01%** | **0.9874** | 0.9901 | 0.999955 | 106 |
| 50/50 ensemble | 15,252,232 | **99.14%** | **0.9897** | 0.9914 | 0.999976 | **92** |

## Why the Historical 99.76% Was Rejected

The original image-level split contained documented overlap:

- exact cross-split duplicates: **10**
- perceptual near-duplicate pairs: **68**
- mapped same-physical-leaf cross-split groups: **4,956**

The final strict review found **39** `dHash≤4` cross-split pairs. With priority `test > validation > train`, **34 train + 4 validation + 0 test** images were quarantined. No detected exact, mapped-leaf or strict `dHash≤4` overlap remains under the implemented audit protocol.

This does not mathematically prove unique physical specimens for every unmapped image; physical-leaf mapping covers about 75.7% of images.

## Reproducibility and External Generalization

EfficientNet seed stability:

| Seed | Test accuracy | Macro-F1 |
|---:|---:|---:|
| 42 | 99.010% | 0.987373 |
| 123 | 98.767% | 0.983668 |
| 777 | 99.225% | 0.990149 |

Mean accuracy: **99.001%**, standard deviation: **0.229 percentage points**. The best seed is not cherry-picked as the headline result.

External PlantDoc zero-shot evaluation:

| Model | Accuracy | Mapped Macro-F1 |
|---|---:|---:|
| EfficientNet-B0 | **23.31%** | 0.2183 |
| 50/50 ensemble | **25.00%** | 0.2349 |

This supports the final interpretation: **strong within-PlantVillage generalization, weak cross-domain transfer**. The near-99% same-domain result must not be presented as near-99% real-world field accuracy.

## Notebook

The main notebook is:

`notebooks/CENG476_Plant_Disease_Main_Experiments_Emir_Evren.ipynb`

It is presentation-aligned and contains implementation cells, recorded outputs, training-methodology explanations, parameter values, experimental comparisons and final validation results.

## Reproduction

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bat
call .\run_ultrastrict_all.bat
call .\run_full_control_all.bat
```

## Repository Structure

| Path | Purpose |
|---|---|
| `src/` | preprocessing, models, training, evaluation and audit code |
| `notebooks/` | main experiment notebook |
| `outputs/figures/` | training curves, confusion matrices, ROC/analysis figures |
| `outputs/audit/` | split-integrity and full-control evidence |
| `report/` | final report and validation summary |
| `run_ultrastrict_all.bat` | final ultra-strict model pipeline |
| `run_full_control_all.bat` | extended validation pipeline |

## Final Interpretation

The main contribution is the **controlled development process**: preprocessing and training-only augmentation, BatchNorm, dropout/weight decay regularization, validation-driven LR control, full transfer learning, validation-only model selection and explicit integrity checks. The historical 99.76% result was not accepted without auditing; under the final stricter protocol EfficientNet-B0 still achieved **99.01%** and the ensemble **99.14%**, while PlantDoc exposed the main limitation: domain dependence.
