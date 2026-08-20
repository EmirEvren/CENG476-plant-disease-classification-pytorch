# Plant Disease Classification with PyTorch

**CENG 476 - Introduction to Deep Learning**  
**Student:** Emir Evren  
**Student ID:** 210444038  
**Task:** 38-class PlantVillage image classification  
**Framework:** PyTorch 2.11.0 + CUDA

## Project Summary

This project compares a custom CNN trained from scratch with ImageNet-pretrained ResNet18 and EfficientNet-B0 models for 38-class plant-disease classification.

A major part of the final project is the **validation of the unusually high accuracy values**. The first image-level split produced a 99.76% ensemble result, but a later audit found cross-split duplicate / same-leaf contamination. That original result is therefore treated as an **optimistic historical result, not the final benchmark**.

The final benchmark uses an **ultra-strict leakage-controlled protocol** based on the PlantVillage official leaf-preserving split, exact-duplicate checks, mapped physical-leaf checks, perceptual near-duplicate screening, and conservative quarantine of suspicious cross-split pairs.

### Final headline results

- **Best fixed single-model run:** EfficientNet-B0 — **99.01% accuracy**, **0.9874 Macro-F1**
- **Validation-selected 50/50 ensemble:** ResNet18 + EfficientNet-B0 — **99.14% accuracy**, **0.9897 Macro-F1**
- **EfficientNet 3-seed stability:** **99.00% mean accuracy**, **0.229 pp standard deviation**
- **External PlantDoc OOD probe:** EfficientNet-B0 **23.31%**, ensemble **25.00%**

The final conclusion is therefore not that the system achieves 99% accuracy in the real world. The evidence supports a narrower statement:

> Near-99% performance is reproducible inside the controlled PlantVillage domain after the audited leakage mechanisms are removed, but the large PlantDoc performance drop shows strong domain dependence and limited real-world generalization.

## Why the Original 99.76% Result Was Re-audited

The first dataset protocol used an image-level split and produced:

| Model | Original test accuracy |
|---|---:|
| Baseline CNN | 87.42% |
| ResNet18 | 99.26% |
| EfficientNet-B0 | 99.52% |
| Ensemble | **99.76%** |

Because these values were suspiciously high, the split was audited. The original protocol contained:

- exact cross-split duplicate images,
- perceptual near-duplicate pairs,
- and, most importantly, mapped cases where photographs of the **same physical leaf** appeared across different splits.

For that reason, the original 99.76% result is **not used as the final reported result**.

## Final Ultra-Strict Split

The final protocol starts from the PlantVillage official leaf-aware split and applies additional integrity checks.

| Split | Images | Purpose |
|---|---:|---|
| Train | **39,091** | Optimization and augmentation |
| Validation | **4,462** | Scheduler, checkpoint selection and ensemble-weight selection |
| Locked official test | **10,709** | Final evaluation and post-hoc auditing |
| Total used | **54,262** | After conservative quarantine |

### Integrity controls

- exact SHA-256 train/test collisions removed from the training side,
- mapped physical-leaf overlap across train/test: **0**,
- mapped physical-leaf overlap across train/validation: **0**,
- strict perceptual near-duplicate threshold: `dHash <= 4`,
- strict cross-split dHash pairs before quarantine: **39**,
- quarantined images: **34 train + 4 validation + 0 test**,
- strict cross-split dHash pairs after quarantine: **0**.

The official test set itself remained unchanged.

### Important methodology boundary

The locked test set was **not used for model training, checkpoint selection, hyperparameter tuning, or ensemble-weight selection**. Test images were included in deterministic, model-independent duplicate / near-duplicate integrity checks and later post-hoc robustness analysis. Test labels and model predictions were not used to decide which training or validation images to quarantine.

The available physical-leaf mapping covers only part of PlantVillage, so the project does **not** claim mathematical proof that every unmapped image belongs to a unique physical leaf. The correct claim is that no exact, mapped-leaf, or strict `dHash <= 4` cross-split overlap remained under the implemented audit protocol.

## Models

### 1. Custom Baseline CNN

```text
Input 3 x 224 x 224
 -> Conv(3->32) + BatchNorm + ReLU + MaxPool
 -> Conv(32->64) + BatchNorm + ReLU + MaxPool
 -> Conv(64->128) + BatchNorm + ReLU + MaxPool
 -> Conv(128->256) + BatchNorm + ReLU + MaxPool
 -> AdaptiveAvgPool(1 x 1)
 -> Dropout
 -> Linear(256 -> 38)
 -> raw logits
```

- trained from scratch,
- 399,142 parameters,
- final dropout: 0.40.

### 2. ResNet18 Transfer Learning

- ImageNet pretrained,
- full end-to-end fine-tuning,
- `Dropout(0.30) + Linear(512, 38)`,
- 11,196,006 parameters.

### 3. EfficientNet-B0 Transfer Learning

- ImageNet pretrained,
- full end-to-end fine-tuning,
- `Dropout(0.30) + Linear(1280, 38)`,
- 4,056,226 parameters.

## Preprocessing and Regularization

Training augmentation:

- `RandomResizedCrop(224, scale=(0.80, 1.00))`
- horizontal flip, `p=0.50`
- rotation, `+/-15 degrees`
- brightness / contrast / saturation jitter, `0.20`
- ImageNet normalization

Validation and test preprocessing is deterministic:

- resize to 256,
- center crop to 224,
- tensor conversion,
- ImageNet normalization.

Regularization / optimization controls include Batch Normalization, dropout, AdamW weight decay, data augmentation, `ReduceLROnPlateau`, validation-only checkpoint selection, and early-stopping logic.

## Training Configuration

All models use AdamW with `betas=(0.9, 0.999)` and unweighted `CrossEntropyLoss`.

| Setting | Baseline CNN | ResNet18 | EfficientNet-B0 |
|---|---:|---:|---:|
| Batch size | 64 | 32 | 32 |
| Maximum epochs | 15 | 12 | 12 |
| Backbone LR | `5e-4` | `1e-4` | `1e-4` |
| Classifier LR | `5e-4` | `5e-4` | `5e-4` |
| Dropout | 0.40 | 0.30 | 0.30 |
| Weight decay | `1e-4` | `1e-4` | `1e-4` |

`ReduceLROnPlateau` monitors validation loss with `factor=0.5`, `patience=2`, and `min_lr=1e-6`.

Best checkpoints are selected by **validation Macro-F1**, with validation loss used as a tie-breaker.

## Final Ultra-Strict Results

| Model | Parameters | Test accuracy | Macro-F1 | Weighted-F1 | Macro ROC-AUC | Errors |
|---|---:|---:|---:|---:|---:|---:|
| Baseline CNN | 399,142 | 84.62% | 0.7813 | 0.8361 | 0.995589 | 1,647 |
| ResNet18 | 11,196,006 | 97.66% | 0.9686 | 0.9763 | 0.999929 | 251 |
| EfficientNet-B0 | 4,056,226 | **99.01%** | **0.9874** | **0.9901** | **0.999955** | **106** |
| ResNet18 + EfficientNet-B0 | 15,252,232 | **99.14%** | **0.9897** | **0.9914** | **0.999976** | **92** |

The ensemble weight was chosen **only from validation performance**. A 50/50 ResNet18 / EfficientNet-B0 soft-voting mixture was selected before final test evaluation.

## Full-Control Validation

After the ultra-strict retraining, a second validation layer was run specifically to investigate whether near-99% accuracy could still be explained by overfitting, unstable training, calibration failure, pipeline leakage, or shortcut behavior.

### Calibration and uncertainty

| Model | Accuracy | ECE (15 bins) | NLL | Multiclass Brier |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 99.01% | **0.003845** | 0.035704 | 0.016145 |
| 50/50 Ensemble | 99.14% | **0.009121** | 0.035419 | 0.017435 |

Bootstrap 95% accuracy confidence intervals (`n=1000` resamples):

- EfficientNet-B0: **98.81% - 99.20%**
- Ensemble: **98.95% - 99.31%**

### Generalization-gap check

At the selected checkpoints:

| Model | Clean-train accuracy | Validation accuracy | Test accuracy |
|---|---:|---:|---:|
| Baseline CNN | 77.89% | 83.55% | 84.62% |
| ResNet18 | 98.16% | 98.81% | 97.66% |
| EfficientNet-B0 | 99.74% | 98.68% | 99.01% |

The transfer-learning models do not show the large train/validation collapse expected from severe conventional overfitting. The scratch baseline is substantially weaker and shows a different generalization profile.

### Random-label sanity control

Training labels were shuffled on a balanced subset while validation retained the true labels.

- chance accuracy for 38 classes: **2.63%**
- true-label validation accuracy after random-label training: **1.84%**
- validation Macro-F1: **0.0183**
- result: **PASS**

This is a pipeline sanity check, not a proof of zero leakage, but it provides evidence against an accidental direct label/path shortcut in the tested pipeline.

### Robustness stress

Selected post-hoc stress results:

| Condition | EfficientNet | Ensemble |
|---|---:|---:|
| Clean | 99.01% | 99.14% |
| Brightness x0.60 | 98.83% | 99.18% |
| Brightness x1.40 | 98.07% | 98.73% |
| Contrast x0.60 | 98.49% | 98.79% |
| JPEG quality 30 | 98.13% | 98.72% |
| Gaussian blur radius 2 | **84.08%** | **86.53%** |
| Rotation 15 degrees | 99.41% | 99.51% |

Large occlusion tests also cause major degradation. These results show that the models depend strongly on visible fine-grained leaf information and are not uniformly robust to image corruption.

Grad-CAM examples are produced as supporting qualitative evidence. Grad-CAM is not treated as proof of causal feature use.

## Seed Stability

EfficientNet-B0 was retrained on the **same fixed ultra-strict manifest** with three random seeds. The test results are reported together; the highest seed is not selected as the final benchmark.

| Seed | Selected epoch | Validation Macro-F1 | Test accuracy | Test Macro-F1 | Errors |
|---:|---:|---:|---:|---:|---:|
| 42 | 5 | 0.982431 | 99.010% | 0.987373 | 106 |
| 123 | 12 | 0.986704 | 98.767% | 0.983668 | 132 |
| 777 | 12 | 0.992056 | 99.225% | 0.990149 | 83 |

Summary:

- mean accuracy: **99.001%**
- standard deviation: **0.229 percentage points**
- range: **0.458 percentage points**

This supports the conclusion that the near-99% PlantVillage result is reproducible and not dependent on one unusually favorable random seed.

## External Out-of-Domain Test: PlantDoc

To test domain dependence, the final models were evaluated without retraining on a mapped subset of the **PlantDoc test split**.

- mapped PlantDoc test images: **236**
- mapped source classes: **27**
- EfficientNet-B0 accuracy: **23.31%**
- EfficientNet-B0 mapped Macro-F1: **0.2183**
- ensemble accuracy: **25.00%**
- ensemble mapped Macro-F1: **0.2349**

This is an **OOD stress test**, not a directly comparable replacement benchmark: PlantDoc and PlantVillage have different acquisition conditions, distributions, and not perfectly identical class definitions. Cross-dataset labels are manually mapped where a reasonable semantic match exists.

The large drop is nevertheless important. It demonstrates that excellent controlled-domain PlantVillage performance should **not** be interpreted as equivalent real-field performance.

## Final Interpretation

The final project distinguishes three different issues:

1. **Data leakage in the original split:** detected; original 99.76% result invalidated as the final benchmark.
2. **Severe conventional train overfitting in the final transfer models:** not strongly supported by the validation/test gaps, calibration, or repeated-seed results.
3. **Domain dependence / limited real-world generalization:** strongly supported by the PlantDoc OOD performance drop.

Therefore the final scientific claim is intentionally limited to the controlled PlantVillage domain.

## Reproducibility

Create the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The complete final validation pipeline can be resumed with:

```bat
call .\run_full_control_all.bat
```

This runs / resumes:

1. calibration, bootstrap CI, per-class and error audit,
2. train-validation-test gap analysis,
3. corruption and occlusion robustness checks,
4. random-label sanity control,
5. Grad-CAM visual audit,
6. Windows-safe PlantDoc OOD download and evaluation,
7. consolidated full-control report,
8. EfficientNet seed stability for seeds 42, 123 and 777.

The ultra-strict training / evaluation workflow is also available through `run_ultrastrict_all.bat`.

## Repository Structure

| Path | Description |
|---|---|
| `src/` | Data preparation, models, training, evaluation, audit and explainability code |
| `notebooks/` | Experiment notebook(s) |
| `outputs/audit/` | Split-integrity and full-control audit outputs |
| `outputs/evaluation/` | Model evaluation metrics and predictions |
| `outputs/experiments/` | Saved experiment settings |
| `outputs/histories/` | Epoch-level training histories |
| `outputs/figures/` | Curves, robustness plots, error sheets and Grad-CAM figures |
| `report/` | Project report and final validation summary |
| `run_ultrastrict_all.bat` | Ultra-strict retraining/evaluation pipeline |
| `run_full_control_all.bat` | Complete full-control validation pipeline |
| `requirements.txt` | Python dependencies excluding PyTorch wheels |

## Limitations

- PlantVillage is a controlled-image benchmark and does not represent the full variability of field conditions.
- Physical-leaf metadata is incomplete, so unmapped samples cannot all be guaranteed to correspond to unique physical leaves.
- `dHash <= 4` is an operational near-duplicate threshold, not a proof that all visually related samples beyond that threshold are independent.
- PlantDoc OOD labels are manually mapped to compatible PlantVillage classes and should be interpreted as a domain-shift probe rather than a directly equivalent benchmark.
- Post-hoc robustness and Grad-CAM analyses diagnose behavior but do not prove causal feature usage.

## References

1. D. P. Hughes and M. Salathe, *An Open Access Repository of Images on Plant Health to Enable the Development of Mobile Disease Diagnostics*, 2015.
2. S. P. Mohanty, D. P. Hughes, and M. Salathe, *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science, 2016.
3. K. He et al., *Deep Residual Learning for Image Recognition*, CVPR, 2016.
4. M. Tan and Q. V. Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML, 2019.
5. S. Ioffe and C. Szegedy, *Batch Normalization*, ICML, 2015.
6. N. Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR, 2014.
7. I. Loshchilov and F. Hutter, *Decoupled Weight Decay Regularization*, ICLR, 2019.
8. R. R. Selvaraju et al., *Grad-CAM*, ICCV, 2017.
