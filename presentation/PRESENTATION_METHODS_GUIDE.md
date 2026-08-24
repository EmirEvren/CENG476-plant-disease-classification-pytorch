# CENG 476 Presentation - Methodology-Focused Guide

This presentation is intentionally organized around **model development and training methodology**, not architectural novelty. The main question is how preprocessing, augmentation, regularization, optimization, learning-rate control, transfer learning and evaluation design changed the behavior and reliability of the final classifier.

## Recommended slide flow

### 1. Title
Plant Disease Classification Using Deep Learning and Transfer Learning.

### 2. Problem, Dataset and Development Plan
- 38-class PlantVillage image classification.
- Final audited split: 39,091 train / 4,462 validation / 10,709 locked test.
- Input: RGB 3 x 224 x 224.
- Fixed train/validation/test design; k-fold CV was not used.
- Development sequence: scratch baseline -> training-method experiments -> transfer learning -> ensemble -> integrity/generalization checks.

### 3. Preprocessing and Data Augmentation
Training transform:
- RandomResizedCrop(224, scale=(0.80, 1.00))
- RandomHorizontalFlip(p=0.5)
- RandomRotation(+/-15 degrees)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
- ImageNet normalization

Validation/test transform:
- Resize(256) -> CenterCrop(224) -> ImageNet normalization

Reason: add moderate training variation while keeping validation/test deterministic. No isolated augmentation-on/off ablation was performed, so no isolated causal gain is claimed.

### 4. Baseline CNN as a Controlled Reference
- Four Conv -> BatchNorm -> ReLU -> MaxPool blocks.
- Channels: 32 -> 64 -> 128 -> 256.
- Adaptive average pooling, Dropout(0.40), Linear(256,38).
- 399,142 trainable parameters.
- Purpose: provide a transparent from-scratch reference rather than claim architectural novelty.

### 5. Development Step: Transfer Learning
- ImageNet-pretrained ResNet18 and EfficientNet-B0.
- Full fine-tuning, not frozen feature extraction.
- Backbone LR 1e-4; new classifier LR 5e-4.
- Reason: update useful pretrained features conservatively while allowing the new 38-class head to adapt faster.

### 6. Regularization Methods and Observed Effects
- BatchNorm after every baseline convolution: stabilizes intermediate activations; no isolated BN ablation.
- Dropout: 0.40 baseline, 0.30 transfer heads.
- Controlled dropout pilot: 0.20 -> 0.4990, 0.40 -> 0.4953, 0.60 -> 0.4311 validation Macro-F1. The 0.60 setting was too aggressive.
- AdamW weight decay 1e-4: explicit regularization; no isolated on/off ablation.
- Moderate augmentation: train-only preventive regularization; no isolated on/off ablation.

### 7. Hyperparameter Tuning and Training Control
- Optimizer: AdamW, betas=(0.9,0.999), weight decay=1e-4.
- Baseline LR pilots: 1e-3, 5e-4, 3e-4; final baseline LR=5e-4.
- ReduceLROnPlateau monitors validation loss; factor=0.5, patience=2, min LR=1e-6.
- Checkpoint selection uses validation Macro-F1; validation loss breaks ties.
- Early stopping patience: 6 baseline, 5 transfer. It was implemented as a safeguard, but the selected final runs reached the maximum epoch count before early stopping terminated them.

### 8. Training Dynamics and Overfitting Analysis
Clean-train / validation / test accuracy:
- Custom CNN: 77.89 / 83.55 / 84.62%
- ResNet18: 98.16 / 98.81 / 97.66%
- EfficientNet-B0: 99.74 / 98.68 / 99.01%

The clean-train subset is balanced at 20 images/class = 760 images and is evaluated without random augmentation. EfficientNet does not show a severe same-domain train-to-validation collapse.

### 9. Model-Development Results
- Custom CNN: 84.62% accuracy, Macro-F1 0.7813.
- ResNet18: 97.66%, Macro-F1 0.9686.
- EfficientNet-B0: 99.01%, Macro-F1 0.9874.
- Validation-selected 50/50 ensemble: 99.14%, Macro-F1 0.9897.

The largest development gain came from transfer learning. EfficientNet-B0 was the strongest individual model; the ensemble reduced errors to 92.

### 10. Final Evaluation Beyond Accuracy
Report accuracy together with Macro-F1, per-class performance, confusion matrix and ROC-AUC.
- EfficientNet Macro ROC-AUC: 0.999955.
- Ensemble Macro ROC-AUC: 0.999976.

### 11. Evaluation Integrity and External Generalization
Historical ensemble accuracy was 99.76%, which triggered an integrity audit.
- exact cross-split duplicates: 10
- perceptual near-duplicate pairs: 68
- mapped same-physical-leaf cross-split groups: 4,956
- strict dHash<=4 pairs before final quarantine: 39
- quarantined: 34 train + 4 validation + 0 test

Final detected exact, mapped-leaf and strict dHash<=4 overlap: 0 under the implemented audit protocol. Physical-leaf mapping covers about 75.7%, so this is not a mathematical uniqueness guarantee for every unmapped specimen.

PlantDoc OOD evaluation without retraining:
- 236 mapped images / 27 mapped source classes
- EfficientNet: 23.31%
- Ensemble: 25.00%

Interpretation: strong PlantVillage-domain generalization, weak cross-domain transfer. This is not direct proof of per-image memorization.

### 12. Lessons Learned and Conclusion
- Training methodology and evaluation design are the main contribution, not architectural novelty.
- Transfer learning produced the largest performance gain.
- Too much dropout slowed learning in the controlled pilot.
- Validation loss is useful for LR scheduling; validation Macro-F1 is used for model selection.
- Very high benchmark results should be audited rather than accepted automatically.
- Near-99% PlantVillage accuracy should not be presented as near-99% field accuracy.

### 13. Q&A
Be ready to explain:
- why augmentation is train-only;
- why CrossEntropyLoss receives raw logits;
- why AdamW and weight decay were used;
- why backbone/head learning rates differ;
- why the scheduler monitors loss but checkpointing uses Macro-F1;
- what early stopping did and did not do;
- why k-fold CV was not used;
- what the leakage audit found;
- why PlantDoc performance is much lower.

## Test-set wording

Use this wording during the defense:

> The locked test set was not used for model training, checkpoint selection, hyperparameter tuning or ensemble-weight selection. Test images were used only in a deterministic, model-independent duplicate and near-duplicate integrity audit.

Do not say that the test set was literally never seen, and do not claim mathematical proof of zero leakage.