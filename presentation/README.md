# Presentation Materials

The final presentation is organized around **training methodology and model development**, in line with the CENG 476 presentation guidance.

Main emphasis:

- preprocessing and ImageNet normalization;
- train-only data augmentation;
- Batch Normalization, dropout and AdamW weight decay;
- learning-rate pilots and differential transfer-learning rates;
- ReduceLROnPlateau and validation-driven checkpoint selection;
- early-stopping implementation and its actual behavior in the final runs;
- train/validation/test dynamics and overfitting analysis;
- transfer learning and validation-selected ensemble development;
- leakage auditing and external PlantDoc generalization limits.

See [`PRESENTATION_METHODS_GUIDE.md`](./PRESENTATION_METHODS_GUIDE.md) for the slide-by-slide content and defense-safe wording.

The PowerPoint uses the same dark editorial visual style as the previously approved project deck; the content has been refocused from architecture novelty toward the training/development process.