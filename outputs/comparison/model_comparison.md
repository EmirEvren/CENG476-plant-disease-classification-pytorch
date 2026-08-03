# PlantVillage Model Comparison

| Model | Parameters | Best Epoch | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline CNN | 399,142 | 15 | 81.21% | 87.42% | 82.01% | 87.11% |
| ResNet18 | 11,196,006 | 12 | 98.83% | 99.26% | 98.69% | 99.27% |
| EfficientNet-B0 | 4,056,226 | 10 | 99.24% | 99.52% | 99.23% | 99.52% |

## Conclusion

The best model is **EfficientNet-B0** with 99.52% test accuracy and 0.9923 test Macro-F1.
EfficientNet-B0 uses 63.8% fewer parameters than ResNet18.
The test set was not used for training, hyperparameter selection, or checkpoint selection.

## Limitation

PlantVillage contains mostly controlled background images. Performance on field images with complex backgrounds, lighting changes, and occlusion may be lower.
