# Report Files

The repository contains the project report in DOCX/PDF form together with the final validation summary.

For the **latest verified numerical results and methodology**, use:

- [`FINAL_VALIDATION_SUMMARY.md`](FINAL_VALIDATION_SUMMARY.md)
- the repository root [`README.md`](../README.md)

The DOCX/PDF report files were created before the final full-control / external-OOD / 3-seed validation pass and should be regenerated from the final verified results before submission. They are retained in the repository for document-layout continuity until that update is completed.

Final values that must appear in the regenerated report:

- EfficientNet-B0 fixed ultra-strict run: **99.01% accuracy**, **0.9874 Macro-F1**.
- Validation-selected 50/50 ResNet18 + EfficientNet-B0 ensemble: **99.14% accuracy**, **0.9897 Macro-F1**.
- EfficientNet seed stability: **99.001% mean accuracy**, **0.229 pp standard deviation** across seeds 42, 123 and 777.
- PlantDoc mapped OOD probe: **23.31% EfficientNet accuracy**, **25.00% ensemble accuracy** on 236 images.

The original 99.76% image-level-split ensemble result must be described only as a historical result that motivated the leakage audit, not as the final benchmark.
