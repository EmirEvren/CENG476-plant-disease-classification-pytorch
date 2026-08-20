@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo [ERROR] Virtual environment Python not found: %PYTHON%
  exit /b 1
)

echo ============================================================
echo FULL CONTROL - CORE / GAPS / ROBUSTNESS / OOD / SANITY
echo Resume-safe: completed stages are skipped automatically.
echo ============================================================

echo [1/7] Calibration, bootstrap CI, per-class and error audit
if exist "outputs\audit\full_control\core_summary.json" (
  echo       already complete - skipping.
) else (
  "%PYTHON%" src\full_control_core.py --batch-size 32 --bootstrap-samples 1000
  if errorlevel 1 exit /b 1
)

echo [2/7] Best-checkpoint train-validation-test gap analysis
if exist "outputs\audit\full_control\generalization_gap.csv" (
  echo       already complete - skipping.
) else (
  "%PYTHON%" src\analyze_generalization_gap.py
  if errorlevel 1 exit /b 1
)

echo [3/7] Corruption robustness and shortcut occlusion stress
if exist "outputs\audit\full_control\robustness_stress_summary.json" (
  echo       already complete - skipping.
) else (
  "%PYTHON%" src\robustness_ultrastrict.py --batch-size 32
  if errorlevel 1 exit /b 1
)

echo [4/7] Random-label pipeline sanity control
if exist "outputs\audit\full_control\random_label_sanity_summary.json" (
  echo       already complete - skipping.
) else (
  "%PYTHON%" src\random_label_sanity_ultrastrict.py --epochs 5 --batch-size 64
  if errorlevel 1 exit /b 1
)

echo [5/7] EfficientNet Grad-CAM visual audit
if exist "outputs\figures\full_control\efficientnet_gradcam_correct.jpg" if exist "outputs\figures\full_control\efficientnet_gradcam_errors.jpg" (
  echo       already complete - skipping.
) else (
  "%PYTHON%" src\gradcam_ultrastrict.py --images-per-sheet 12
  if errorlevel 1 exit /b 1
)

echo [6/7] External PlantDoc out-of-domain evaluation
if exist "outputs\audit\full_control\plantdoc_ood_summary.json" (
  echo       already complete - skipping.
) else (
  echo       preparing Windows-safe mapped PlantDoc test folders...
  "%PYTHON%" src\download_plantdoc_windows.py
  if errorlevel 1 exit /b 1
  "%PYTHON%" src\evaluate_plantdoc_ood.py --batch-size 32
  if errorlevel 1 exit /b 1
)

echo [7/7] Build consolidated report
"%PYTHON%" src\build_full_control_report.py
if errorlevel 1 exit /b 1

echo ============================================================
echo FAST/MEDIUM FULL CONTROL COMPLETED
echo ============================================================
echo Audit outputs: outputs\audit\full_control
echo Figures: outputs\figures\full_control
echo Report: outputs\audit\full_control\FULL_CONTROL_REPORT.md
endlocal
