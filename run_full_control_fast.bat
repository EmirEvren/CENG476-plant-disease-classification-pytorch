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
echo ============================================================

echo [1/7] Calibration, bootstrap CI, per-class and error audit
"%PYTHON%" src\full_control_core.py --batch-size 32 --bootstrap-samples 1000
if errorlevel 1 exit /b 1

echo [2/7] Best-checkpoint train-validation-test gap analysis
"%PYTHON%" src\analyze_generalization_gap.py
if errorlevel 1 exit /b 1

echo [3/7] Corruption robustness and shortcut occlusion stress
"%PYTHON%" src\robustness_ultrastrict.py --batch-size 32
if errorlevel 1 exit /b 1

echo [4/7] Random-label pipeline sanity control
"%PYTHON%" src\random_label_sanity_ultrastrict.py --epochs 5 --batch-size 64
if errorlevel 1 exit /b 1

echo [5/7] EfficientNet Grad-CAM visual audit
"%PYTHON%" src\gradcam_ultrastrict.py --images-per-sheet 12
if errorlevel 1 exit /b 1

echo [6/7] External PlantDoc out-of-domain evaluation
if not exist "data\external\PlantDoc-Dataset\test" (
  if not exist "data\external" mkdir "data\external"
  echo Cloning official Cropped-PlantDoc dataset...
  git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git "data\external\PlantDoc-Dataset"
  if errorlevel 1 exit /b 1
)
"%PYTHON%" src\evaluate_plantdoc_ood.py --batch-size 32
if errorlevel 1 exit /b 1

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
