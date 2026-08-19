@echo off
setlocal
cd /d "%~dp0"
set PY=.\.venv\Scripts\python.exe

if not exist "%PY%" (
    echo [ERROR] Python virtual environment not found: %PY%
    exit /b 1
)

if not exist "outputs\audit\final_leafsafe_protocol_audit\dhash_strict_cross_split_pairs.csv" (
    echo [ERROR] Strict dHash CSV not found.
    echo Run: run_final_leafsafe_audit.bat
    exit /b 1
)

echo ============================================================
echo FOCUSED STRICT dHASH REVIEW
 echo ============================================================
%PY% src\review_strict_dhash_pairs.py
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo STRICT dHASH REVIEW COMPLETED
 echo ============================================================
echo Upload this image for final visual review:
echo outputs\audit\final_leafsafe_protocol_audit\strict_dhash_unmapped_contact_sheet.jpg
endlocal
