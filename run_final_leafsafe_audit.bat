@echo off
setlocal
cd /d "%~dp0"
set PY=.\.venv\Scripts\python.exe

if not exist "%PY%" (
    echo [ERROR] Python virtual environment not found: %PY%
    exit /b 1
)

if not exist "outputs\audit\official_leaf_safe_split_manifest.csv" (
    echo [ERROR] Final leaf-safe manifest not found.
    echo Run: %PY% src\build_official_leaf_safe_manifest.py
    exit /b 1
)

echo ============================================================
echo FINAL LEAF-SAFE LEAKAGE / NEAR-DUPLICATE AUDIT
echo ============================================================
echo This runs SHA-256, mapped leaf-ID, dHash, ImageNet embedding,
echo and visual review-sheet generation on the final manifest.
echo.

%PY% src\audit_final_leafsafe_protocol.py --batch-size 64 --num-workers 2 --top-embedding-pairs 200 --contact-sheet-pairs 24
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo FINAL AUDIT COMPLETED
echo ============================================================
echo Summary:
echo outputs\audit\final_leafsafe_protocol_audit\final_audit_summary.json
echo dHash review sheet:
echo outputs\audit\final_leafsafe_protocol_audit\dhash_review_contact_sheet.jpg
echo Embedding review sheet:
echo outputs\audit\final_leafsafe_protocol_audit\embedding_top_pairs_contact_sheet.jpg
endlocal
