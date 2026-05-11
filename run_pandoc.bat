@echo off
REM Convert the Markdown report to PDF using pandoc + xelatex
REM Requires pandoc and a LaTeX engine installed on the system.

cd /d d:\SAE
IF NOT EXIST SAE_ablation_report.md (
    echo SAE_ablation_report.md not found. Aborting.
    pause
    exit /b 1
)

pandoc SAE_ablation_report.md -o SAE_ablation_report.pdf --pdf-engine=xelatex -V geometry:margin=1in
if %ERRORLEVEL% neq 0 (
    echo Pandoc conversion failed. Ensure pandoc and LaTeX are installed.
) else (
    echo PDF created: d:\SAE\SAE_ablation_report.pdf
)
pause
