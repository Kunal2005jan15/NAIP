@echo off
REM =============================================================
REM NAIP - Scheduled Refresh Wrapper (for Windows Task Scheduler)
REM =============================================================
REM Activates the naip conda environment and runs the full
REM live-data refresh orchestrator. Point Task Scheduler's
REM "Action" at THIS file, not directly at the Python script -
REM Task Scheduler runs with a minimal environment that won't
REM have conda activated otherwise.
REM =============================================================

cd /d C:\Users\Kunal\NAIP

call C:\Users\Kunal\anaconda3\condabin\conda.bat activate naip

python src\00_run_full_refresh.py

REM Exit code is preserved from the Python script - 0 = success,
REM 1 = pipeline step failed or sanity gate flagged a problem.
REM Visible in Task Scheduler's history under "Last Run Result".
exit /b %errorlevel%