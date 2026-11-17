@echo off
REM ============================================================================
REM US Flight Delay Prediction: End-to-End Big Data Analytics Pipeline
REM Windows One-Click Execution Script
REM ============================================================================

title US Flight Delay Prediction - Big Data Analytics System
color 0A

echo ============================================================================
echo   US Flight Delay Prediction: Hadoop MapReduce to Machine Learning
echo   CA3 Assessment - Windows Launcher
echo ============================================================================
echo.

REM 1. Check Python installation
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PY_CMD=py
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set PY_CMD=python
    ) else (
        echo [ERROR] Python was not found in PATH!
        echo Please install Python 3.10+ from https://www.python.org/
        pause
        exit /b 1
    )
)

echo [Step 1/5] Python detected:
%PY_CMD% --version

REM 2. Check or create virtual environment
if not exist ".venv" (
    echo.
    echo [Step 2/5] Creating Python virtual environment in .venv ...
    %PY_CMD% -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo.
    echo [Step 2/5] Virtual environment (.venv) already exists.
)

REM 3. Activate virtual environment
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Could not activate .venv directly. Continuing with active python...
)

REM 4. Install / Verify dependencies
echo.
echo [Step 3/5] Checking and installing dependencies from requirements.txt ...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some dependencies could not be installed automatically.
)

REM 5. Run automated pipeline verification tests
echo.
echo [Step 4/5] Running automated verification test suite ...
python tests\test_pipeline.py
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some tests reported issues, but proceeding to dashboard launch...
)

REM 6. Launch Streamlit Dashboard
echo.
echo [Step 5/5] Launching Streamlit Big Data Analytics Dashboard ...
echo Point your browser to http://localhost:8501
echo Press Ctrl+C in this console to stop the server.
echo ============================================================================
streamlit run app.py

pause
