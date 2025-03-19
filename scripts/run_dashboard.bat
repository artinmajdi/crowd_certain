@echo off
echo =========================================================
echo           Starting Crowd-Certain Dashboard
echo =========================================================
echo.

:: Get the project root directory
set "SCRIPT_DIR=%~dp0"
set "CROWD_CERTAIN_ROOT=%SCRIPT_DIR%..\"
set "PROJECT_ROOT=%CROWD_CERTAIN_ROOT%..\"

:: Check if streamlit is installed
where streamlit >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Streamlit is not installed. Installing dependencies...
    pip install -r "%PROJECT_ROOT%crowd_certain\config\requirements.txt"
    echo.
)

:: Run the dashboard
echo Launching Crowd-Certain Dashboard...
cd "%PROJECT_ROOT%"
streamlit run "%CROWD_CERTAIN_ROOT%utilities\dashboard.py"
pause
