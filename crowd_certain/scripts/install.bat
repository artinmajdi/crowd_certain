@echo off
echo =========================================================
echo           Crowd-Certain Installation Script
echo =========================================================
echo.

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python
    goto :python_found
)

where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python3
    goto :python_found
)

echo Error: Neither 'python' nor 'python3' commands were found.
echo Please install Python 3.10 or higher before continuing.
echo Visit https://www.python.org/downloads/ for installation instructions.
exit /b 1

:python_found
:: Check Python version
for /f "tokens=*" %%a in ('%PYTHON_CMD% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%a
for /f "tokens=*" %%a in ('%PYTHON_CMD% -c "import sys; print(sys.version_info.major)"') do set PYTHON_MAJOR_VERSION=%%a

if %PYTHON_MAJOR_VERSION% LSS 3 (
    echo Error: Python 3.10 or higher is required, but Python %PYTHON_VERSION% was found.
    echo Please install Python 3.10 or higher before continuing.
    echo Visit https://www.python.org/downloads/ for installation instructions.
    exit /b 1
)

for /f "tokens=*" %%a in ('%PYTHON_CMD% -c "import sys; print(sys.version_info.minor)"') do set PYTHON_MINOR_VERSION=%%a
if %PYTHON_MAJOR_VERSION% EQU 3 (
    if %PYTHON_MINOR_VERSION% LSS 10 (
        echo Warning: Python 3.10 or higher is recommended, but Python %PYTHON_VERSION% was found.
        echo Some features may not work correctly.
        set /p continue_anyway="Do you want to continue anyway? (y/n): "
        if /i not "%continue_anyway%"=="y" (
            echo Installation aborted.
            exit /b 1
        )
    )
)

echo Using Python %PYTHON_VERSION%
echo.

echo Please select installation method:
echo   1) Pip with virtual environment (.venv)
echo   2) Pip (system-wide or in current environment)
echo   3) Conda/Mamba (recommended for complex dependencies)
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    call :install_with_venv
) else if "%choice%"=="2" (
    call :install_with_pip
) else if "%choice%"=="3" (
    call :install_with_conda
) else (
    echo Invalid choice. Exiting.
    exit /b 1
)

echo.
echo =========================================================
echo Crowd-Certain installation process completed!
echo See the documentation in the docs/ directory for usage examples.
echo =========================================================

:: Ask if user wants to activate the environment now
if "%choice%"=="1" (
    echo.
    echo Note: Due to how batch scripts work, we cannot directly activate the environment in your current shell.
    echo Instead, we'll create a small activation script for you.

    :: Create activation script
    echo @echo off > crowd_certain\config\activate.bat
    echo call .venv\Scripts\activate.bat >> crowd_certain\config\activate.bat

    echo.
    echo To activate the virtual environment, run:
    echo crowd_certain\config\activate.bat

) else if "%choice%"=="3" (
    echo.
    echo Note: Due to how batch scripts work, we cannot directly activate the conda environment in your current shell.
    echo Instead, we'll create a small activation script for you.

    :: Create activation script for conda
    echo @echo off > crowd_certain\config\activate.bat
    echo call conda activate crowd-certain >> crowd_certain\config\activate.bat

    echo.
    echo To activate the conda environment, run:
    echo crowd_certain\config\activate.bat
)

exit /b 0

:install_with_venv
echo Creating virtual environment (.venv)...

:: Check if virtual environment already exists
if exist .venv (
    echo A virtual environment (.venv) already exists.
    set /p use_existing="Do you want to use the existing environment? (y/n): "
    if /i "%use_existing%"=="y" (
        echo Using existing virtual environment.
    ) else (
        set /p delete_existing="Do you want to delete the existing environment and create a new one? (y/n): "
        if /i "%delete_existing%"=="y" (
            echo Removing existing virtual environment...
            rmdir /s /q .venv
            echo Creating new virtual environment (.venv)...
            %PYTHON_CMD% -m venv .venv
        ) else (
            echo Installation aborted.
            exit /b 1
        )
    )
) else (
    %PYTHON_CMD% -m venv .venv
)

echo Virtual environment ready.

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -e .

echo.
echo Installation completed successfully!
echo To activate the virtual environment in the future, run:
echo   .venv\Scripts\activate.bat
exit /b 0

:install_with_pip
echo Installing with pip...
pip install -e .
echo Installation completed successfully!
exit /b 0

:install_with_conda
echo Installing with conda...

where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: conda is not installed or not in PATH.
    echo Please install conda first or choose the pip installation method.
    exit /b 1
)

where mamba >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Mamba not found. Installing mamba using conda...
    conda install -c conda-forge mamba -y
)

REM Create conda environment
if defined CONDA_DEFAULT_ENV (
    if /I not "%CONDA_DEFAULT_ENV%"=="base" (
        echo Updating existing conda environment %CONDA_DEFAULT_ENV%...
        echo Removing existing conda environment...
        conda env remove -n crowd-certain
        echo Creating new conda environment from environment.yml...
        mamba env update --file crowd_certain\config\environment.yml
    ) else (
        REM Create a new environment
        echo Creating conda environment from environment.yml...
        mamba env update --file crowd_certain\config\environment.yml
    )
) else (
    REM Create a new environment
    echo Creating conda environment from environment.yml...
    mamba env update --file crowd_certain\config\environment.yml
)

:: Activate the conda environment
echo Activating conda environment...
echo Please activate the conda environment manually with:
echo   conda activate crowd-certain
echo Then run:
echo   pip install -e .

set /p activated="Have you activated the conda environment? (y/n): "
if /i "%activated%"=="y" (
    echo Installing package in development mode...
    pip install -e .
    echo.
    echo Installation completed successfully!
) else (
    echo Please complete the installation by running:
    echo   conda activate crowd-certain
    echo   pip install -e .
)

exit /b 0
