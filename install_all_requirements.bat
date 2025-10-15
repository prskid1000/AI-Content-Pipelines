@echo off
echo ========================================
echo Installing All Requirements
echo ========================================
echo.

REM Check if .venv exists
if not exist ".venv" (
    echo ERROR: .venv folder not found!
    echo Please create a virtual environment first.
    pause
    exit /b 1
)

REM Install root requirements.txt
echo [1/3] Installing root requirements.txt...
if exist "requirements.txt" (
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install root requirements.txt
        pause
        exit /b 1
    )
    echo SUCCESS: Root requirements installed
) else (
    echo WARNING: Root requirements.txt not found, skipping...
)
echo.

REM Install ComfyUI requirements.txt
echo [2/3] Installing ComfyUI requirements.txt...
if exist "ComfyUI\requirements.txt" (
    .venv\Scripts\python.exe -m pip install -r ComfyUI\requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install ComfyUI requirements.txt
        pause
        exit /b 1
    )
    echo SUCCESS: ComfyUI requirements installed
) else (
    echo WARNING: ComfyUI\requirements.txt not found, skipping...
)
echo.

REM Install all custom_nodes requirements.txt
echo [3/3] Installing custom nodes requirements...
echo.

REM Loop through all subdirectories in ComfyUI\custom_nodes
for /d %%i in (ComfyUI\custom_nodes\*) do (
    if exist "%%i\requirements.txt" (
        echo Installing requirements for: %%~nxi
        .venv\Scripts\python.exe -m pip install -r "%%i\requirements.txt"
        if errorlevel 1 (
            echo WARNING: Failed to install %%~nxi requirements.txt
            echo Continuing with next custom node...
        ) else (
            echo SUCCESS: %%~nxi requirements installed
        )
        echo.
    )
)

echo ========================================
echo All requirements installation completed!
echo ========================================
pause

