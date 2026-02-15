@echo off
REM Python dependencies reference: https://deepwiki.com/visualbruno/ComfyUI-Hunyuan3d-2-1/2.1-python-dependencies
echo ========================================
echo Installing ComfyUI-Hunyuan3d-2-1
echo ========================================
echo.

set "ROOT=%~dp0"
cd /d "%ROOT%"

REM Check if .venv exists
if not exist ".venv" (
    echo ERROR: .venv folder not found!
    echo Please create a virtual environment first.
    pause
    exit /b 1
)

REM Clone repo into ComfyUI/custom_nodes if not already present
if not exist "ComfyUI\custom_nodes\ComfyUI-Hunyuan3d-2-1" (
    echo [1/4] Cloning ComfyUI-Hunyuan3d-2-1...
    cd ComfyUI\custom_nodes
    git clone https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1.git
    if errorlevel 1 (
        echo ERROR: Failed to clone ComfyUI-Hunyuan3d-2-1
        pause
        exit /b 1
    )
    cd /d "%ROOT%"
    echo SUCCESS: Repository cloned
) else (
    echo [1/4] ComfyUI-Hunyuan3d-2-1 already exists, skipping clone...
)
echo.

REM Install node requirements.txt
echo [2/4] Installing ComfyUI-Hunyuan3d-2-1 requirements.txt...
if exist "ComfyUI\custom_nodes\ComfyUI-Hunyuan3d-2-1\requirements.txt" (
    .venv\Scripts\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Hunyuan3d-2-1\requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements.txt
        pause
        exit /b 1
    )
    echo SUCCESS: Requirements installed
) else (
    echo ERROR: ComfyUI\custom_nodes\ComfyUI-Hunyuan3d-2-1\requirements.txt not found
    pause
    exit /b 1
)
echo.

REM Install rembg (required by hy3dshape BackgroundRemover)
echo [3/5] Installing rembg...
.venv\Scripts\python.exe -m pip install rembg
if errorlevel 1 (
    echo ERROR: Failed to install rembg
    pause
    exit /b 1
)
echo SUCCESS: rembg installed
echo.

REM Install custom_rasterizer wheel
echo [4/5] Installing custom_rasterizer wheel...
set "WHEEL1=ComfyUI\custom_nodes\ComfyUI-Hunyuan3d-2-1\hy3dpaint\custom_rasterizer\dist\custom_rasterizer-0.1-cp312-cp312-win_amd64.whl"
if exist "%WHEEL1%" (
    .venv\Scripts\python.exe -m pip install "%WHEEL1%"
    if errorlevel 1 (
        echo ERROR: Failed to install custom_rasterizer
        pause
        exit /b 1
    )
    echo SUCCESS: custom_rasterizer installed
) else (
    echo WARNING: %WHEEL1% not found. You may need to build it first.
)
echo.

REM Install mesh_inpaint_processor wheel
echo [5/5] Installing mesh_inpaint_processor wheel...
set "WHEEL2=ComfyUI\custom_nodes\ComfyUI-Hunyuan3d-2-1\hy3dpaint\DifferentiableRenderer\dist\mesh_inpaint_processor-0.0.0-cp312-cp312-win_amd64.whl"
if exist "%WHEEL2%" (
    .venv\Scripts\python.exe -m pip install "%WHEEL2%"
    if errorlevel 1 (
        echo ERROR: Failed to install mesh_inpaint_processor
        pause
        exit /b 1
    )
    echo SUCCESS: mesh_inpaint_processor installed
) else (
    echo WARNING: %WHEEL2% not found. You may need to build it first.
)
echo.

echo ========================================
echo ComfyUI-Hunyuan3d-2-1 installation completed!
echo ========================================
pause
