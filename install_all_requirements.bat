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

REM Install wheel first (required for flash-attn)
echo Installing wheel (required for flash-attn)...
.venv\Scripts\python.exe -m pip install wheel
if errorlevel 1 (
    echo ERROR: Failed to install wheel
    pause
    exit /b 1
)
echo SUCCESS: wheel installed
echo.

REM Install PyTorch first
echo [1/4] Installing PyTorch (nightly with CUDA 13.0)...
.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)
echo SUCCESS: PyTorch installed
echo.

REM Install flash-attn (pre-built wheel for Python 3.12 + CUDA 13.0)
echo [2/4] Installing flash-attn (pre-built wheel for CUDA 13.0)...
.venv\Scripts\python.exe -m pip install https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.8.3+cu130torch2.9.0cxx11abiTRUE-cp312-cp312-win_amd64.whl
if errorlevel 1 (
    echo ERROR: Failed to install flash-attn
    pause
    exit /b 1
)
echo SUCCESS: flash-attn installed
echo.

REM Install ONNX Runtime GPU (nightly CUDA 13.0)
echo [3/4] Installing ONNX Runtime GPU (nightly CUDA 13.0)...
.venv\Scripts\python.exe -m pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu
if errorlevel 1 (
    echo ERROR: Failed to install ONNX Runtime GPU
    pause
    exit /b 1
)
echo SUCCESS: ONNX Runtime GPU installed
echo.

REM Install additional required Python packages
echo Installing additional required Python packages...
.venv\Scripts\python.exe -m pip install coloredlogs flatbuffers numpy packaging protobuf sympy
if errorlevel 1 (
    echo ERROR: Failed to install additional Python packages
    pause
    exit /b 1
)
echo SUCCESS: Additional Python packages installed
echo.

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
echo Applying compatibility fixes...
echo ========================================
echo.
echo Installing exact versions for librosa compatibility...
.venv\Scripts\python.exe -m pip install "numba==0.59.1" "numpy==1.26.4" "librosa==0.11.0"
if errorlevel 1 (
    echo WARNING: Failed to install compatible versions
    echo You may encounter issues with TTS Audio Suite
) else (
    echo SUCCESS: numba 0.59.1, numpy 1.26.4, librosa 0.11.0 installed
)
echo.

REM Install PyTorch first
echo [0/3] Installing Triton
.venv\Scripts\python.exe -m pip3 install triton-windows
if errorlevel 1 (
    echo ERROR: Failed to install Triton
    pause
    exit /b 1
)
echo SUCCESS: Triton installed
echo.

echo ========================================
echo Installing Additional Tools via winget...
REM Install ffmpeg
winget install "FFmpeg (Shared)"
echo.

echo ========================================
echo All requirements installation completed!
echo ========================================
pause

