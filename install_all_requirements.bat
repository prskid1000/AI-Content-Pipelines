@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM COMFYUI - RTX 5070 Ti BLACKWELL SM120
REM PYTHON 3.14 / PYTORCH 2.14 / CUDA 13.2
REM ============================================================
REM
REM Target hardware:
REM   GPU        : NVIDIA RTX 5070 Ti Laptop
REM   Arch       : Blackwell SM120
REM   RAM        : 64 GB
REM   CPU        : Intel Core Ultra 9 275HX
REM
REM Software:
REM   Python     : 3.14
REM   PyTorch    : 2.14.0 + cu132
REM   TorchVision: 0.29.0 + cu132
REM   Sage       : 2.2.0.post6 + cu132 + Torch 2.14
REM   Attention  : PyTorch SDPA (Native CUDA 13.2) / SageAttention
REM   Triton     : 3.8.0.post28
REM
REM IMPORTANT:
REM   This script expects:
REM
REM       .venv\Scripts\python.exe
REM
REM   Python 3.14 venv should already exist.
REM
REM ============================================================


REM ============================================================
REM 0. PATHS
REM ============================================================

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

set "VENV=%ROOT%\.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "PIP=%PYTHON% -m pip"

set "COMFYUI=%ROOT%\ComfyUI"

set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"


echo.
echo ============================================================
echo   COMFYUI RTX 5070 Ti / SM120
echo   PYTHON 3.14 / TORCH 2.14 / CUDA 13.2
echo ============================================================
echo.
echo ROOT    = %ROOT%
echo VENV    = %VENV%
echo PYTHON  = %PYTHON%
echo COMFYUI = %COMFYUI%
echo CUDA    = %CUDA_HOME%
echo.


REM ============================================================
REM 1. VERIFY PYTHON
REM ============================================================

if not exist "%PYTHON%" (
    echo [ERROR] Python virtual environment not found:
    echo.
    echo %PYTHON%
    echo.
    echo Create Python 3.14 venv first.
    pause
    exit /b 1
)

echo [OK] Virtual environment found.
echo.

"%PYTHON%" --version

if errorlevel 1 (
    echo [ERROR] Cannot execute Python.
    pause
    exit /b 1
)

echo.


REM ============================================================
REM 2. VERIFY PYTHON IS 3.14
REM ============================================================

"%PYTHON%" -c "import sys; print('Python version:', sys.version); assert sys.version_info[:2] == (3,14), 'This script requires Python 3.14'"

if errorlevel 1 (
    echo.
    echo [ERROR] This venv is NOT Python 3.14.
    echo.
    echo Current interpreter:
    "%PYTHON%" --version
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Python 3.14 detected.
echo.


REM ============================================================
REM 3. BASIC BUILD TOOLS
REM ============================================================

echo ============================================================
echo [1/10] Updating pip / wheel / packaging / ninja / build
echo ============================================================

%PIP% install --upgrade pip wheel packaging ninja build

if errorlevel 1 (
    echo [WARNING] Some basic build tools failed.
    echo Continuing...
)

echo.


REM ============================================================
REM 4. REMOVE OLD ACCELERATOR PACKAGES
REM ============================================================

echo ============================================================
echo [2/10] Removing incompatible packages (including ABI-mismatched xformers)
echo ============================================================

%PIP% uninstall -y ^
    torch ^
    torchvision ^
    torchaudio ^
    xformers ^
    triton ^
    triton-windows ^
    flash-attn ^
    flash_attn ^
    flash_attn_3 ^
    sageattention

echo.


REM ============================================================
REM 5. INSTALL EXACT PYTORCH, TORCHAUDIO & AUDIO SHIMS
REM ============================================================

echo ============================================================
echo [3/10] Installing PyTorch 2.14.0 + cu132, Torchaudio & Audio Shims
echo ============================================================

%PIP% install --no-cache-dir ^
    torch==2.14.0+cu132 ^
    torchvision==0.29.0+cu132 ^
    --index-url https://download.pytorch.org/whl/cu132

if errorlevel 1 (
    echo.
    echo [ERROR] PyTorch 2.14 cu132 installation failed.
    pause
    exit /b 1
)

echo.
echo [INFO] Installing torchaudio (no-deps) to protect CUDA Torch 2.14...
%PIP% install --no-cache-dir --no-deps torchaudio

echo.
echo [INFO] Installing audioop-lts (Python 3.14 audioop shim for pydub/omnivoice)...
%PIP% install --no-cache-dir audioop-lts

echo.
echo [OK] PyTorch, Torchaudio, and audio compatibility layers installed.
echo.


REM ============================================================
REM 6. VERIFY GPU
REM ============================================================

echo ============================================================
echo [4/10] Verifying CUDA / GPU
echo ============================================================

"%PYTHON%" -c "import torch; print('Torch          :', torch.__version__); print('Torch CUDA     :', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU            :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('Capability     :', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'NONE')"

if errorlevel 1 (
    echo [ERROR] Torch verification failed.
    pause
    exit /b 1
)

echo.


REM ============================================================
REM 7. TORCHCODEC
REM ============================================================

echo ============================================================
echo [5/10] Installing TorchCodec
echo ============================================================

%PIP% install --upgrade torchcodec

if errorlevel 1 (
    echo [WARNING] TorchCodec installation failed.
    echo Continuing...
)

echo.


REM ============================================================
REM 8. FLASH ATTENTION 3
REM ============================================================

echo ============================================================
echo [6/10] Installing FlashAttention 3
echo ============================================================

%PIP% install --no-cache-dir ^
    flash_attn_3 ^
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu132_torch2140

if errorlevel 1 (
    echo.
    echo [WARNING] FlashAttention 3 was not installed.
    echo PyTorch stack will remain untouched; ComfyUI uses SDPA/SageAttention.
)

echo.


REM ============================================================
REM 9. TRITON-WINDOWS
REM ============================================================

echo ============================================================
echo [7/10] Installing Triton-Windows 3.8.0.post28
echo ============================================================

%PIP% install --no-cache-dir triton-windows==3.8.0.post28

if errorlevel 1 (
    echo [WARNING] Triton-Windows installation failed.
    echo Continuing...
)

echo.


REM ============================================================
REM 10. SAGEATTENTION - EXACT PREBUILT WHEEL
REM ============================================================

echo ============================================================
echo [8/10] Installing SageAttention
echo ============================================================

%PIP% install --no-cache-dir --no-deps ^
"https://huggingface.co/ussoewwin/Sage-Attention-for-Windows/resolve/main/sageattention-2.2.0.post6+cu132torch2.14.0-cp314-cp314-win_amd64.whl"

if errorlevel 1 (
    echo.
    echo [WARNING] SageAttention installation failed.
    echo Continuing without SageAttention.
)

echo.


REM ============================================================
REM 11. ONNX RUNTIME GPU
REM ============================================================

echo ============================================================
echo [9/10] Installing ONNX Runtime GPU
echo ============================================================

%PIP% uninstall -y onnxruntime onnxruntime-gpu

%PIP% install --no-cache-dir onnxruntime-gpu==1.30.0

if errorlevel 1 (
    echo [WARNING] ONNX Runtime GPU installation failed.
    echo Continuing...
)

echo.


REM ============================================================
REM 12. COMFYUI
REM ============================================================

echo ============================================================
echo [10/10] Installing / updating ComfyUI
echo ============================================================

if not exist "%COMFYUI%\.git" (
    echo [INFO] ComfyUI not found.
    echo [INFO] Cloning current ComfyUI...
    git clone https://github.com/Comfy-Org/ComfyUI.git "%COMFYUI%"
    if errorlevel 1 (
        echo [ERROR] Failed to clone ComfyUI.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Existing ComfyUI repository found.
    echo [INFO] Updating...
    pushd "%COMFYUI%"
    git pull --ff-only
    if errorlevel 1 (
        echo [WARNING] ComfyUI update failed.
        echo Continuing with existing version.
    )
    popd
)

echo.


REM ============================================================
REM COMFYUI REQUIREMENTS
REM ============================================================

echo ============================================================
echo Installing ComfyUI requirements
echo ============================================================

set "FILTERED_REQ=%TEMP%\comfyui_requirements_filtered.txt"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
 "$src='%COMFYUI%\requirements.txt'; $dst='%FILTERED_REQ%'; Get-Content -LiteralPath $src | Where-Object { $_ -notmatch '^\s*(torch|torchvision|torchaudio|xformers|triton|triton-windows|flash[-_]attn|flash_attn_3|sageattention|onnxruntime)\b' } | Set-Content -Encoding UTF8 -LiteralPath $dst"

if errorlevel 1 (
    echo [ERROR] Could not create filtered requirements.
    pause
    exit /b 1
)

%PIP% install -r "%FILTERED_REQ%"

if errorlevel 1 (
    echo [WARNING] Some ComfyUI dependencies failed.
    echo Continuing...
)

echo.


REM ============================================================
REM CUSTOM NODES
REM ============================================================

echo ============================================================
echo Installing custom nodes
echo ============================================================

call :sync_node "https://github.com/evanspearman/ComfyMath" "%COMFYUI%\custom_nodes\ComfyMath"
call :sync_node "https://github.com/Lightricks/ComfyUI-LTXVideo" "%COMFYUI%\custom_nodes\ComfyUI-LTXVideo"
call :sync_node "https://github.com/ThanaritKanjanametawatAU/ComfyUI-MediaUtilities" "%COMFYUI%\custom_nodes\ComfyUI-MediaUtilities"
call :sync_node "https://github.com/yuvraj108c/ComfyUI-Whisper" "%COMFYUI%\custom_nodes\ComfyUI-Whisper"
call :sync_node "https://github.com/jerrywap/ComfyUI_LoadImageFromHttpURL" "%COMFYUI%\custom_nodes\ComfyUI_LoadImageFromHttpURL"
call :sync_node "https://github.com/ltdrdata/ComfyUI-Manager" "%COMFYUI%\custom_nodes\comfyui-manager"
call :sync_node "https://github.com/gseth/ControlAltAI-Nodes" "%COMFYUI%\custom_nodes\ControlAltAI-Nodes"
call :sync_node "https://github.com/city96/ComfyUI-GGUF" "%COMFYUI%\custom_nodes\ComfyUI-GGUF"
call :sync_node "https://github.com/kijai/ComfyUI-KJNodes" "%COMFYUI%\custom_nodes\ComfyUI-KJNodes"
call :sync_node "https://github.com/1038lab/ComfyUI-RMBG" "%COMFYUI%\custom_nodes\ComfyUI-RMBG"
call :sync_node "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite" "%COMFYUI%\custom_nodes\ComfyUI-VideoHelperSuite"
call :sync_node "https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS.git" "%COMFYUI%\custom_nodes\ComfyUI-OmniVoice-TTS"


REM ============================================================
REM NODE DEPENDENCIES
REM ============================================================

echo.
echo ============================================================
echo Installing custom-node dependencies
echo ============================================================
echo.

call :install_node_requirements "%COMFYUI%\custom_nodes\ComfyMath"
call :install_node_requirements "%COMFYUI%\custom_nodes\ComfyUI-LTXVideo"
call :install_node_requirements "%COMFYUI%\custom_nodes\ComfyUI-MediaUtilities"
call :install_node_requirements "%COMFYUI%\custom_nodes\ComfyUI-Whisper"
call :install_node_requirements "%COMFYUI%\custom_nodes\ComfyUI_LoadImageFromHttpURL"
call :install_node_requirements "%COMFYUI%\custom_nodes\comfyui-manager"
call :install_node_requirements "%COMFYUI%\custom_nodes\ControlAltAI-Nodes"
call :install_node_requirements "%COMFYUI%\custom_nodes\ComfyUI-GGUF"
call :install_node_requirements "%COMFYUI%\custom_nodes\ComfyUI-KJNodes"
call :install_node_requirements "%COMFYUI%\custom_nodes\ComfyUI-RMBG"
call :install_node_requirements "%COMFYUI%\custom_nodes\ComfyUI-VideoHelperSuite"


REM ============================================================
REM OMNIVOICE SPECIAL INSTALL
REM ============================================================

echo.
echo ============================================================
echo Installing OmniVoice TTS
echo ============================================================
echo.

if exist "%COMFYUI%\custom_nodes\ComfyUI-OmniVoice-TTS\install.py" (
    pushd "%COMFYUI%\custom_nodes\ComfyUI-OmniVoice-TTS"
    "%PYTHON%" install.py
    if errorlevel 1 (
        echo [WARNING] OmniVoice install.py reported an error.
    )
    popd
) else (
    echo [WARNING] OmniVoice install.py not found.
)

echo.


REM ============================================================
REM ACCELERATOR INTEGRITY AUDIT (ZERO BANDWIDTH CHECK)
REM ============================================================

echo ============================================================
echo Auditing accelerator stack integrity (No redundant downloads)
echo ============================================================
echo.

"%PYTHON%" -c "import torch; assert torch.__version__ == '2.14.0+cu132' and torch.cuda.is_available(), 'Torch altered'" >nul 2>&1

if errorlevel 1 (
    echo [ALERT] Torch stack was altered by a dependency. Restoring PyTorch...
    %PIP% install --no-cache-dir ^
        torch==2.14.0+cu132 ^
        torchvision==0.29.0+cu132 ^
        --index-url https://download.pytorch.org/whl/cu132
) else (
    echo [OK] PyTorch 2.14.0+cu132 and CUDA bindings remain intact.
)

echo.


REM ============================================================
REM FINAL VERIFICATION
REM ============================================================

echo ============================================================
echo FINAL VERIFICATION
echo ============================================================
echo.

echo ------------------------------------------------------------
echo Python
echo ------------------------------------------------------------
"%PYTHON%" --version

echo.
echo ------------------------------------------------------------
echo PyTorch / CUDA / GPU
echo ------------------------------------------------------------
"%PYTHON%" -c "import torch; print('Torch          :',torch.__version__); print('Torch CUDA     :',torch.version.cuda); print('CUDA available :',torch.cuda.is_available()); print('GPU            :',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('Capability     :',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'NONE')"

echo.
echo ------------------------------------------------------------
echo TorchVision
echo ------------------------------------------------------------
"%PYTHON%" -c "import torchvision; print('TorchVision    :',torchvision.__version__)" 2>nul

echo.
echo ------------------------------------------------------------
echo Audio Stack (Torchaudio / AudioOp / OmniVoice)
echo ------------------------------------------------------------
"%PYTHON%" -c "import torchaudio; print('TorchAudio     :',torchaudio.__version__)" 2>nul
"%PYTHON%" -c "import audioop; print('AudioOp (LTS)  : OK')" 2>nul || echo [WARNING] audioop missing
"%PYTHON%" -c "import omnivoice; print('OmniVoice      : OK')" 2>nul || echo [WARNING] omnivoice import failed

echo.
echo ------------------------------------------------------------
echo Triton
echo ------------------------------------------------------------
"%PYTHON%" -c "import triton; print('Triton         :',getattr(triton,'__version__','unknown'))" 2>nul

echo.
echo ------------------------------------------------------------
echo SageAttention
echo ------------------------------------------------------------
"%PYTHON%" -c "import sageattention; print('SageAttention  : OK')" 2>nul || echo [WARNING] SageAttention import failed

echo.
echo ------------------------------------------------------------
echo FlashAttention
echo ------------------------------------------------------------
"%PYTHON%" -c "import flash_attn_3; print('FlashAttention3: OK')" 2>nul || echo [WARNING] FlashAttention3 import failed

echo.
echo ------------------------------------------------------------
echo ONNX Runtime
echo ------------------------------------------------------------
"%PYTHON%" -c "import onnxruntime as ort; print('ONNX Runtime   :',ort.__version__); print('Providers      :',ort.get_available_providers())" 2>nul

echo.
echo ------------------------------------------------------------
echo TorchCodec
echo ------------------------------------------------------------
"%PYTHON%" -c "import torchcodec; print('TorchCodec     :',torchcodec.__version__)" 2>nul

echo.
echo ============================================================
echo PIP CHECK
echo ============================================================
echo.
%PIP% check

echo.
echo ============================================================
echo SETUP COMPLETE
echo ============================================================
echo.
echo ComfyUI:
echo   %COMFYUI%
echo.
echo Python:
echo   %PYTHON%
echo.
echo Start ComfyUI:
echo.
echo   cd /d "%COMFYUI%"
echo   "%PYTHON%" main.py
echo.
pause
exit /b 0


REM ============================================================
REM FUNCTION: SYNC NODE
REM ============================================================

:sync_node

set "NODE_URL=%~1"
set "NODE_DIR=%~2"

echo.
echo ------------------------------------------------------------
echo NODE
echo URL    : %NODE_URL%
echo TARGET : %NODE_DIR%
echo ------------------------------------------------------------

if not exist "%NODE_DIR%\.git" (
    echo [INFO] Cloning...
    git clone "%NODE_URL%" "%NODE_DIR%"
    if errorlevel 1 (
        echo [WARNING] Clone failed:
        echo %NODE_URL%
    )
) else (
    echo [INFO] Repository already exists.
    echo [INFO] Updating...
    pushd "%NODE_DIR%"
    git pull --ff-only
    if errorlevel 1 (
        echo [WARNING] Pull failed.
        echo Local changes/divergence were NOT overwritten.
    )
    popd
)

exit /b 0


REM ============================================================
REM FUNCTION: INSTALL NODE REQUIREMENTS
REM ============================================================

:install_node_requirements

set "NODE_DIR=%~1"
set "NODE_REQ=%NODE_DIR%\requirements.txt"

if not exist "%NODE_REQ%" (
    echo [INFO] No requirements.txt:
    echo %NODE_DIR%
    exit /b 0
)

echo.
echo ------------------------------------------------------------
echo NODE REQUIREMENTS
echo %NODE_DIR%
echo ------------------------------------------------------------

set "FILTERED_NODE_REQ=%TEMP%\node_requirements_filtered.txt"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
 "$src='%NODE_REQ%'; $dst='%FILTERED_NODE_REQ%'; Get-Content -LiteralPath $src | Where-Object { $_ -notmatch '^\s*(torch|torchvision|torchaudio|xformers|triton|triton-windows|flash[-_]attn|flash_attn_3|sageattention|onnxruntime)\b' } | Set-Content -Encoding UTF8 -LiteralPath $dst"

if errorlevel 1 (
    echo [WARNING] Requirement filtering failed.
    exit /b 0
)

%PIP% install -r "%FILTERED_NODE_REQ%"

if errorlevel 1 (
    echo [WARNING] Node requirements reported an error.
    echo Continuing...
)

exit /b 0