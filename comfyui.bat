@echo off
cd /d %~dp0
python ComfyUI\main.py --listen --disable-pinned-memory --async-offload 16 --use-flash-attention --lowvram --force-fp16 --bf16-unet
pause
