@echo off
cd /d %~dp0
python ComfyUI\main.py --listen --async-offload 16 --use-sage-attention --cache-none --disable-smart-memory
pause
