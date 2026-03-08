@echo off
cd /d %~dp0
python ComfyUI\main.py --cpu --listen --async-offload 16 --cache-none --disable-smart-memory
pause
