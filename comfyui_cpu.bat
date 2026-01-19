@echo off
cd /d %~dp0
python ComfyUI\main.py --cpu --listen --disable-pinned-memory --async-offload 16 
pause
