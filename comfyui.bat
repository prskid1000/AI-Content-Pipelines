@echo off
cd /d %~dp0
python ComfyUI\main.py --listen --disable-pinned-memory --async-offload 16
pause
