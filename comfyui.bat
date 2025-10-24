@echo off
cd /d %~dp0
python ComfyUI\main.py --listen --lowvram --async-offload
pause
