@echo off
cd /d %~dp0
python ComfyUI\main.py --listen --async-offload 16 --cache-none --reserve-vram 0.3
pause
