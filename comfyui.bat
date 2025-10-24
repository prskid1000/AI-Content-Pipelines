@echo off
cd /d %~dp0
python ComfyUI\main.py --listen --use-sage-attention
pause
