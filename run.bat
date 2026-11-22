@echo off
title VoiceTyper Launcher
echo Starting VoiceTyper...
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo VoiceTyper exited with an error.
    pause
)
