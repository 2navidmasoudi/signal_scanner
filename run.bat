@echo off
setlocal
cd /d "%~dp0"
py -3 main.py
if errorlevel 1 (
    echo.
    echo Scanner exited with an error. Check the message above.
    pause
)
endlocal
