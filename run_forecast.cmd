@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_forecast.ps1"
if errorlevel 1 (
    echo.
    echo EMRC Forecast could not start. Read the Python/Dependency error above.
    echo The window will remain open so you can copy the suggested install command.
    pause
)
