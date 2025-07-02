@echo off
echo ========================================
echo Research Agent 2 - Copy to Windows
echo ========================================
echo.
echo This script will copy Research Agent 2 to your Windows Documents folder
echo.

set SOURCE=\\wsl.localhost\Ubuntu\home\tkim\github\research_agent2
set DEST=%USERPROFILE%\Documents\research_agent2

echo Source: %SOURCE%
echo Destination: %DEST%
echo.

if exist "%DEST%" (
    echo WARNING: Destination folder already exists!
    echo.
    choice /C YN /M "Do you want to overwrite it?"
    if errorlevel 2 goto :cancel
    echo.
    echo Removing existing folder...
    rmdir /S /Q "%DEST%"
)

echo Copying files to Windows filesystem...
xcopy "%SOURCE%" "%DEST%" /E /I /H /Y

if errorlevel 1 (
    echo.
    echo ERROR: Failed to copy files
    echo Please make sure WSL is running and the path is correct
    pause
    exit /b 1
)

echo.
echo ========================================
echo Copy completed successfully!
echo.
echo The project is now at: %DEST%
echo.
echo Next steps:
echo 1. Open the folder: %DEST%
echo 2. Double-click: run_research_agent.bat
echo ========================================
echo.
echo Press any key to open the folder...
pause >nul
start explorer "%DEST%"
exit /b 0

:cancel
echo.
echo Operation cancelled.
pause
exit /b 0