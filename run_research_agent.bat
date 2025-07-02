@echo off
echo ========================================
echo Research Agent 2 Web UI Launcher
echo ========================================
echo.

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo Working directory: %CD%
echo.

REM Check if web_app.py exists
if not exist "web_app.py" (
    echo ERROR: web_app.py not found in %CD%
    echo Please ensure you're running this script from the research_agent2 directory
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Please ensure Python is installed and in PATH
        pause
        exit /b 1
    )
    echo Virtual environment created.
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing Python dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    echo Dependencies installed.
    echo.
)

REM Check if React app is built
if not exist "web_ui\build\index.html" (
    echo Building React frontend...
    cd web_ui
    
    REM Check if npm is available
    where npm >nul 2>nul
    if errorlevel 1 (
        echo ERROR: npm not found. Please install Node.js from https://nodejs.org/
        pause
        exit /b 1
    )
    
    if not exist "node_modules" (
        echo Installing Node.js dependencies...
        call npm install
        if errorlevel 1 (
            echo ERROR: Failed to install Node.js dependencies
            cd ..
            pause
            exit /b 1
        )
    )
    echo Building production bundle...
    call npm run build
    if errorlevel 1 (
        echo ERROR: Failed to build React app
        cd ..
        pause
        exit /b 1
    )
    cd ..
    echo Frontend built successfully.
    echo.
)

REM Start the web server
echo Starting Research Agent 2 server...
echo.
echo The application will open in your default browser in 5 seconds...
echo If it doesn't, navigate to: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server.
echo ========================================

REM Start the browser after a delay
start /b cmd /c "timeout /t 5 >nul && start http://localhost:5000"

REM Run the Flask app with full path
python "%SCRIPT_DIR%web_app.py"