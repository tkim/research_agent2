# Research Agent 2 Web UI Setup Guide for Windows 11

This guide will help you set up and run the Research Agent 2 web interface on Windows 11.

## Prerequisites

1. **Python 3.7 or higher**
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"

2. **Node.js and npm**
   - Download from [nodejs.org](https://nodejs.org/)
   - Choose the LTS version
   - npm is included with Node.js

3. **Git** (optional but recommended)
   - Download from [git-scm.com](https://git-scm.com/download/win)

## Installation Steps

### 1. Clone or Download the Repository

Using Git:
```bash
git clone https://github.com/tkim/research_agent2.git
cd research_agent2
```

Or download and extract the ZIP file from GitHub.

### 2. Set Up Python Backend

Open PowerShell or Command Prompt as Administrator and navigate to the project directory:

```bash
# Navigate to project directory
cd C:\path\to\research_agent2

# Create virtual environment
python -m venv venv

# Activate virtual environment
# For PowerShell:
.\venv\Scripts\Activate.ps1
# For Command Prompt:
.\venv\Scripts\activate.bat

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Set Up React Frontend

In the same terminal or a new one:

```bash
# Navigate to web UI directory
cd web_ui

# Install Node.js dependencies
npm install

# Build the React app
npm run build
```

### 4. Configure API Keys (Optional)

Create a `config_local.json` file in the project root:

```json
{
  "apis": {
    "newsapi": {
      "auth_config": {
        "api_key": "your_news_api_key_here"
      }
    },
    "google": {
      "auth_config": {
        "api_key": "your_google_api_key_here",
        "search_engine_id": "your_search_engine_id"
      }
    }
  }
}
```

## Running the Application

### Option 1: Using the Convenience Script

Create a batch file `run_research_agent.bat` in the project root:

```batch
@echo off
echo Starting Research Agent 2 Web UI...
call venv\Scripts\activate.bat
start python web_app.py
timeout /t 5
start http://localhost:5000
```

Double-click the batch file to start the application.

### Option 2: Manual Start

1. Open PowerShell or Command Prompt
2. Navigate to the project directory
3. Activate the virtual environment
4. Run the Flask app:

```bash
# Activate virtual environment
.\venv\Scripts\activate.bat

# Start the web server
python web_app.py
```

5. Open your web browser and go to: `http://localhost:5000`

## Windows-Specific Considerations

### Windows Defender / Firewall

When you first run the application, Windows Defender Firewall may ask for permission. Click "Allow access" to enable the web server.

### PowerShell Execution Policy

If you encounter issues running scripts in PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Port Already in Use

If port 5000 is already in use, you can change it by setting an environment variable:

```bash
# PowerShell
$env:PORT = "8080"
python web_app.py

# Command Prompt
set PORT=8080
python web_app.py
```

## Development Mode

For development with hot-reloading:

1. Start the Flask backend:
```bash
python web_app.py
```

2. In a new terminal, start the React development server:
```bash
cd web_ui
npm start
```

The React dev server will run on http://localhost:3000 and proxy API requests to the Flask backend.

## Troubleshooting

### "python" is not recognized

- Ensure Python is added to PATH during installation
- Try using `python3` instead of `python`
- Reinstall Python with "Add to PATH" option checked

### npm command not found

- Ensure Node.js is installed
- Restart your terminal after installation
- Check Node.js is in PATH: `where node`

### Module not found errors

- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### React build errors

- Delete `node_modules` folder and `package-lock.json`
- Run `npm install` again
- Ensure you have sufficient disk space

### CORS errors in browser

- Ensure the Flask app is running
- Check that you're accessing the correct URL
- Clear browser cache and cookies

## Security Considerations

1. **API Keys**: Never commit API keys to version control. Use `config_local.json` which should be in `.gitignore`

2. **Network Access**: By default, the app only accepts connections from localhost. To allow network access, modify `web_app.py`:
   ```python
   app.run(host='0.0.0.0', port=port, debug=False)  # Remove debug=True for production
   ```

3. **HTTPS**: For production use, consider using a reverse proxy (nginx) with SSL certificates

## Performance Optimization

1. **Build React for Production**: Always use `npm run build` for better performance

2. **Caching**: The app includes basic caching. Clear browser cache if you see stale data

3. **Resource Limits**: For large research queries, consider increasing Python's recursion limit or using pagination

## Updates

To update the application:

```bash
# Update code
git pull origin main

# Update Python dependencies
pip install -r requirements.txt --upgrade

# Update and rebuild frontend
cd web_ui
npm update
npm run build
```

## Support

For issues specific to Windows setup:
1. Check Windows Event Viewer for system errors
2. Ensure Windows is up to date
3. Disable antivirus temporarily to test if it's blocking the app
4. Check the GitHub issues page for similar problems

---

Happy researching! 🔍