# Research Agent 2 - Windows Setup Script
# This script copies the project from WSL to Windows and sets it up

Write-Host "========================================"
Write-Host "Research Agent 2 - Windows Setup" -ForegroundColor Cyan
Write-Host "========================================"
Write-Host ""

# Define paths
$sourcePath = "\\wsl.localhost\Ubuntu\home\tkim\github\research_agent2"
$destPath = "$env:USERPROFILE\Documents\research_agent2"

# Check if source exists
if (-not (Test-Path $sourcePath)) {
    Write-Host "ERROR: Source path not found: $sourcePath" -ForegroundColor Red
    Write-Host "Please make sure WSL is running and the path is correct" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Source: $sourcePath" -ForegroundColor Green
Write-Host "Destination: $destPath" -ForegroundColor Green
Write-Host ""

# Check if destination exists
if (Test-Path $destPath) {
    Write-Host "WARNING: Destination folder already exists!" -ForegroundColor Yellow
    $response = Read-Host "Do you want to overwrite it? (Y/N)"
    if ($response -ne 'Y' -and $response -ne 'y') {
        Write-Host "Operation cancelled." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 0
    }
    Write-Host "Removing existing folder..." -ForegroundColor Yellow
    Remove-Item -Path $destPath -Recurse -Force
}

# Copy files
Write-Host "Copying files to Windows filesystem..." -ForegroundColor Cyan
try {
    Copy-Item -Path $sourcePath -Destination $destPath -Recurse -Force -ErrorAction Stop
    Write-Host "Copy completed successfully!" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to copy files" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================"
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "The project is now at: $destPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Open the folder: $destPath"
Write-Host "2. Double-click: run_research_agent.bat"
Write-Host "========================================"
Write-Host ""

$openFolder = Read-Host "Do you want to open the folder now? (Y/N)"
if ($openFolder -eq 'Y' -or $openFolder -eq 'y') {
    Start-Process explorer.exe -ArgumentList $destPath
}

# Optionally run the setup
$runSetup = Read-Host "Do you want to run Research Agent 2 now? (Y/N)"
if ($runSetup -eq 'Y' -or $runSetup -eq 'y') {
    Set-Location $destPath
    & ".\run_research_agent.bat"
}