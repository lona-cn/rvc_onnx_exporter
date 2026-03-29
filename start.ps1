# RVC ONNX Exporter Startup Script

param(
    [int]$Port = 8000
)

# Force output encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   RVC ONNX Exporter Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set working directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Detect Python interpreter
$PythonCmd = $null

# Prefer virtual environment
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonCmd = $VenvPython
    Write-Host "[Info] Using virtual environment Python" -ForegroundColor Yellow
} else {
    $PythonCmd = "python"
}

# Check Python
try {
    $Version = & $PythonCmd --version 2>&1
    Write-Host "[OK] Python version: $Version" -ForegroundColor Green
} catch {
    Write-Host "[Error] Python not found. Please install Python 3.9+" -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit 1
}

Write-Host ""
Write-Host "[1/3] Checking dependencies..." -ForegroundColor Cyan

# Check if torch is installed
$TorchInstalled = & $PythonCmd -m pip show torch 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "      First run detected. Installing dependencies..." -ForegroundColor Yellow
    & $PythonCmd -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[Error] Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit..."
        exit 1
    }
}
Write-Host "      Dependencies OK" -ForegroundColor Green

Write-Host ""
Write-Host "[2/3] Starting service..." -ForegroundColor Cyan
Write-Host "      Service URL: http://localhost:$Port" -ForegroundColor White
Write-Host "      API Docs:   http://localhost:$Port/docs" -ForegroundColor White
Write-Host ""

# Start service
try {
    & $PythonCmd "api.py"
} catch {
    Write-Host "[Error] Failed to start service: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Service stopped" -ForegroundColor Yellow
Read-Host "Press Enter to exit..."
