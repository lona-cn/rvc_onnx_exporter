# RVC ONNX Exporter 启动脚本

param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   RVC ONNX Exporter 启动器" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 设置工作目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# 设置 Python 解释器
$PythonCmd = $null

# 优先使用虚拟环境
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonCmd = $VenvPython
    Write-Host "[提示] 使用虚拟环境 Python" -ForegroundColor Yellow
} else {
    $PythonCmd = "python"
}

# 检查 Python
try {
    $Version = & $PythonCmd --version 2>&1
    Write-Host "[OK] Python 版本: $Version" -ForegroundColor Green
} catch {
    Write-Host "[错误] 未找到 Python，请先安装 Python 3.9+" -ForegroundColor Red
    Read-Host "按回车键退出..."
    exit 1
}

Write-Host ""
Write-Host "[1/3] 检查依赖..." -ForegroundColor Cyan

# 检查 torch 是否已安装
$TorchInstalled = & $PythonCmd -m pip show torch 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "      首次运行，正在安装依赖..." -ForegroundColor Yellow
    & $PythonCmd -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] 依赖安装失败" -ForegroundColor Red
        Read-Host "按回车键退出..."
        exit 1
    }
}
Write-Host "      依赖检查完成" -ForegroundColor Green

Write-Host ""
Write-Host "[2/3] 启动服务..." -ForegroundColor Cyan
Write-Host "      服务地址: http://localhost:$Port" -ForegroundColor White
Write-Host "      API文档:   http://localhost:$Port/docs" -ForegroundColor White
Write-Host ""

# 启动服务
try {
    & $PythonCmd "api.py"
} catch {
    Write-Host "[错误] 服务启动失败: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "服务已停止" -ForegroundColor Yellow
Read-Host "按回车键退出..."
