@echo off
chcp 65001 >nul
echo ========================================
echo    RVC ONNX Exporter 启动器
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.9+
    pause
    exit /b 1
)

echo [1/3] 检查依赖...
pip show torch >nul 2>&1
if errorlevel 1 (
    echo        首次运行，正在安装依赖...
    pip install -r requirements.txt
)
echo        依赖检查完成

echo.
echo [2/3] 启动服务...
echo        服务地址: http://localhost:8000
echo        API文档:  http://localhost:8000/docs
echo.

REM 启动服务
python api.py

pause
