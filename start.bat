@echo off
chcp 65001 >nul
echo ========================================
echo    RVC ONNX Exporter 启动器
echo ========================================
echo.

REM 设置Python解释器优先级：venv > 系统Python
set PYTHON_CMD=

REM 检查是否存在虚拟环境
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
    echo [提示] 使用虚拟环境 Python
) else (
    set "PYTHON_CMD=python"
)

REM 检查Python是否安装
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.9+
    pause
    exit /b 1
)

echo [1/3] 检查依赖...
%PYTHON_CMD% -m pip show torch >nul 2>&1
if errorlevel 1 (
    echo        首次运行，正在安装依赖...
    %PYTHON_CMD% -m pip install -r requirements.txt
)
echo        依赖检查完成

echo.
echo [2/3] 启动服务...
echo        服务地址: http://localhost:8000
echo        API文档:  http://localhost:8000/docs
echo.

REM 启动服务
%PYTHON_CMD% api.py

pause
