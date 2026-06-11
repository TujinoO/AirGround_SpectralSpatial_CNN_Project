@echo off
REM Run GSRSL Pipeline Example
REM 运行GSRSL管道示例

echo ==========================================
echo GSRSL Pipeline Example
echo GSRSL管道示例
echo ==========================================
echo.
echo This script runs the complete GSRSL pipeline demo with synthetic data.
echo 该脚本使用合成数据运行完整的GSRSL管道演示。
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8 or higher.
    echo 错误：未找到Python。请安装Python 3.8或更高版本。
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
echo 检查依赖项...
python -c "import numpy, pandas, scipy, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Error: Required packages not installed.
    echo 错误：未安装所需的包。
    echo.
    echo Please install dependencies:
    echo 请安装依赖项：
    echo   pip install -r requirements.txt
    echo.
    exit /b 1
)

echo ✓ All dependencies found
echo ✓ 找到所有依赖项
echo.

REM Run the demo
echo Running demo...
echo 运行演示...
echo.

python examples\demo_build_gsrsl.py

if %errorlevel% equ 0 (
    echo.
    echo ==========================================
    echo Demo completed successfully!
    echo 演示成功完成！
    echo ==========================================
) else (
    echo.
    echo ==========================================
    echo Demo failed with exit code: %errorlevel%
    echo 演示失败，退出代码：%errorlevel%
    echo ==========================================
)

exit /b %errorlevel%
