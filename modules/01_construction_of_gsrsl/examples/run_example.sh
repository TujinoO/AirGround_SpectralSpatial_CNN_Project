#!/bin/bash
# Run GSRSL Pipeline Example
# 运行GSRSL管道示例

echo "=========================================="
echo "GSRSL Pipeline Example"
echo "GSRSL管道示例"
echo "=========================================="
echo ""
echo "This script runs the complete GSRSL pipeline demo with synthetic data."
echo "该脚本使用合成数据运行完整的GSRSL管道演示。"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8 or higher."
    echo "错误：未找到Python。请安装Python 3.8或更高版本。"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
echo "检查依赖项..."
python -c "import numpy, pandas, scipy, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Required packages not installed."
    echo "错误：未安装所需的包。"
    echo ""
    echo "Please install dependencies:"
    echo "请安装依赖项："
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
fi

echo "✓ All dependencies found"
echo "✓ 找到所有依赖项"
echo ""

# Run the demo
echo "Running demo..."
echo "运行演示..."
echo ""

python examples/demo_build_gsrsl.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Demo completed successfully!"
    echo "演示成功完成！"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Demo failed with exit code: $exit_code"
    echo "演示失败，退出代码：$exit_code"
    echo "=========================================="
fi

exit $exit_code
