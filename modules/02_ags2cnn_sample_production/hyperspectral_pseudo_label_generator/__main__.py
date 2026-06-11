"""
Entry point for running the module as a script.
将模块作为脚本运行的入口点。

Usage:
    python -m hyperspectral_pseudo_label_generator --help
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())

