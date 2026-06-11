"""
Setup script for GSRSL Pipeline package
GSRSL管道包的安装脚本
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file if it exists
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="gsrsl-pipeline",
    version="0.1.0",
    author="GSRSL Team",
    description="Ground Standard Reference Spectral Library data processing pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="hyperspectral remote-sensing spectral-analysis lithology",
    project_urls={
        "Documentation": "https://github.com/gsrsl/gsrsl-pipeline",
        "Source": "https://github.com/gsrsl/gsrsl-pipeline",
    },
)
