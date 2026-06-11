"""
AG-S²CNN 安装脚本
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ag-s2cnn",
    version="0.1.0",
    description="AG-S²CNN: Air-Ground Spectral-Spatial CNN for Lithium Ore Prediction",
    author="AG-S²CNN Team",
    author_email="",
    url="",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="deep-learning hyperspectral remote-sensing lithium-ore pytorch",
)
