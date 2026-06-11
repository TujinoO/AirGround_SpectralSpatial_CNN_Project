"""
Hyperspectral Pseudo-Label Generator
高光谱伪标签生成器

A system for processing GF-5 satellite imagery using Physical-Guided PCA and 
Spectral Information Divergence (SID) to generate high-confidence pseudo-labels 
for deep learning model training in lithium-bearing pegmatite mineral identification.

使用物理引导PCA和光谱信息散度(SID)处理GF-5卫星影像的系统，
为含锂伟晶岩矿物识别的深度学习模型训练生成高置信度伪标签。
"""

__version__ = "0.1.0"
__author__ = "Hyperspectral Processing Team"

from .config import ProcessingConfig
from .pipeline import PseudoLabelPipeline

__all__ = ["ProcessingConfig", "PseudoLabelPipeline"]
