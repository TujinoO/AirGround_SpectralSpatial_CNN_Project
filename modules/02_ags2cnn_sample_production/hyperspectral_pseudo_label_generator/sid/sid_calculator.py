"""
SID (Spectral Information Divergence) calculator module.
光谱信息散度(SID)计算器模块。

This module implements SID calculation between pixels and reference spectra.
该模块实现像素与参考光谱之间的SID计算。
"""

import logging
from typing import Dict

import numpy as np

from hyperspectral_pseudo_label_generator.config import ProcessingConfig


logger = logging.getLogger(__name__)


class SIDCalculator:
    """
    Calculates Spectral Information Divergence between pixels and reference spectra.
    计算像素与参考光谱之间的光谱信息散度。
    
    SID is a symmetric divergence measure:
    SID是对称散度度量：
    SID(p, q) = D_KL(p||q) + D_KL(q||p)
    
    where D_KL is the Kullback-Leibler divergence:
    其中D_KL是Kullback-Leibler散度：
    D_KL(p||q) = Σ p(i) * log(p(i) / q(i))
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize SID calculator.
        初始化SID计算器。
        
        Args:
            config: Processing configuration (处理配置)
        """
        self.config = config
        logger.info("Initialized SIDCalculator")
    
    def _to_probability(self, data: np.ndarray) -> np.ndarray:
        """
        Convert data to probability distribution.
        将数据转换为概率分布。
        
        This method normalizes scaled PCA component values to form valid
        probability distributions where each row sums to 1. This is required
        for SID calculation which operates on probability distributions.
        
        该方法将缩放后的PCA成分值归一化以形成有效的概率分布，
        其中每行总和为1。这是SID计算所需的，因为SID操作概率分布。
        
        Args:
            data: Input data (n, K) where n is number of samples and K is features
                 输入数据 (n, K)，其中n是样本数，K是特征数
                 
        Returns:
            Probability distributions (n, K) where each row sums to 1
            概率分布 (n, K)，其中每行总和为1
            
        Notes:
            - Adds epsilon (1e-10) to avoid zeros for numerical stability
              添加epsilon (1e-10) 以避免零值，保证数值稳定性
            - Each row is normalized by dividing by the row sum
              每行通过除以行总和进行归一化
            - Validates that output sums to 1 within numerical tolerance
              验证输出在数值容差内总和为1
        """
        # Add epsilon to avoid zeros
        # 添加epsilon以避免零
        data_safe = data + self.config.epsilon
        
        # Normalize to sum to 1
        # 归一化使总和为1
        row_sums = data_safe.sum(axis=1, keepdims=True)
        prob_dist = data_safe / row_sums
        
        # Validate that probabilities sum to 1 (within numerical tolerance)
        # 验证概率总和为1 (在数值容差内)
        sums = prob_dist.sum(axis=1)
        if not np.allclose(sums, 1.0, rtol=1e-6, atol=1e-9):
            logger.warning(
                f"Probability distribution normalization: some rows do not sum to 1.0. "
                f"Min sum: {sums.min():.10f}, Max sum: {sums.max():.10f}"
            )
        
        return prob_dist
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Calculate Kullback-Leibler divergence D_KL(p||q).
        计算Kullback-Leibler散度 D_KL(p||q)。
        
        The KL divergence measures how one probability distribution diverges
        from a second, expected probability distribution. It is defined as:
        D_KL(p||q) = Σ p(i) * log(p(i) / q(i))
        
        KL散度衡量一个概率分布与第二个预期概率分布的偏离程度。定义为：
        D_KL(p||q) = Σ p(i) * log(p(i) / q(i))
        
        Args:
            p: Probability distribution (n, K) where n is number of samples
              概率分布 (n, K)，其中n是样本数
            q: Probability distribution (1, K) or (n, K)
              概率分布 (1, K) 或 (n, K)
              
        Returns:
            KL divergence for each row (n,)
            每行的KL散度 (n,)
            
        Notes:
            - Adds epsilon for numerical stability to prevent log(0) and division by zero
              添加epsilon以保证数值稳定性，防止log(0)和除以零
            - Uses natural logarithm (base e)
              使用自然对数(以e为底)
            - Result is always non-negative
              结果总是非负的
            - D_KL(p||q) = 0 if and only if p = q
              当且仅当p = q时，D_KL(p||q) = 0
        """
        # D_KL(p||q) = Σ p(i) * log(p(i) / q(i))
        # Add epsilon for numerical stability
        # 添加epsilon以保证数值稳定性
        p_safe = p + self.config.epsilon
        q_safe = q + self.config.epsilon
        
        # Calculate log ratio
        # 计算对数比率
        log_ratio = np.log(p_safe / q_safe)
        
        # Sum over features (axis=1)
        # 对特征求和 (axis=1)
        kl = np.sum(p_safe * log_ratio, axis=1)
        
        # Validate that KL divergence is non-negative
        # 验证KL散度为非负
        if np.any(kl < -1e-6):  # Allow small negative values due to numerical errors
            logger.warning(
                f"KL divergence calculation produced negative values. "
                f"Min value: {kl.min():.10f}. This may indicate numerical instability."
            )
            # Clip to zero to ensure non-negativity
            kl = np.maximum(kl, 0.0)
        
        return kl
    
    def calculate_sid(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Calculate Spectral Information Divergence (SID) between distributions.
        计算分布之间的光谱信息散度(SID)。
        
        SID is a symmetric divergence measure defined as:
        SID(p, q) = D_KL(p||q) + D_KL(q||p)
        
        SID是对称散度度量，定义为：
        SID(p, q) = D_KL(p||q) + D_KL(q||p)
        
        Args:
            p: Probability distribution (n, K) where n is number of samples
              概率分布 (n, K)，其中n是样本数
            q: Probability distribution (1, K) or (n, K)
              概率分布 (1, K) 或 (n, K)
              
        Returns:
            SID scores (n,) where lower scores indicate higher similarity
            SID分数 (n,)，较低的分数表示较高的相似性
            
        Notes:
            - SID is symmetric: SID(p, q) = SID(q, p)
              SID是对称的: SID(p, q) = SID(q, p)
            - SID is always non-negative
              SID总是非负的
            - SID(p, p) = 0
            - Lower SID values indicate more similar distributions
              较低的SID值表示更相似的分布
        """
        # Calculate SID: D_KL(p||q) + D_KL(q||p)
        # 计算SID: D_KL(p||q) + D_KL(q||p)
        kl_pq = self._kl_divergence(p, q)
        kl_qp = self._kl_divergence(q, p)
        
        sid = kl_pq + kl_qp
        
        # Validate that SID is non-negative
        # 验证SID为非负
        if np.any(sid < -1e-6):  # Allow small negative values due to numerical errors
            logger.warning(
                f"SID calculation produced negative values. "
                f"Min value: {sid.min():.10f}. This may indicate numerical instability."
            )
            # Clip to zero to ensure non-negativity
            sid = np.maximum(sid, 0.0)
        
        return sid
    
    def calculate(self, 
                  image: np.ndarray, 
                  reference_spectra: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Calculate SID scores between image pixels and reference spectra.
        计算影像像素与参考光谱之间的SID分数。
        
        This method processes the image in spatial chunks to manage memory
        efficiently for large images. Each pixel is compared against all
        reference spectra to produce a 3D array of SID scores.
        
        该方法在空间块中处理影像以有效管理大型影像的内存。
        每个像素与所有参考光谱进行比较，以产生SID分数的3D数组。
        
        Args:
            image: PCA-transformed and scaled image (H, W, K)
                  PCA转换和缩放后的影像 (H, W, K)
            reference_spectra: Transformed reference spectra for each class
                              每个类别的转换参考光谱
                              Dictionary mapping class_id (0-4) to spectrum (K,)
                              将class_id (0-4) 映射到光谱 (K,) 的字典
                              
        Returns:
            SID scores with shape (H, W, num_classes)
            形状为(H, W, num_classes)的SID分数
            Lower scores indicate higher similarity
            较低的分数表示较高的相似性
            
        Notes:
            - Processes image in chunks of size chunk_size x chunk_size
              以chunk_size x chunk_size大小的块处理影像
            - Converts data to probability distributions before SID calculation
              在SID计算前将数据转换为概率分布
            - Memory efficient for large images
              对大型影像内存高效
        """
        H, W, K = image.shape
        num_classes = len(reference_spectra)
        
        logger.info(f"Starting SID calculation for image shape {image.shape} with {num_classes} classes")
        
        # Initialize output
        # 初始化输出
        sid_scores = np.zeros((H, W, num_classes), dtype=np.float32)
        
        # Process in chunks to manage memory
        # 分块处理以管理内存
        chunk_size = self.config.chunk_size
        
        # Calculate total number of chunks for progress logging
        # 计算总块数用于进度日志
        n_chunks_h = (H + chunk_size - 1) // chunk_size
        n_chunks_w = (W + chunk_size - 1) // chunk_size
        total_chunks = n_chunks_h * n_chunks_w
        processed_chunks = 0
        
        for i in range(0, H, chunk_size):
            for j in range(0, W, chunk_size):
                # Extract chunk
                # 提取块
                i_end = min(i + chunk_size, H)
                j_end = min(j + chunk_size, W)
                chunk = image[i:i_end, j:j_end, :]
                
                # Calculate SID for this chunk
                # 计算此块的SID
                chunk_sid = self._calculate_chunk(chunk, reference_spectra)
                
                # Store results
                # 存储结果
                sid_scores[i:i_end, j:j_end, :] = chunk_sid
                
                # Progress logging
                # 进度日志
                processed_chunks += 1
                if processed_chunks % max(1, total_chunks // 10) == 0:
                    progress = (processed_chunks / total_chunks) * 100
                    logger.info(f"SID calculation progress: {progress:.1f}% ({processed_chunks}/{total_chunks} chunks)")
        
        logger.info(f"SID calculation completed. Output shape: {sid_scores.shape}")
        
        return sid_scores
    
    def _calculate_chunk(self, 
                        chunk: np.ndarray, 
                        reference_spectra: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Calculate SID for a spatial chunk.
        计算空间块的SID。
        
        Args:
            chunk: Image chunk (h, w, K)
                  影像块
            reference_spectra: Reference spectra (参考光谱)
                              Dictionary mapping class_id to spectrum (K,)
                              将class_id映射到光谱 (K,) 的字典
            
        Returns:
            SID scores for chunk (h, w, num_classes)
            块的SID分数
        """
        h, w, K = chunk.shape
        num_classes = len(reference_spectra)
        
        # Reshape chunk to (h*w, K)
        # 重塑块为 (h*w, K)
        pixels = chunk.reshape(-1, K)
        
        # Normalize to probability distributions
        # 归一化为概率分布
        pixels_prob = self._to_probability(pixels)
        
        # Initialize output
        # 初始化输出
        sid_chunk = np.zeros((h * w, num_classes), dtype=np.float32)
        
        # Calculate SID for each class
        # 计算每个类别的SID
        for class_id in sorted(reference_spectra.keys()):
            ref_spectrum = reference_spectra[class_id]
            ref_prob = self._to_probability(ref_spectrum.reshape(1, -1))
            
            # Calculate SID between pixels and reference
            # 计算像素与参考之间的SID
            sid_chunk[:, class_id] = self.calculate_sid(pixels_prob, ref_prob)
        
        # Reshape back to (h, w, num_classes)
        # 重塑回 (h, w, num_classes)
        return sid_chunk.reshape(h, w, num_classes)
