"""
Main pipeline orchestrator for hyperspectral pseudo-label generation.
高光谱伪标签生成的主流程编排器。

This module integrates all components: input → MNF/AMCS → SID → classification → output
"""

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np

from .config import ProcessingConfig
from .input.image_loader import HyperspectralImageLoader
from .input.reference_loader import ReferenceSpectraLoader
from .input.metadata_parser import MetadataParser
from .input.validator import InputValidator
from .pca.physical_guided_pca import PhysicalGuidedPCA
from .sid.sid_calculator import SIDCalculator
from .classification.thresholder import PercentileThresholder
from .classification.classifier import WinnerTakesAllClassifier
from .output.serializer import OutputSerializer
from .output.visualizer import VisualizationGenerator
from .output.statistics import StatisticsLogger


class PseudoLabelPipeline:
    """
    Main pipeline orchestrator for pseudo-label generation.
    伪标签生成的主流程编排器。
    
    This class coordinates all processing stages from input loading to output generation.
    此类协调从输入加载到输出生成的所有处理阶段。
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize the pipeline with configuration.
        使用配置初始化流程。
        
        Args:
            config: Processing configuration (处理配置)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        try:
            self.config.validate()
            self.logger.info("Configuration validated successfully")
        except ValueError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise
    
    def run(self,
            image_path: str,
            reference_path: str,
            metadata_path: str,
            output_filename: str = "pseudo_label_map.npy") -> np.ndarray:
        """
        Run the complete pseudo-label generation pipeline.
        运行完整的伪标签生成流程。
        
        Args:
            image_path: Path to hyperspectral image file
                       高光谱影像文件路径
            reference_path: Path to reference spectra file
                           参考光谱文件路径
            metadata_path: Path to wavelength metadata file
                          波长元数据文件路径
            output_filename: Name of output file (default: pseudo_label_map.npy)
                            输出文件名称
                            
        Returns:
            Pseudo-label map with shape (H, W)
            形状为(H, W)的伪标签图
            
        Raises:
            ValueError: If input validation fails (如果输入验证失败)
            IOError: If file operations fail (如果文件操作失败)
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Pseudo-Label Generation Pipeline")
        self.logger.info("开始伪标签生成流程")
        self.logger.info("=" * 60)
        
        try:
            self.logger.info("\n[Stage 1/6] Loading and validating inputs...")
            self.logger.info("[阶段 1/6] 加载和验证输入...")
            
            image, reference_spectra, wavelengths = self._load_inputs(
                image_path, reference_path, metadata_path
            )
            
            self.logger.info("\n[Stage 2/6] Performing Physical-Guided MNF...")
            self.logger.info("[阶段 2/6] 执行物理引导MNF...")
            
            mnf_image, mnf_reference = self._apply_mnf(image, reference_spectra, wavelengths)
            
            self.logger.info("\n[Stage 3/6] Calculating Spectral Information Divergence...")
            self.logger.info("[阶段 3/6] 计算光谱信息散度...")
            
            sid_scores_raw = self._calculate_sid(mnf_image, mnf_reference)
            sid_scores = sid_scores_raw
            if self.config.use_illumination_invariant_sid:
                normalized_image = self._normalize_spectra_cube(mnf_image)
                normalized_reference = {
                    class_id: self._normalize_spectrum(spectrum)
                    for class_id, spectrum in mnf_reference.items()
                }
                sid_scores_invariant = self._calculate_sid(normalized_image, normalized_reference)
                sid_scores = self._fuse_sid_scores(sid_scores_raw, sid_scores_invariant)
            sid_scores = self._aggregate_sid_scores(sid_scores)
            reliable_mask = self._compute_reliable_mask(image=image, wavelengths=wavelengths)
            
            self.logger.info("\n[Stage 4/6] Applying percentile-based thresholding...")
            self.logger.info("[阶段 4/6] 应用基于百分位数的阈值...")
            
            masks = self._apply_thresholding(sid_scores, reliable_mask=reliable_mask)
            
            self.logger.info("\n[Stage 5/6] Classifying pixels with winner-takes-all...")
            self.logger.info("[阶段 5/6] 使用赢者通吃策略分类像素...")
            
            pseudo_labels = self._classify_pixels(sid_scores, masks, reliable_mask=reliable_mask)
            pseudo_labels = self._suppress_shadow_labels(
                pseudo_labels,
                image=image,
                wavelengths=wavelengths,
            )
            pseudo_labels = self._enforce_geological_consistency(
                pseudo_labels,
                sid_scores=sid_scores,
                image=image,
                wavelengths=wavelengths,
            )
            
            self.logger.info("\n[Stage 6/6] Generating outputs...")
            self.logger.info("[阶段 6/6] 生成输出...")
            
            self._generate_outputs(
                pseudo_labels,
                output_filename,
                reference_image_path=image_path,
                image=image,
                wavelengths=wavelengths,
            )
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info("流程成功完成！")
            self.logger.info("=" * 60)
            
            return pseudo_labels
            
        except Exception as e:
            self.logger.error(f"\nPipeline failed: {e}")
            self.logger.error(f"流程失败: {e}")
            raise
    
    def _load_inputs(self,
                     image_path: str,
                     reference_path: str,
                     metadata_path: str) -> tuple:
        """
        Load and validate all input data.
        加载和验证所有输入数据。
        
        Args:
            image_path: Path to hyperspectral image
            reference_path: Path to reference spectra
            metadata_path: Path to wavelength metadata
            
        Returns:
            Tuple of (image, reference_spectra, wavelengths)
            
        Raises:
            ValueError: If validation fails
            IOError: If file loading fails
        """
        try:
            # Load hyperspectral image
            # 加载高光谱影像
            self.logger.info(f"  Loading hyperspectral image from: {image_path}")
            image = HyperspectralImageLoader.load(image_path)
            self.logger.info(f"  Image shape: {image.shape}")
            
            # Load reference spectra
            # 加载参考光谱
            self.logger.info(f"  Loading reference spectra from: {reference_path}")
            reference_spectra = ReferenceSpectraLoader.load(reference_path)
            self.logger.info(f"  Loaded {len(reference_spectra)} reference classes")
            
            # Load wavelength metadata
            # 加载波长元数据
            self.logger.info(f"  Loading wavelength metadata from: {metadata_path}")
            wavelengths = MetadataParser.parse_wavelengths(metadata_path)
            self.logger.info(f"  Loaded {len(wavelengths)} wavelengths")
            
            # Validate all inputs
            # 验证所有输入
            self.logger.info("  Validating inputs...")
            InputValidator.validate_image(image)
            InputValidator.validate_reference_spectra(reference_spectra)
            InputValidator.validate_wavelengths(wavelengths)
            self.logger.info("  Input validation: PASSED")
            
            return image, reference_spectra, wavelengths
            
        except FileNotFoundError as e:
            error_msg = f"File not found: {e}"
            self.logger.error(f"  {error_msg}")
            raise IOError(error_msg)
        except ValueError as e:
            error_msg = f"Input validation failed: {e}"
            self.logger.error(f"  {error_msg}")
            raise
        except Exception as e:
            error_msg = f"Failed to load inputs: {e}"
            self.logger.error(f"  {error_msg}")
            raise IOError(error_msg)
    
    def _apply_mnf(self,
                   image: np.ndarray,
                   reference_spectra: Dict[int, np.ndarray],
                   wavelengths: np.ndarray) -> tuple:
        """
        Apply physical-guided MNF transformation and reconstruction.
        应用物理引导MNF变换与重构。
        
        Args:
            image: Hyperspectral image (H, W, 297)
            reference_spectra: Reference spectra dictionary
            wavelengths: Wavelength array (297,)
            
        Returns:
            Tuple of (mnf_reconstructed_image, mnf_reconstructed_reference)
        """
        try:
            pca = PhysicalGuidedPCA(self.config, wavelengths)

            self.logger.info(f"  Fitting MNF with {self.config.n_components} AMCS components...")
            mnf_features = pca.fit_transform(image)
            self.logger.info(f"  MNF feature shape: {mnf_features.shape}")

            explained_variance = pca.get_explained_variance()
            total_variance = explained_variance.sum() * 100
            self.logger.info(f"  Selected MNF variance ratio: {total_variance:.2f}%")

            reconstructed_image = pca.get_reconstructed_image()
            self.logger.info(f"  Reconstructed image shape: {reconstructed_image.shape}")

            self.logger.info("  Reconstructing reference spectra...")
            reconstructed_reference = pca.transform_reference_reconstructed(reference_spectra)
            self.logger.info(f"  Reconstructed {len(reconstructed_reference)} reference spectra")

            return reconstructed_image, reconstructed_reference
            
        except Exception as e:
            error_msg = f"MNF transformation failed: {e}"
            self.logger.error(f"  {error_msg}")
            raise ValueError(error_msg)
    
    def _calculate_sid(self,
                       image: np.ndarray,
                       reference_spectra: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Calculate Spectral Information Divergence scores.
        计算光谱信息散度分数。
        
        Args:
            image: Input image for SID (H, W, bands/components)
            reference_spectra: Reference spectra for SID
            
        Returns:
            SID scores with shape (H, W, num_classes)
        """
        try:
            sid_calculator = SIDCalculator(self.config)
            H, W, K = image.shape
            self.logger.info(f"  Processing image of size {H}x{W} with {K} components...")
            self.logger.info(f"  Using chunk size: {self.config.chunk_size}")
            
            sid_scores = sid_calculator.calculate(image, reference_spectra)
            self.logger.info(f"  SID scores shape: {sid_scores.shape}")
            
            # Log score statistics
            # 记录分数统计
            for class_id in range(len(reference_spectra)):
                class_scores = sid_scores[:, :, class_id]
                mean_score = class_scores.mean()
                min_score = class_scores.min()
                self.logger.info(f"  Class {class_id}: mean SID = {mean_score:.4f}, min SID = {min_score:.4f}")
            
            return sid_scores
            
        except Exception as e:
            error_msg = f"SID calculation failed: {e}"
            self.logger.error(f"  {error_msg}")
            raise ValueError(error_msg)
    
    def _apply_thresholding(self, sid_scores: np.ndarray, reliable_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply percentile-based thresholding.
        应用基于百分位数的阈值。
        
        Args:
            sid_scores: SID scores (H, W, num_classes)
            
        Returns:
            Binary masks (H, W, num_classes)
        """
        try:
            thresholder = PercentileThresholder(self.config)
            self.logger.info("  Computing percentile thresholds...")
            thresholds: Dict[int, float] = {}
            num_classes = sid_scores.shape[2]
            for class_id in range(num_classes):
                if num_classes == 3:
                    if class_id == 0:
                        percentile = self.config.ore_percentile
                    elif class_id == 1:
                        percentile = self.config.poor_percentile
                    else:
                        percentile = self.config.wall_percentile
                else:
                    percentile = (self.config.ore_percentile if class_id in [0, 1, 2]
                                else self.config.non_ore_percentile)

                class_scores = sid_scores[:, :, class_id]
                if reliable_mask is not None:
                    valid_scores = class_scores[reliable_mask]
                    if valid_scores.size == 0:
                        valid_scores = class_scores.reshape(-1)
                else:
                    valid_scores = class_scores.reshape(-1)
                threshold = float(np.percentile(valid_scores, percentile))
                thresholds[class_id] = threshold
            
            for class_id, threshold in thresholds.items():
                if sid_scores.shape[2] == 3:
                    if class_id == 0:
                        percentile = self.config.ore_percentile
                    elif class_id == 1:
                        percentile = self.config.poor_percentile
                    else:
                        percentile = self.config.wall_percentile
                else:
                    percentile = (self.config.ore_percentile if class_id in [0, 1, 2]
                                else self.config.non_ore_percentile)
                self.logger.info(f"  Class {class_id}: threshold = {threshold:.4f} "
                               f"({percentile}th percentile)")
            
            self.logger.info("  Applying thresholds to create masks...")
            thresholder.thresholds = thresholds
            masks = thresholder.apply_thresholds(sid_scores)
            if reliable_mask is not None:
                masks &= reliable_mask[:, :, None]
            
            for class_id in range(masks.shape[2]):
                passing_pixels = masks[:, :, class_id].sum()
                total_pixels = masks.shape[0] * masks.shape[1]
                percentage = passing_pixels / total_pixels * 100
                self.logger.info(f"  Class {class_id}: {passing_pixels} pixels pass "
                               f"({percentage:.2f}%)")
            
            return masks
            
        except Exception as e:
            error_msg = f"Thresholding failed: {e}"
            self.logger.error(f"  {error_msg}")
            raise ValueError(error_msg)
    
    def _classify_pixels(self,
                        sid_scores: np.ndarray,
                        masks: np.ndarray,
                        reliable_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Classify pixels using winner-takes-all strategy.
        使用赢者通吃策略分类像素。
        
        Args:
            sid_scores: SID scores (H, W, num_classes)
            masks: Binary masks (H, W, num_classes)
            
        Returns:
            Pseudo-label map (H, W)
        """
        try:
            classifier = WinnerTakesAllClassifier(self.config)
            self.logger.info("  Classifying pixels with ambiguity detection...")
            self.logger.info(f"  Ambiguity threshold: {self.config.ambiguity_threshold}")
            
            pseudo_labels = classifier.classify(sid_scores, masks)
            if reliable_mask is not None:
                pseudo_labels = pseudo_labels.copy()
                pseudo_labels[~reliable_mask] = 0
            self.logger.info(f"  Pseudo-label map shape: {pseudo_labels.shape}")
            self.logger.info(f"  Pseudo-label map dtype: {pseudo_labels.dtype}")
            
            unique, counts = np.unique(pseudo_labels, return_counts=True)
            total_pixels = pseudo_labels.size
            
            label_names = {
                0: "Unlabeled/Ambiguous",
                1: "Rich Ore Pegmatite",
                2: "Poor Ore Pegmatite",
                3: "Wall Rock"
            }
            
            self.logger.info("  Classification results:")
            for label, count in zip(unique, counts):
                percentage = count / total_pixels * 100
                self.logger.info(f"    {label_names.get(label, f'Label {label}')}: "
                               f"{count} pixels ({percentage:.2f}%)")
            
            return pseudo_labels
            
        except Exception as e:
            error_msg = f"Classification failed: {e}"
            self.logger.error(f"  {error_msg}")
            raise ValueError(error_msg)

    def _normalize_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        spec = spectrum.astype(np.float32, copy=False)
        norm = float(np.sqrt(np.sum(spec * spec)))
        if norm <= self.config.epsilon:
            return spec.copy()
        return (spec / norm).astype(np.float32, copy=False)

    def _normalize_spectra_cube(self, image: np.ndarray) -> np.ndarray:
        cube = image.astype(np.float32, copy=False)
        norm = np.sqrt(np.sum(cube * cube, axis=2, keepdims=True))
        safe_norm = np.maximum(norm, self.config.epsilon)
        return (cube / safe_norm).astype(np.float32, copy=False)

    def _fuse_sid_scores(self, sid_raw: np.ndarray, sid_invariant: np.ndarray) -> np.ndarray:
        weight = float(self.config.sid_invariant_weight)
        disagreement_penalty = float(self.config.sid_disagreement_penalty)
        disagreement = np.abs(sid_raw - sid_invariant)
        fused = (1.0 - weight) * sid_raw + weight * sid_invariant + disagreement_penalty * disagreement
        self.logger.info(
            "  Fused raw/invariant SID scores with weight %.2f and disagreement penalty %.2f",
            weight,
            disagreement_penalty,
        )
        return fused.astype(np.float32, copy=False)

    def _compute_reliable_mask(self, image: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
        if image.ndim != 3:
            return np.ones(image.shape[:2], dtype=bool)

        gray = self._compute_visible_grayscale(image, wavelengths)
        finite_gray = gray[np.isfinite(gray)]
        if finite_gray.size == 0:
            return np.ones(gray.shape, dtype=bool)

        local_window = int(self.config.shadow_local_window_size)
        local_mean = self._box_mean_filter(gray, window_size=local_window)
        normalized_gray = gray / np.maximum(local_mean, self.config.epsilon)
        finite_normalized = normalized_gray[np.isfinite(normalized_gray)]
        if finite_normalized.size == 0:
            return np.ones(gray.shape, dtype=bool)

        global_threshold = float(np.percentile(finite_gray, self.config.pre_shadow_global_percentile))
        local_threshold = float(np.percentile(finite_normalized, self.config.pre_shadow_local_percentile))
        shadow_mask = (gray <= global_threshold) & (normalized_gray <= local_threshold)

        gradient = self._compute_gradient_magnitude(gray)
        finite_gradient = gradient[np.isfinite(gradient)]
        if finite_gradient.size == 0:
            edge_mask = np.zeros_like(shadow_mask, dtype=bool)
        else:
            edge_threshold = float(np.percentile(finite_gradient, self.config.pre_edge_percentile))
            edge_mask = gradient >= edge_threshold

        unreliable = shadow_mask | edge_mask
        radius = int(self.config.pre_unreliable_dilation_radius)
        if radius > 0:
            window = radius * 2 + 1
            dilated = self._box_mean_filter(unreliable.astype(np.float32, copy=False), window_size=window) > 0
            unreliable = dilated

        reliable_mask = ~unreliable
        reliable_ratio = float(np.mean(reliable_mask) * 100.0)
        self.logger.info("  Reliable pre-mask keeps %.2f%% pixels for thresholding/classification", reliable_ratio)
        return reliable_mask

    def _suppress_shadow_labels(
        self,
        pseudo_labels: np.ndarray,
        image: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        percentile = float(self.config.shadow_exclusion_percentile)
        if percentile <= 0.0:
            self.logger.info("  Shadow suppression disabled (shadow_exclusion_percentile <= 0)")
            return pseudo_labels

        if image.ndim != 3:
            return pseudo_labels

        gray = self._compute_visible_grayscale(image, wavelengths)

        finite = gray[np.isfinite(gray)]
        if finite.size == 0:
            self.logger.warning("  Shadow suppression skipped: no finite grayscale values")
            return pseudo_labels

        global_threshold = float(np.percentile(finite, percentile))
        local_percentile = float(self.config.shadow_local_percentile)
        local_window = int(self.config.shadow_local_window_size)
        local_mean = self._box_mean_filter(gray, window_size=local_window)
        normalized_gray = gray / np.maximum(local_mean, self.config.epsilon)
        finite_normalized = normalized_gray[np.isfinite(normalized_gray)]
        if finite_normalized.size == 0:
            self.logger.warning("  Shadow suppression skipped: invalid local-normalized grayscale")
            return pseudo_labels
        local_threshold = float(np.percentile(finite_normalized, local_percentile))
        deep_local_percentile = max(0.5, local_percentile * 0.6)
        deep_local_threshold = float(np.percentile(finite_normalized, deep_local_percentile))
        shadow_mask = (
            (gray <= global_threshold) & (normalized_gray <= local_threshold)
        ) | (normalized_gray <= deep_local_threshold)
        labeled_mask = pseudo_labels > 0
        removed_mask = shadow_mask & labeled_mask
        removed_count = int(np.count_nonzero(removed_mask))
        if removed_count == 0:
            self.logger.info(
                "  Shadow suppression applied at global P%.2f, local P%.2f and deep-local P%.2f but removed 0 labeled pixels",
                percentile,
                local_percentile,
                deep_local_percentile,
            )
            return pseudo_labels

        filtered = pseudo_labels.copy()
        filtered[removed_mask] = 0
        self.logger.info(
            "  Shadow suppression removed %d labeled pixels at global P%.2f, local P%.2f and deep-local P%.2f",
            removed_count,
            percentile,
            local_percentile,
            deep_local_percentile,
        )
        return filtered

    def _compute_visible_grayscale(self, image: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
        bands = image.shape[2]
        if wavelengths.ndim == 1 and wavelengths.shape[0] == bands:
            vis_mask = (wavelengths >= 430.0) & (wavelengths <= 700.0)
            if np.any(vis_mask):
                return np.nanmean(image[:, :, vis_mask], axis=2).astype(np.float32, copy=False)
        return np.nanmean(image, axis=2).astype(np.float32, copy=False)

    def _enforce_geological_consistency(
        self,
        pseudo_labels: np.ndarray,
        sid_scores: np.ndarray,
        image: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        if pseudo_labels.size == 0 or sid_scores.ndim != 3 or sid_scores.shape[2] < 2:
            return pseudo_labels

        filtered = pseudo_labels.copy()
        gray = self._compute_visible_grayscale(image, wavelengths)
        gradient = self._compute_gradient_magnitude(gray)
        grad_finite = gradient[np.isfinite(gradient)]
        if grad_finite.size > 0:
            edge_threshold = float(np.percentile(grad_finite, self.config.edge_exclusion_percentile))
            edge_mask = gradient >= edge_threshold
            edge_removed = int(np.count_nonzero((filtered > 0) & edge_mask))
            if edge_removed > 0:
                filtered[(filtered > 0) & edge_mask] = 0
                self.logger.info(
                    "  Edge suppression removed %d labels at gradient percentile %.2f",
                    edge_removed,
                    self.config.edge_exclusion_percentile,
                )

        filtered = self._remove_isolated_labels(filtered)

        rich_mask = filtered == 1
        if np.any(rich_mask):
            rich_density = self._box_mean_filter(
                rich_mask.astype(np.float32, copy=False),
                self.config.rich_core_window_size,
            )
            rich_core = rich_mask & (rich_density >= self.config.rich_core_min_density)
            rich_score = sid_scores[:, :, 0]
            poor_score = sid_scores[:, :, 1]
            sid_margin = poor_score - rich_score
            convert_to_poor = rich_mask & (~rich_core) & (sid_margin <= self.config.rich_to_poor_sid_margin)
            converted_count = int(np.count_nonzero(convert_to_poor))
            if converted_count > 0:
                filtered[convert_to_poor] = 2
                self.logger.info(
                    "  Converted %d non-core rich labels to poor labels (SID margin <= %.6f)",
                    converted_count,
                    self.config.rich_to_poor_sid_margin,
                )

        poor_mask = filtered == 2
        if np.any(poor_mask):
            rich_binary = (filtered == 1).astype(np.float32, copy=False)
            radius_window = 2 * int(self.config.poor_near_rich_radius) + 1
            near_rich = self._box_mean_filter(rich_binary, radius_window) > 0
            poor_score = sid_scores[:, :, 1]
            poor_conf_threshold = float(np.percentile(poor_score, self.config.poor_confidence_percentile))
            confident_poor = poor_score <= poor_conf_threshold
            keep_poor = poor_mask & (near_rich | confident_poor)
            removed_poor = poor_mask & (~keep_poor)
            removed_poor_count = int(np.count_nonzero(removed_poor))
            if removed_poor_count > 0:
                filtered[removed_poor] = 0
                self.logger.info(
                    "  Removed %d isolated poor labels away from rich transition belt",
                    removed_poor_count,
                )

        filtered = self._remove_isolated_labels(filtered)
        return filtered

    def _compute_gradient_magnitude(self, gray: np.ndarray) -> np.ndarray:
        grad_y, grad_x = np.gradient(gray.astype(np.float32, copy=False))
        return np.sqrt(grad_x * grad_x + grad_y * grad_y).astype(np.float32, copy=False)

    def _remove_isolated_labels(self, pseudo_labels: np.ndarray) -> np.ndarray:
        filtered = pseudo_labels.copy()
        neighbor_count = self._count_eight_neighbors(filtered > 0)
        rich_mask = filtered == 1
        poor_mask = filtered == 2
        rich_remove = rich_mask & (neighbor_count < self.config.rich_min_neighbors)
        poor_remove = poor_mask & (neighbor_count < self.config.poor_min_neighbors)
        removed = int(np.count_nonzero(rich_remove) + np.count_nonzero(poor_remove))
        if removed > 0:
            filtered[rich_remove | poor_remove] = 0
            self.logger.info(
                "  Removed %d isolated labels (rich < %d, poor < %d neighbors)",
                removed,
                self.config.rich_min_neighbors,
                self.config.poor_min_neighbors,
            )
        return filtered

    def _count_eight_neighbors(self, mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask.astype(np.uint8, copy=False), ((1, 1), (1, 1)), mode="constant", constant_values=0)
        upper = padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        middle = padded[1:-1, :-2] + padded[1:-1, 2:]
        lower = padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        neighbors = upper + middle + lower
        return neighbors.astype(np.int16, copy=False)

    def _box_mean_filter(self, image_2d: np.ndarray, window_size: int) -> np.ndarray:
        radius = max(1, window_size // 2)
        padded = np.pad(image_2d, ((radius, radius), (radius, radius)), mode="reflect")
        integral = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
        k = 2 * radius + 1
        bottom_right = integral[k:, k:]
        top_right = integral[:-k, k:]
        bottom_left = integral[k:, :-k]
        top_left = integral[:-k, :-k]
        total = bottom_right - top_right - bottom_left + top_left
        return (total / float(k * k)).astype(np.float32, copy=False)

    def _aggregate_sid_scores(self, sid_scores: np.ndarray) -> np.ndarray:
        if sid_scores.ndim != 3:
            raise ValueError(f"Expected SID scores to be 3D, got {sid_scores.ndim}D")
        if sid_scores.shape[2] != 5:
            raise ValueError(f"Expected 5 SID classes, got {sid_scores.shape[2]}")

        rich = np.min(sid_scores[:, :, 0:3], axis=2)
        poor = sid_scores[:, :, 3]
        wall = sid_scores[:, :, 4]
        aggregated = np.stack([rich, poor, wall], axis=2)
        self.logger.info("  Aggregated SID scores into 3 classes (rich/poor/wall)")

        window = self.config.sid_smoothing_window
        if window > 1:
            self.logger.info(f"  Applying spatial smoothing to SID scores with {window}x{window} window...")
            smoothed = np.zeros_like(aggregated)
            for c in range(aggregated.shape[2]):
                smoothed[:, :, c] = self._box_mean_filter(aggregated[:, :, c], window)
            return smoothed

        return aggregated
    
    def _generate_outputs(self,
                         pseudo_labels: np.ndarray,
                         output_filename: str,
                         reference_image_path: str,
                         image: np.ndarray,
                         wavelengths: np.ndarray) -> None:
        """
        Generate and save all outputs.
        生成并保存所有输出。
        
        Args:
            pseudo_labels: Pseudo-label map (H, W)
            output_filename: Name of output file
            reference_image_path: Path to the input image (used for GeoTIFF georeferencing)
            image: Input image array (H, W, bands)
            wavelengths: Wavelength array (bands,)
        """
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            self.logger.info(f"  Output directory: {self.config.output_dir}")
            
            output_path = os.path.join(self.config.output_dir, output_filename)
            self.logger.info(f"  Saving pseudo-label map to: {output_path}")
            OutputSerializer.save_with_reference(
                pseudo_labels,
                output_path,
                reference_image_path=reference_image_path,
            )
            self.logger.info("  Pseudo-label map saved successfully")
            
            self.logger.info("  Generating visualization...")
            VisualizationGenerator.generate(
                pseudo_labels,
                self.config.output_dir,
                reference_image_path=reference_image_path,
            )
            VisualizationGenerator.generate_with_rgb_overlay(
                pseudo_labels,
                self.config.output_dir,
                image=image,
                wavelengths=wavelengths,
                reference_image_path=reference_image_path,
            )
            viz_path = os.path.join(self.config.output_dir, "pseudo_label_visualization.png")
            self.logger.info(f"  Visualization saved to: {viz_path}")
            overlay_path = os.path.join(self.config.output_dir, "pseudo_label_overlay_rgb.png")
            self.logger.info(f"  Grayscale overlay visualization saved to: {overlay_path}")
            
            self.logger.info("  Logging summary statistics...")
            StatisticsLogger.log_summary(pseudo_labels)
            
        except Exception as e:
            error_msg = f"Output generation failed: {e}"
            self.logger.error(f"  {error_msg}")
            raise IOError(error_msg)

