"""
Configuration module for the hyperspectral pseudo-label generator.
高光谱伪标签生成器的配置模块。
"""

from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    """
    Configuration parameters for the pseudo-label generation pipeline.
    伪标签生成流程的配置参数。
    """
    # PCA parameters (PCA参数)
    n_components: int = 10  # Number of PCA components to select (选择的PCA成分数量)
    
    # Thresholding parameters (阈值参数)
    ore_percentile: float = 17.0  # Percentile threshold for ore classes (矿石类别的百分位数阈值)
    non_ore_percentile: float = 6.5  # Percentile threshold for non-ore classes (非矿石类别的百分位数阈值)
    poor_percentile: float = 12.0  # Percentile threshold for poor-ore class in 3-class mode (三分类中贫矿类别的百分位数阈值)
    wall_percentile: float = 8.5  # Percentile threshold for wall-rock class in 3-class mode (三分类中围岩类别的百分位数阈值)
    ambiguity_threshold: float = 2e-7  # Threshold for ambiguity detection (歧义检测的阈值)
    
    # Numerical stability (数值稳定性)
    epsilon: float = 1e-10  # Small value to prevent division by zero (防止除以零的小值)
    
    # Memory management (内存管理)
    chunk_size: int = 1000  # Spatial chunk size for processing (处理的空间块大小)
    
    # Wavelength range for Al-OH absorption (Al-OH吸收的波长范围)
    aloh_min_wavelength: float = 2150.0  # nm
    aloh_max_wavelength: float = 2250.0  # nm

    # AMCS parameters (自动化MNF分量优选参数)
    amcs_shadow_corr_threshold: float = 0.14
    amcs_snr_threshold: float = 1.8
    amcs_moran_threshold: float = 0.10

    # Spatial smoothing for SID scores (空间平滑SID分数)
    sid_smoothing_window: int = 3
    use_illumination_invariant_sid: bool = True
    sid_invariant_weight: float = 0.6
    sid_disagreement_penalty: float = 0.6

    pre_shadow_global_percentile: float = 25.0
    pre_shadow_local_percentile: float = 25.0
    pre_edge_percentile: float = 90.0
    pre_unreliable_dilation_radius: int = 2

    # Post-classification shadow suppression (分类后阴影抑制)
    shadow_exclusion_percentile: float = 22.0
    shadow_local_percentile: float = 22.0
    shadow_local_window_size: int = 151
    edge_exclusion_percentile: float = 88.0
    rich_min_neighbors: int = 1
    poor_min_neighbors: int = 1
    rich_core_window_size: int = 25
    rich_core_min_density: float = 0.015
    poor_near_rich_radius: int = 12
    rich_to_poor_sid_margin: float = -3e-4
    poor_confidence_percentile: float = 2.0
    
    # Output paths (输出路径)
    output_dir: str = "./output"
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        验证配置参数。
        
        Raises:
            ValueError: If any parameter is invalid (如果任何参数无效)
        """
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        if not (0 < self.ore_percentile < 100):
            raise ValueError("ore_percentile must be between 0 and 100")
        if not (0 < self.non_ore_percentile < 100):
            raise ValueError("non_ore_percentile must be between 0 and 100")
        if not (0 < self.poor_percentile < 100):
            raise ValueError("poor_percentile must be between 0 and 100")
        if not (0 < self.wall_percentile < 100):
            raise ValueError("wall_percentile must be between 0 and 100")
        if self.ambiguity_threshold < 0:
            raise ValueError("ambiguity_threshold must be non-negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.aloh_min_wavelength >= self.aloh_max_wavelength:
            raise ValueError("aloh_min_wavelength must be less than aloh_max_wavelength")
        if self.aloh_min_wavelength < 400 or self.aloh_max_wavelength > 2500:
            raise ValueError("Al-OH wavelength range must be within [400, 2500] nm")
        if not (0 <= self.amcs_shadow_corr_threshold <= 1):
            raise ValueError("amcs_shadow_corr_threshold must be between 0 and 1")
        if self.amcs_snr_threshold <= 0:
            raise ValueError("amcs_snr_threshold must be positive")
        if not (-1 <= self.amcs_moran_threshold <= 1):
            raise ValueError("amcs_moran_threshold must be between -1 and 1")
        if self.sid_smoothing_window < 1:
            raise ValueError("sid_smoothing_window must be >= 1")
        if self.sid_smoothing_window % 2 == 0:
            raise ValueError("sid_smoothing_window must be odd")
        if not (0 <= self.sid_invariant_weight <= 1):
            raise ValueError("sid_invariant_weight must be in [0, 1]")
        if self.sid_disagreement_penalty < 0:
            raise ValueError("sid_disagreement_penalty must be non-negative")
        if not (0 <= self.pre_shadow_global_percentile < 50):
            raise ValueError("pre_shadow_global_percentile must be in [0, 50)")
        if not (0 <= self.pre_shadow_local_percentile < 50):
            raise ValueError("pre_shadow_local_percentile must be in [0, 50)")
        if not (50 <= self.pre_edge_percentile < 100):
            raise ValueError("pre_edge_percentile must be in [50, 100)")
        if self.pre_unreliable_dilation_radius < 0:
            raise ValueError("pre_unreliable_dilation_radius must be non-negative")
        if not (0 <= self.shadow_exclusion_percentile < 50):
            raise ValueError("shadow_exclusion_percentile must be in [0, 50)")
        if not (0 <= self.shadow_local_percentile < 50):
            raise ValueError("shadow_local_percentile must be in [0, 50)")
        if self.shadow_local_window_size < 3:
            raise ValueError("shadow_local_window_size must be >= 3")
        if self.shadow_local_window_size % 2 == 0:
            raise ValueError("shadow_local_window_size must be odd")
        if not (50 <= self.edge_exclusion_percentile < 100):
            raise ValueError("edge_exclusion_percentile must be in [50, 100)")
        if self.rich_min_neighbors < 0 or self.rich_min_neighbors > 8:
            raise ValueError("rich_min_neighbors must be in [0, 8]")
        if self.poor_min_neighbors < 0 or self.poor_min_neighbors > 8:
            raise ValueError("poor_min_neighbors must be in [0, 8]")
        if self.rich_core_window_size < 3:
            raise ValueError("rich_core_window_size must be >= 3")
        if self.rich_core_window_size % 2 == 0:
            raise ValueError("rich_core_window_size must be odd")
        if not (0 <= self.rich_core_min_density <= 1):
            raise ValueError("rich_core_min_density must be in [0, 1]")
        if self.poor_near_rich_radius < 0:
            raise ValueError("poor_near_rich_radius must be non-negative")
        if not (0 < self.poor_confidence_percentile < 100):
            raise ValueError("poor_confidence_percentile must be between 0 and 100")
