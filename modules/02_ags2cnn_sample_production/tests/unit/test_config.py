"""
Unit tests for configuration module.
配置模块的单元测试。
"""

import pytest
from hyperspectral_pseudo_label_generator.config import ProcessingConfig


class TestProcessingConfig:
    """Test ProcessingConfig dataclass."""
    
    def test_default_configuration(self):
        """Test that default configuration is valid."""
        config = ProcessingConfig()
        config.validate()  # Should not raise
        
        # Check default values
        assert config.n_components == 10
        assert config.ore_percentile == 17.0
        assert config.non_ore_percentile == 6.5
        assert config.poor_percentile == 12.0
        assert config.wall_percentile == 8.5
        assert config.ambiguity_threshold == 2e-7
        assert config.epsilon == 1e-10
        assert config.chunk_size == 1000
        assert config.aloh_min_wavelength == 2150.0
        assert config.aloh_max_wavelength == 2250.0
        assert config.amcs_shadow_corr_threshold == 0.14
        assert config.amcs_snr_threshold == 1.8
        assert config.amcs_moran_threshold == 0.10
        assert config.sid_smoothing_window == 3
        assert config.use_illumination_invariant_sid is True
        assert config.sid_invariant_weight == 0.6
        assert config.sid_disagreement_penalty == 0.6
        assert config.pre_shadow_global_percentile == 25.0
        assert config.pre_shadow_local_percentile == 25.0
        assert config.pre_edge_percentile == 90.0
        assert config.pre_unreliable_dilation_radius == 2
        assert config.shadow_exclusion_percentile == 22.0
        assert config.shadow_local_percentile == 22.0
        assert config.shadow_local_window_size == 151
        assert config.edge_exclusion_percentile == 88.0
        assert config.rich_min_neighbors == 1
        assert config.poor_min_neighbors == 1
        assert config.rich_core_window_size == 25
        assert config.rich_core_min_density == 0.015
        assert config.poor_near_rich_radius == 12
        assert config.rich_to_poor_sid_margin == -3e-4
        assert config.poor_confidence_percentile == 2.0
        assert config.output_dir == "./output"
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = ProcessingConfig(
            n_components=10,
            ore_percentile=1.0,
            non_ore_percentile=3.0,
            ambiguity_threshold=0.05,
            chunk_size=500
        )
        config.validate()  # Should not raise
        
        assert config.n_components == 10
        assert config.ore_percentile == 1.0
        assert config.non_ore_percentile == 3.0
        assert config.ambiguity_threshold == 0.05
        assert config.chunk_size == 500
    
    def test_invalid_n_components(self):
        """Test that invalid n_components raises ValueError."""
        config = ProcessingConfig(n_components=0)
        with pytest.raises(ValueError, match="n_components must be positive"):
            config.validate()
        
        config = ProcessingConfig(n_components=-5)
        with pytest.raises(ValueError, match="n_components must be positive"):
            config.validate()
    
    def test_invalid_ore_percentile(self):
        """Test that invalid ore_percentile raises ValueError."""
        config = ProcessingConfig(ore_percentile=0)
        with pytest.raises(ValueError, match="ore_percentile must be between 0 and 100"):
            config.validate()
        
        config = ProcessingConfig(ore_percentile=100)
        with pytest.raises(ValueError, match="ore_percentile must be between 0 and 100"):
            config.validate()
        
        config = ProcessingConfig(ore_percentile=-1)
        with pytest.raises(ValueError, match="ore_percentile must be between 0 and 100"):
            config.validate()
    
    def test_invalid_non_ore_percentile(self):
        """Test that invalid non_ore_percentile raises ValueError."""
        config = ProcessingConfig(non_ore_percentile=0)
        with pytest.raises(ValueError, match="non_ore_percentile must be between 0 and 100"):
            config.validate()
        
        config = ProcessingConfig(non_ore_percentile=101)
        with pytest.raises(ValueError, match="non_ore_percentile must be between 0 and 100"):
            config.validate()
    
    def test_invalid_ambiguity_threshold(self):
        """Test that invalid ambiguity_threshold raises ValueError."""
        config = ProcessingConfig(ambiguity_threshold=-0.1)
        with pytest.raises(ValueError, match="ambiguity_threshold must be non-negative"):
            config.validate()

    def test_invalid_poor_or_wall_percentile(self):
        config = ProcessingConfig(poor_percentile=0)
        with pytest.raises(ValueError, match="poor_percentile must be between 0 and 100"):
            config.validate()

        config = ProcessingConfig(wall_percentile=100)
        with pytest.raises(ValueError, match="wall_percentile must be between 0 and 100"):
            config.validate()
    
    def test_invalid_epsilon(self):
        """Test that invalid epsilon raises ValueError."""
        config = ProcessingConfig(epsilon=0)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            config.validate()
        
        config = ProcessingConfig(epsilon=-1e-10)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            config.validate()
    
    def test_invalid_chunk_size(self):
        """Test that invalid chunk_size raises ValueError."""
        config = ProcessingConfig(chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            config.validate()
        
        config = ProcessingConfig(chunk_size=-100)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            config.validate()
    
    def test_invalid_wavelength_range(self):
        """Test that invalid wavelength range raises ValueError."""
        # Min >= Max
        config = ProcessingConfig(aloh_min_wavelength=2250.0, aloh_max_wavelength=2150.0)
        with pytest.raises(ValueError, match="aloh_min_wavelength must be less than aloh_max_wavelength"):
            config.validate()
        
        # Out of valid range
        config = ProcessingConfig(aloh_min_wavelength=300.0)
        with pytest.raises(ValueError, match="Al-OH wavelength range must be within"):
            config.validate()
        
        config = ProcessingConfig(aloh_max_wavelength=3000.0)
        with pytest.raises(ValueError, match="Al-OH wavelength range must be within"):
            config.validate()

    def test_invalid_sid_smoothing_window(self):
        config = ProcessingConfig(sid_smoothing_window=0)
        with pytest.raises(ValueError, match="sid_smoothing_window must be >= 1"):
            config.validate()

        config = ProcessingConfig(sid_smoothing_window=2)
        with pytest.raises(ValueError, match="sid_smoothing_window must be odd"):
            config.validate()

    def test_invalid_shadow_exclusion_percentile(self):
        config = ProcessingConfig(shadow_exclusion_percentile=-1)
        with pytest.raises(ValueError, match="shadow_exclusion_percentile must be in \\[0, 50\\)"):
            config.validate()

        config = ProcessingConfig(shadow_exclusion_percentile=50)
        with pytest.raises(ValueError, match="shadow_exclusion_percentile must be in \\[0, 50\\)"):
            config.validate()

    def test_invalid_pre_mask_parameters(self):
        config = ProcessingConfig(sid_invariant_weight=1.2)
        with pytest.raises(ValueError, match="sid_invariant_weight must be in \\[0, 1\\]"):
            config.validate()

        config = ProcessingConfig(sid_disagreement_penalty=-0.1)
        with pytest.raises(ValueError, match="sid_disagreement_penalty must be non-negative"):
            config.validate()

        config = ProcessingConfig(pre_shadow_global_percentile=50)
        with pytest.raises(ValueError, match="pre_shadow_global_percentile must be in \\[0, 50\\)"):
            config.validate()

        config = ProcessingConfig(pre_shadow_local_percentile=50)
        with pytest.raises(ValueError, match="pre_shadow_local_percentile must be in \\[0, 50\\)"):
            config.validate()

        config = ProcessingConfig(pre_edge_percentile=49)
        with pytest.raises(ValueError, match="pre_edge_percentile must be in \\[50, 100\\)"):
            config.validate()

        config = ProcessingConfig(pre_unreliable_dilation_radius=-1)
        with pytest.raises(ValueError, match="pre_unreliable_dilation_radius must be non-negative"):
            config.validate()

    def test_invalid_shadow_local_percentile(self):
        config = ProcessingConfig(shadow_local_percentile=50)
        with pytest.raises(ValueError, match="shadow_local_percentile must be in \\[0, 50\\)"):
            config.validate()

    def test_invalid_shadow_local_window_size(self):
        config = ProcessingConfig(shadow_local_window_size=2)
        with pytest.raises(ValueError, match="shadow_local_window_size must be >= 3"):
            config.validate()

        config = ProcessingConfig(shadow_local_window_size=100)
        with pytest.raises(ValueError, match="shadow_local_window_size must be odd"):
            config.validate()

    def test_invalid_edge_exclusion_percentile(self):
        config = ProcessingConfig(edge_exclusion_percentile=49)
        with pytest.raises(ValueError, match="edge_exclusion_percentile must be in \\[50, 100\\)"):
            config.validate()

    def test_invalid_neighbor_constraints(self):
        config = ProcessingConfig(rich_min_neighbors=9)
        with pytest.raises(ValueError, match="rich_min_neighbors must be in \\[0, 8\\]"):
            config.validate()

        config = ProcessingConfig(poor_min_neighbors=-1)
        with pytest.raises(ValueError, match="poor_min_neighbors must be in \\[0, 8\\]"):
            config.validate()

    def test_invalid_rich_core_constraints(self):
        config = ProcessingConfig(rich_core_window_size=2)
        with pytest.raises(ValueError, match="rich_core_window_size must be >= 3"):
            config.validate()

        config = ProcessingConfig(rich_core_window_size=24)
        with pytest.raises(ValueError, match="rich_core_window_size must be odd"):
            config.validate()

        config = ProcessingConfig(rich_core_min_density=1.2)
        with pytest.raises(ValueError, match="rich_core_min_density must be in \\[0, 1\\]"):
            config.validate()

    def test_invalid_transition_constraints(self):
        config = ProcessingConfig(poor_near_rich_radius=-1)
        with pytest.raises(ValueError, match="poor_near_rich_radius must be non-negative"):
            config.validate()

        config = ProcessingConfig(poor_confidence_percentile=0)
        with pytest.raises(ValueError, match="poor_confidence_percentile must be between 0 and 100"):
            config.validate()
