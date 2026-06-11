"""
Basic usage example for hyperspectral pseudo-label generator.
高光谱伪标签生成器的基本使用示例。

This example demonstrates how to configure and use the system.
此示例演示如何配置和使用系统。
"""

from hyperspectral_pseudo_label_generator import ProcessingConfig
from hyperspectral_pseudo_label_generator.logging_config import setup_logging, get_logger


def main():
    """Main example function."""
    # Set up logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("Hyperspectral Pseudo-Label Generator - Basic Usage Example")
    logger.info("高光谱伪标签生成器 - 基本使用示例")
    logger.info("=" * 60)
    
    # Create configuration with default values
    logger.info("\n1. Creating configuration with default values...")
    config = ProcessingConfig()
    
    # Validate configuration
    try:
        config.validate()
        logger.info("   Configuration validation: PASSED")
    except ValueError as e:
        logger.error(f"   Configuration validation: FAILED - {e}")
        return
    
    # Display configuration
    logger.info("\n2. Configuration parameters:")
    logger.info(f"   - PCA components: {config.n_components}")
    logger.info(f"   - Ore percentile threshold: {config.ore_percentile}%")
    logger.info(f"   - Non-ore percentile threshold: {config.non_ore_percentile}%")
    logger.info(f"   - Ambiguity threshold: {config.ambiguity_threshold}")
    logger.info(f"   - Numerical epsilon: {config.epsilon}")
    logger.info(f"   - Chunk size: {config.chunk_size} pixels")
    logger.info(f"   - Al-OH wavelength range: {config.aloh_min_wavelength}-{config.aloh_max_wavelength} nm")
    logger.info(f"   - Output directory: {config.output_dir}")
    
    # Create custom configuration
    logger.info("\n3. Creating custom configuration...")
    custom_config = ProcessingConfig(
        n_components=10,
        ore_percentile=1.0,
        non_ore_percentile=3.0,
        ambiguity_threshold=0.05,
        chunk_size=500,
        output_dir="./custom_output"
    )
    
    try:
        custom_config.validate()
        logger.info("   Custom configuration validation: PASSED")
        logger.info(f"   - Custom PCA components: {custom_config.n_components}")
        logger.info(f"   - Custom ore percentile: {custom_config.ore_percentile}%")
    except ValueError as e:
        logger.error(f"   Custom configuration validation: FAILED - {e}")
    
    # Test invalid configuration
    logger.info("\n4. Testing invalid configuration (should fail)...")
    invalid_config = ProcessingConfig(n_components=-5)
    try:
        invalid_config.validate()
        logger.error("   Invalid configuration validation: UNEXPECTED PASS")
    except ValueError as e:
        logger.info(f"   Invalid configuration validation: CORRECTLY FAILED")
        logger.info(f"   Error message: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Example completed successfully!")
    logger.info("示例成功完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
