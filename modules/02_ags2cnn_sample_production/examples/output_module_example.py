"""
Example demonstrating the output module usage.
演示输出模块使用的示例。
"""

import numpy as np
import logging
from hyperspectral_pseudo_label_generator.output import (
    OutputSerializer,
    VisualizationGenerator,
    StatisticsLogger
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Demonstrate output module functionality."""
    
    # Create a sample pseudo-label map
    # 创建示例伪标签图
    print("Creating sample pseudo-label map...")
    np.random.seed(42)
    H, W = 100, 100
    pseudo_labels = np.random.randint(0, 4, size=(H, W), dtype=np.uint8)
    
    # Ensure we have some of each label for demonstration
    # 确保我们有每个标签的一些示例
    pseudo_labels[0:20, 0:20] = 0  # Unlabeled
    pseudo_labels[20:40, 0:20] = 1  # Ore
    pseudo_labels[40:60, 0:20] = 2  # Wall Rock
    pseudo_labels[60:80, 0:20] = 3  # Barren Pegmatite
    
    print(f"Pseudo-label map shape: {pseudo_labels.shape}")
    print(f"Pseudo-label map dtype: {pseudo_labels.dtype}")
    
    # 1. Serialize the pseudo-label map
    # 1. 序列化伪标签图
    print("\n1. Serializing pseudo-label map...")
    output_path = "output/example_pseudo_label_map.npy"
    OutputSerializer.save(pseudo_labels, output_path)
    print(f"   Saved to: {output_path}")
    
    # Verify serialization
    # 验证序列化
    loaded = np.load(output_path)
    print(f"   Verification: Arrays equal = {np.array_equal(loaded, pseudo_labels)}")
    
    # 2. Generate visualization
    # 2. 生成可视化
    print("\n2. Generating visualization...")
    VisualizationGenerator.generate(pseudo_labels, "output")
    print("   Saved to: output/pseudo_label_visualization.png")
    
    # 3. Log statistics
    # 3. 记录统计信息
    print("\n3. Logging statistics...")
    StatisticsLogger.log_summary(pseudo_labels)
    
    print("\n✓ Output module demonstration complete!")
    print("  Check the 'output' directory for generated files.")


if __name__ == "__main__":
    main()
