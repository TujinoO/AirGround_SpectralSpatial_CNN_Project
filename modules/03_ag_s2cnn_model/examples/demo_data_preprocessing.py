"""
数据预处理示例脚本

演示如何预处理高光谱影像数据，包括数据加载、归一化、格式转换等。

使用方法:
    python examples/demo_data_preprocessing.py --input data/raw_image.tif --output data/processed_image.npy
"""

import os
import sys
import argparse
import numpy as np
from typing import Tuple, Dict

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_hyperspectral_image(file_path: str) -> np.ndarray:
    """
    加载高光谱影像
    
    支持的格式:
        - .npy: NumPy 数组
        - .tif/.tiff: GeoTIFF 格式（需要 rasterio 或 GDAL）
        - .mat: MATLAB 格式（需要 scipy）
    
    参数:
        file_path (str): 文件路径
    
    返回:
        numpy.ndarray: 高光谱影像 (H, W, Bands)
    """
    print(f"加载影像: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.npy':
        # NumPy 格式
        image = np.load(file_path)
        
    elif ext in ['.tif', '.tiff']:
        # GeoTIFF 格式
        try:
            import rasterio
            with rasterio.open(file_path) as src:
                # 读取所有波段
                image = src.read()  # (Bands, H, W)
                # 转换为 (H, W, Bands)
                image = np.transpose(image, (1, 2, 0))
        except ImportError:
            print("错误: 需要安装 rasterio 来读取 TIFF 文件")
            print("安装命令: pip install rasterio")
            raise
    
    elif ext == '.mat':
        # MATLAB 格式
        try:
            from scipy.io import loadmat
            data = loadmat(file_path)
            # 假设数据存储在 'image' 键中
            image = data.get('image', data.get('data', None))
            if image is None:
                # 尝试找到第一个非元数据的数组
                for key, value in data.items():
                    if not key.startswith('__') and isinstance(value, np.ndarray):
                        image = value
                        break
            if image is None:
                raise ValueError("无法在 .mat 文件中找到影像数据")
        except ImportError:
            print("错误: 需要安装 scipy 来读取 MAT 文件")
            print("安装命令: pip install scipy")
            raise
    
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    print(f"影像形状: {image.shape}")
    print(f"数据类型: {image.dtype}")
    print(f"数值范围: [{image.min():.4f}, {image.max():.4f}]")
    
    return image


def normalize_image(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    归一化影像数据
    
    参数:
        image (numpy.ndarray): 输入影像 (H, W, Bands)
        method (str): 归一化方法
            - 'minmax': 最小-最大归一化到 [0, 1]
            - 'zscore': Z-score 标准化
            - 'percentile': 百分位数归一化
    
    返回:
        numpy.ndarray: 归一化后的影像
    """
    print(f"\n归一化影像 (方法: {method})...")
    
    if method == 'minmax':
        # 最小-最大归一化
        min_val = image.min()
        max_val = image.max()
        normalized = (image - min_val) / (max_val - min_val + 1e-8)
        
    elif method == 'zscore':
        # Z-score 标准化
        mean = image.mean()
        std = image.std()
        normalized = (image - mean) / (std + 1e-8)
        
    elif method == 'percentile':
        # 百分位数归一化（去除异常值）
        p2 = np.percentile(image, 2)
        p98 = np.percentile(image, 98)
        normalized = np.clip((image - p2) / (p98 - p2 + 1e-8), 0, 1)
        
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    print(f"归一化后范围: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    return normalized.astype(np.float32)


def remove_bad_bands(image: np.ndarray, bad_bands: list = None) -> Tuple[np.ndarray, list]:
    """
    移除坏波段
    
    参数:
        image (numpy.ndarray): 输入影像 (H, W, Bands)
        bad_bands (list): 坏波段索引列表
    
    返回:
        tuple: (处理后的影像, 保留的波段索引)
    """
    if bad_bands is None or len(bad_bands) == 0:
        print("\n未指定坏波段，保留所有波段")
        return image, list(range(image.shape[2]))
    
    print(f"\n移除坏波段: {bad_bands}")
    
    # 创建保留波段的掩码
    all_bands = set(range(image.shape[2]))
    good_bands = sorted(list(all_bands - set(bad_bands)))
    
    # 选择好波段
    image_clean = image[:, :, good_bands]
    
    print(f"原始波段数: {image.shape[2]}")
    print(f"保留波段数: {image_clean.shape[2]}")
    
    return image_clean, good_bands


def create_ground_truth_from_shapefile(image_shape: Tuple[int, int], 
                                       shapefile_path: str = None) -> np.ndarray:
    """
    从 Shapefile 创建 Ground Truth 标签图
    
    参数:
        image_shape (tuple): 影像形状 (H, W)
        shapefile_path (str): Shapefile 路径
    
    返回:
        numpy.ndarray: Ground Truth 标签图 (H, W)
    """
    print(f"\n创建 Ground Truth 标签图...")
    
    if shapefile_path is None:
        print("警告: 未提供 Shapefile，创建模拟标签图")
        # 创建模拟标签图
        gt = np.zeros(image_shape, dtype=np.int32)
        # 添加一些随机标签
        num_samples = 100
        for _ in range(num_samples):
            i = np.random.randint(0, image_shape[0])
            j = np.random.randint(0, image_shape[1])
            label = np.random.randint(0, 4)
            gt[i, j] = label
        return gt
    
    try:
        import geopandas as gpd
        from rasterio import features
        
        # 读取 Shapefile
        gdf = gpd.read_file(shapefile_path)
        
        # 假设标签存储在 'class' 列中
        if 'class' not in gdf.columns:
            raise ValueError("Shapefile 中未找到 'class' 列")
        
        # 栅格化
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['class']))
        gt = features.rasterize(
            shapes,
            out_shape=image_shape,
            fill=0,
            dtype=np.int32
        )
        
        print(f"Ground Truth 形状: {gt.shape}")
        print(f"类别数: {len(np.unique(gt))}")
        
        return gt
        
    except ImportError:
        print("错误: 需要安装 geopandas 来处理 Shapefile")
        print("安装命令: pip install geopandas")
        raise


def create_gsrsl(num_classes: int, num_bands: int, 
                 reference_spectra_path: str = None) -> Dict[int, np.ndarray]:
    """
    创建地面标准参考光谱库 (GSRSL)
    
    参数:
        num_classes (int): 类别数
        num_bands (int): 波段数
        reference_spectra_path (str): 参考光谱文件路径
    
    返回:
        dict: GSRSL 字典 {class_id: spectrum}
    """
    print(f"\n创建 GSRSL (类别数: {num_classes}, 波段数: {num_bands})...")
    
    if reference_spectra_path is not None and os.path.exists(reference_spectra_path):
        # 从文件加载参考光谱
        print(f"从文件加载: {reference_spectra_path}")
        spectra = np.load(reference_spectra_path, allow_pickle=True)
        
        if isinstance(spectra, dict):
            gsrsl = spectra
        else:
            # 假设是数组格式 (num_classes, num_bands)
            gsrsl = {i: spectra[i] for i in range(len(spectra))}
    else:
        # 创建模拟参考光谱
        print("警告: 创建模拟参考光谱")
        gsrsl = {}
        for class_id in range(num_classes):
            # 生成具有不同特征的光谱
            spectrum = np.random.randn(num_bands).astype(np.float32)
            # 添加一些特征峰
            peak_position = int(num_bands * (class_id + 1) / (num_classes + 1))
            spectrum[peak_position] += 2.0
            gsrsl[class_id] = spectrum
    
    print(f"GSRSL 包含 {len(gsrsl)} 个类别的参考光谱")
    
    return gsrsl


def visualize_data(image: np.ndarray, ground_truth: np.ndarray = None, 
                   save_path: str = 'data_preview.png'):
    """
    可视化数据
    
    参数:
        image (numpy.ndarray): 影像 (H, W, Bands)
        ground_truth (numpy.ndarray): Ground Truth (H, W)
        save_path (str): 保存路径
    """
    import matplotlib.pyplot as plt
    
    print(f"\n生成数据预览...")
    
    # 选择 RGB 波段（假设波段 50, 30, 20 对应 R, G, B）
    if image.shape[2] >= 60:
        rgb_bands = [50, 30, 20]
    else:
        # 如果波段数不足，均匀选择
        rgb_bands = [
            image.shape[2] * 2 // 3,
            image.shape[2] // 2,
            image.shape[2] // 3
        ]
    
    rgb_image = image[:, :, rgb_bands]
    # 归一化到 [0, 1]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
    
    # 创建图形
    if ground_truth is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # RGB 合成图
        axes[0].imshow(rgb_image)
        axes[0].set_title('RGB Composite')
        axes[0].axis('off')
        
        # Ground Truth
        axes[1].imshow(ground_truth, cmap='tab10')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(rgb_image)
        ax.set_title('RGB Composite')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"数据预览已保存到: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='高光谱数据预处理')
    parser.add_argument('--input', type=str, required=True, help='输入影像路径')
    parser.add_argument('--output', type=str, required=True, help='输出影像路径')
    parser.add_argument('--gt-input', type=str, default=None, help='Ground Truth 输入路径')
    parser.add_argument('--gt-output', type=str, default=None, help='Ground Truth 输出路径')
    parser.add_argument('--gsrsl-output', type=str, default=None, help='GSRSL 输出路径')
    parser.add_argument('--normalize', type=str, default='minmax', 
                       choices=['minmax', 'zscore', 'percentile'], help='归一化方法')
    parser.add_argument('--bad-bands', type=int, nargs='+', default=None, 
                       help='坏波段索引列表')
    parser.add_argument('--num-classes', type=int, default=4, help='类别数')
    parser.add_argument('--visualize', action='store_true', help='生成数据预览')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("高光谱数据预处理")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 加载影像
    image = load_hyperspectral_image(args.input)
    
    # 移除坏波段
    if args.bad_bands is not None:
        image, good_bands = remove_bad_bands(image, args.bad_bands)
    
    # 归一化
    image = normalize_image(image, method=args.normalize)
    
    # 保存处理后的影像
    print(f"\n保存处理后的影像到: {args.output}")
    np.save(args.output, image)
    
    # 处理 Ground Truth
    ground_truth = None
    if args.gt_output is not None:
        if args.gt_input is not None:
            # 加载现有的 Ground Truth
            print(f"\n加载 Ground Truth: {args.gt_input}")
            ground_truth = np.load(args.gt_input)
        else:
            # 创建模拟 Ground Truth
            ground_truth = create_ground_truth_from_shapefile(
                image.shape[:2],
                shapefile_path=None
            )
        
        print(f"保存 Ground Truth 到: {args.gt_output}")
        np.save(args.gt_output, ground_truth)
    
    # 创建 GSRSL
    if args.gsrsl_output is not None:
        gsrsl = create_gsrsl(
            num_classes=args.num_classes,
            num_bands=image.shape[2],
            reference_spectra_path=None
        )
        
        print(f"保存 GSRSL 到: {args.gsrsl_output}")
        np.save(args.gsrsl_output, gsrsl)
    
    # 可视化
    if args.visualize:
        visualize_data(
            image,
            ground_truth,
            save_path=os.path.join(os.path.dirname(args.output), 'data_preview.png')
        )
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("预处理完成!")
    print("=" * 60)
    print(f"输出影像: {args.output}")
    print(f"  - 形状: {image.shape}")
    print(f"  - 数据类型: {image.dtype}")
    print(f"  - 数值范围: [{image.min():.4f}, {image.max():.4f}]")
    
    if ground_truth is not None:
        print(f"\nGround Truth: {args.gt_output}")
        print(f"  - 形状: {ground_truth.shape}")
        print(f"  - 类别数: {len(np.unique(ground_truth))}")
        print(f"  - 样本数: {np.sum(ground_truth > 0)}")
    
    if args.gsrsl_output is not None:
        print(f"\nGSRSL: {args.gsrsl_output}")
        print(f"  - 类别数: {args.num_classes}")
        print(f"  - 波段数: {image.shape[2]}")
    
    print("\n提示: 使用处理后的数据更新 config.yaml 中的路径")


if __name__ == '__main__':
    # 示例用法
    if len(sys.argv) == 1:
        print("示例用法:")
        print("\n1. 基本预处理:")
        print("   python examples/demo_data_preprocessing.py \\")
        print("       --input data/raw_image.npy \\")
        print("       --output data/processed_image.npy \\")
        print("       --normalize minmax")
        
        print("\n2. 完整预处理（包括 GT 和 GSRSL）:")
        print("   python examples/demo_data_preprocessing.py \\")
        print("       --input data/raw_image.npy \\")
        print("       --output data/processed_image.npy \\")
        print("       --gt-output data/ground_truth.npy \\")
        print("       --gsrsl-output data/gsrsl.npy \\")
        print("       --num-classes 4 \\")
        print("       --visualize")
        
        print("\n3. 移除坏波段:")
        print("   python examples/demo_data_preprocessing.py \\")
        print("       --input data/raw_image.npy \\")
        print("       --output data/processed_image.npy \\")
        print("       --bad-bands 0 1 2 100 101 102 \\")
        print("       --normalize percentile")
    else:
        main()
