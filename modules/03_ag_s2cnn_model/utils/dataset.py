"""
数据加载与增强模块

包含高光谱数据集类和数据增强类
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import random
import os
from typing import Dict, Any, Tuple


class HyperspectralDataset(Dataset):
    """
    高光谱数据集类
    
    功能:
        1. 从全景高光谱影像中提取邻域切片
        2. 匹配对应的地面标准参考光谱
        3. 处理边界样本的填充
    
    参数:
        image_cube (ndarray): 全景高光谱影像 (H, W, Bands)
        ground_truth (ndarray): 标签图 (H, W)
        gsrsl (dict): 地面标准参考光谱库 {class_id: spectrum}
        spatial_size (int): 邻域大小，默认 13
        augmentation (bool): 是否启用数据增强，默认 False
    
    需求: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
    """
    
    def __init__(self, image_cube, ground_truth, gsrsl, spatial_size=13, augmentation=False,
                 sample_list=None, valid_labels=None):
        """
        初始化数据集
        
        参数:
            image_cube (ndarray): 全景高光谱影像 (H, W, Bands)
            ground_truth (ndarray): 标签图 (H, W)
            gsrsl (dict): 地面标准参考光谱库 {class_id: spectrum}
            spatial_size (int): 邻域大小，默认 13
            augmentation (bool): 是否启用数据增强
        
        需求: 9.1
        """
        self.image_cube = image_cube
        self.ground_truth = ground_truth
        self.gsrsl = gsrsl
        self.spatial_size = spatial_size
        self.augmentation = augmentation
        self.valid_labels = valid_labels
        
        # 验证输入维度
        assert len(image_cube.shape) == 3, \
            f"影像立方体应为 3D 数组 (H, W, Bands)，实际为 {image_cube.shape}"
        assert len(ground_truth.shape) == 2, \
            f"标签图应为 2D 数组 (H, W)，实际为 {ground_truth.shape}"
        assert image_cube.shape[:2] == ground_truth.shape, \
            f"影像和标签的空间维度不匹配: {image_cube.shape[:2]} vs {ground_truth.shape}"
        
        if sample_list is not None:
            self.samples = [(int(i), int(j), int(label)) for i, j, label in sample_list]
        else:
            self.samples = self._extract_samples(valid_labels=self.valid_labels)
        
        # 数据增强器
        if augmentation:
            self.augmenter = DataAugmentation()
        else:
            self.augmenter = None
    
    def _extract_samples(self, valid_labels=None):
        """
        提取所有非零标签的样本坐标
        
        返回:
            list: 样本列表，每个元素为 (i, j, label)
        
        需求: 9.2, 9.3
        """
        samples = []
        h, w = self.ground_truth.shape
        
        valid_set = None if valid_labels is None else set(int(v) for v in valid_labels)

        for i in range(h):
            for j in range(w):
                label = int(self.ground_truth[i, j])
                if valid_set is None:
                    if label > 0:
                        samples.append((i, j, label))
                elif label in valid_set:
                    samples.append((i, j, label))
        
        return samples
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        参数:
            idx (int): 样本索引
        
        返回:
            x_sat (Tensor): 卫星流 (1, Bands, S, S)
            x_ref (Tensor): 地面流 (1, Bands, 1, 1)
            label (int): 类别标签
        
        需求: 9.3, 9.4, 9.5, 9.6
        """
        i, j, label = self.samples[idx]
        half_size = self.spatial_size // 2
        h, w, bands = self.image_cube.shape
        
        # 计算邻域边界
        i_start = i - half_size
        i_end = i + half_size + 1
        j_start = j - half_size
        j_end = j + half_size + 1
        
        # 边界填充处理 (需求 9.6)
        # 如果邻域超出影像边界，使用镜像填充 (Mirror Padding)
        if i_start < 0 or i_end > h or j_start < 0 or j_end > w:
            # 计算各方向需要填充的大小
            pad_top = max(0, -i_start)
            pad_bottom = max(0, i_end - h)
            pad_left = max(0, -j_start)
            pad_right = max(0, j_end - w)
            
            # 提取边界内的有效区域
            valid_i_start = max(0, i_start)
            valid_i_end = min(h, i_end)
            valid_j_start = max(0, j_start)
            valid_j_end = min(w, j_end)
            valid_patch = self.image_cube[valid_i_start:valid_i_end, valid_j_start:valid_j_end, :]
            
            # 使用 np.pad 进行镜像填充
            patch = np.pad(
                valid_patch,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='reflect'
            )
        else:
            # 直接提取邻域切片 (需求 9.3)
            patch = self.image_cube[i_start:i_end, j_start:j_end, :]
        
        # 转换为 (Bands, S, S) 格式
        patch = np.transpose(patch, (2, 0, 1))  # (Bands, S, S)
        
        # 获取对应的标准参考光谱 (需求 9.4)
        # 根据模型设计文档，FRef 是作为“基准标尺”引入的理想成矿模式光谱。
        # 这里统一使用目标矿物（富矿伟晶岩，映射为类别 1）的标准光谱作为输入。
        target_class_idx = 1
        if target_class_idx in self.gsrsl:
            ref_spectrum = self.gsrsl[target_class_idx]  # (Bands,)
        else:
            # 兼容旧配置
            if 0 in self.gsrsl:
                ref_spectrum = self.gsrsl[0]
            else:
                ref_spectrum = np.zeros(bands, dtype=self.image_cube.dtype)
        
        # 转换为张量 (需求 9.5)
        x_sat = torch.from_numpy(patch).float().unsqueeze(0)  # (1, Bands, S, S)
        x_ref = torch.from_numpy(ref_spectrum).float().view(1, -1, 1, 1)  # (1, Bands, 1, 1)
        
        # 数据增强
        if self.augmenter is not None:
            x_sat, x_ref = self.augmenter(x_sat, x_ref)
        
        return x_sat, x_ref, label


class DataAugmentation:
    """
    数据增强类
    
    功能:
        1. 几何变换（旋转、翻转）
        2. 光谱噪声注入
    
    参数:
        noise_std (float): 高斯噪声标准差，默认 0.01
    
    需求: 7.1, 7.2, 7.3
    """
    
    def __init__(self, noise_std=0.01):
        """
        初始化数据增强器
        
        参数:
            noise_std (float): 高斯噪声标准差
        
        需求: 7.3
        """
        self.noise_std = noise_std
    
    def __call__(self, x_sat, x_ref):
        """
        应用数据增强
        
        参数:
            x_sat (Tensor): 卫星流 (1, Bands, S, S)
            x_ref (Tensor): 地面流 (1, Bands, 1, 1)
        
        返回:
            增强后的 x_sat, x_ref
        
        需求: 7.1, 7.2, 7.3, 7.4
        """
        # 几何变换（仅对卫星流） (需求 7.1, 7.2)
        x_sat = self._geometric_transform(x_sat)
        
        # 光谱噪声注入（仅对卫星流） (需求 7.3)
        x_sat = self._add_spectral_noise(x_sat)
        
        # 地面流保持不变（物理标准不应被扰动） (需求 7.4)
        
        return x_sat, x_ref
    
    def _geometric_transform(self, x):
        """
        随机旋转和翻转
        
        参数:
            x (Tensor): 输入张量 (1, Bands, S, S)
        
        返回:
            Tensor: 变换后的张量 (1, Bands, S, S)
        
        需求: 7.1, 7.2
        """
        # 随机旋转 (0, 90, 180, 270度) (需求 7.1)
        k = random.randint(0, 3)
        x = torch.rot90(x, k, dims=[2, 3])
        
        # 随机水平翻转 (需求 7.2)
        if random.random() > 0.5:
            x = torch.flip(x, dims=[3])
        
        # 随机垂直翻转 (需求 7.2)
        if random.random() > 0.5:
            x = torch.flip(x, dims=[2])
        
        return x
    
    def _add_spectral_noise(self, x):
        """
        添加高斯噪声
        
        参数:
            x (Tensor): 输入张量 (1, Bands, S, S)
        
        返回:
            Tensor: 添加噪声后的张量 (1, Bands, S, S)
        
        需求: 7.3
        """
        # 生成零均值、标准差为 noise_std 的高斯噪声
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise
        return x



def _load_raster_array(file_path: str, is_label: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.tif', '.tiff']:
        try:
            import rasterio
        except ImportError as e:
            raise ImportError(
                "读取 GeoTIFF 需要 rasterio，请先安装: pip install rasterio"
            ) from e
        with rasterio.open(file_path) as src:
            if is_label:
                if src.count < 1:
                    raise ValueError(f"标签 GeoTIFF 无可读波段: {file_path}")
                array = src.read(1)
            else:
                array = src.read()
                array = np.transpose(array, (1, 2, 0))
            metadata = {
                'path': os.path.abspath(file_path),
                'format': 'tif',
                'crs_obj': src.crs,
                'crs': src.crs.to_string() if src.crs is not None else None,
                'transform_obj': src.transform,
                'transform': tuple(src.transform),
                'height': int(src.height),
                'width': int(src.width),
                'count': int(src.count),
                'georef_available': src.crs is not None and src.transform is not None
            }
            return array, metadata
    array = np.load(file_path, allow_pickle=False)
    metadata = {
        'path': os.path.abspath(file_path),
        'format': 'npy',
        'crs_obj': None,
        'crs': None,
        'transform_obj': None,
        'transform': None,
        'height': int(array.shape[0]) if array.ndim >= 2 else None,
        'width': int(array.shape[1]) if array.ndim >= 2 else None,
        'count': int(array.shape[2]) if array.ndim == 3 else 1,
        'georef_available': False
    }
    return array, metadata


def _is_same_transform(transform_a, transform_b, atol: float = 1e-9) -> bool:
    if transform_a is None or transform_b is None:
        return False
    return bool(np.allclose(np.array(transform_a, dtype=np.float64), np.array(transform_b, dtype=np.float64), atol=atol))


def load_hyperspectral_data(image_path: str, gt_path: str, gsrsl_path: str, return_metadata: bool = False):
    """
    加载高光谱数据，包含文件路径验证
    
    参数:
        image_path (str): 高光谱影像文件路径
        gt_path (str): Ground Truth 标签文件路径
        gsrsl_path (str): GSRSL 标准光谱库文件路径
    
    返回:
        image_cube (ndarray): 高光谱影像立方体
        ground_truth (ndarray): 标签图
        gsrsl (dict): 地面标准参考光谱库
    
    抛出:
        FileNotFoundError: 当文件不存在时
    
    需求: 15.3
    """
    from utils.file_utils import validate_file_exists
    
    # 验证文件是否存在 (需求 15.3)
    validate_file_exists(image_path, "高光谱影像文件")
    validate_file_exists(gt_path, "Ground Truth 标签文件")
    validate_file_exists(gsrsl_path, "GSRSL 标准光谱库文件")
    
    try:
        image_cube, image_meta = _load_raster_array(image_path, is_label=False)
        ground_truth, gt_meta = _load_raster_array(gt_path, is_label=True)
        gsrsl = np.load(gsrsl_path, allow_pickle=True).item()
    except Exception as e:
        raise IOError(
            f"加载数据文件失败: {str(e)}\n"
            f"影像路径: {os.path.abspath(image_path)}\n"
            f"标签路径: {os.path.abspath(gt_path)}\n"
            f"光谱库路径: {os.path.abspath(gsrsl_path)}"
        ) from e
    if image_cube.ndim != 3:
        raise ValueError(f"影像应为三维数组 (H, W, Bands)，当前形状: {image_cube.shape}")
    if ground_truth.ndim == 3 and ground_truth.shape[2] == 1:
        ground_truth = np.squeeze(ground_truth, axis=2)
    if ground_truth.ndim != 2:
        raise ValueError(f"标签图应为二维数组 (H, W)，当前形状: {ground_truth.shape}")
    if image_cube.shape[:2] != ground_truth.shape[:2]:
        raise ValueError(f"影像和标签空间尺寸不一致: {image_cube.shape[:2]} vs {ground_truth.shape}")

    georef_check = {
        'enabled': bool(image_meta['format'] == 'tif' and gt_meta['format'] == 'tif'),
        'matched': False,
        'reason': ''
    }
    if georef_check['enabled']:
        if image_meta['crs'] != gt_meta['crs']:
            georef_check['reason'] = f"CRS 不一致: image={image_meta['crs']}, gt={gt_meta['crs']}"
            raise ValueError(f"影像与标签地理参考不一致: {georef_check['reason']}")
        if not _is_same_transform(image_meta['transform'], gt_meta['transform']):
            georef_check['reason'] = "Affine transform 不一致"
            raise ValueError(f"影像与标签地理参考不一致: {georef_check['reason']}")
        georef_check['matched'] = True
        georef_check['reason'] = 'image 与 gt 地理参考一致'

    metadata = {
        'image': image_meta,
        'ground_truth': gt_meta,
        'georef_check': georef_check
    }

    if return_metadata:
        return image_cube, ground_truth, gsrsl, metadata
    return image_cube, ground_truth, gsrsl


def save_geotiff(output_path: str, array: np.ndarray, reference_meta: Dict[str, Any], dtype: str, nodata=None):
    try:
        import rasterio
    except ImportError as e:
        raise ImportError(
            "写出 GeoTIFF 需要 rasterio，请先安装: pip install rasterio"
        ) from e

    transform_obj = reference_meta.get('transform_obj')
    crs_obj = reference_meta.get('crs_obj')
    if transform_obj is None or crs_obj is None:
        raise ValueError("缺少可用地理参考，无法写出 GeoTIFF")

    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"当前仅支持输出二维栅格，收到形状: {arr.shape}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=crs_obj,
        transform=transform_obj,
        nodata=nodata,
        compress='deflate'
    ) as dst:
        dst.write(arr.astype(dtype), 1)
