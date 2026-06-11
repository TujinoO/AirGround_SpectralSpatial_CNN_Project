"""
文件路径错误处理工具模块

提供文件路径验证和错误处理功能
"""

import os
import numpy as np

try:
    from osgeo import gdal
except ImportError:
    gdal = None


def save_geotiff(array: np.ndarray, output_path: str, reference_tiff_path: str) -> None:
    """
    保存数组为带有地理坐标信息的 GeoTIFF 文件
    
    参数:
        array (np.ndarray): 要保存的 2D 数据数组 (H, W)
        output_path (str): 输出的 TIFF 文件路径
        reference_tiff_path (str): 包含原始地理坐标信息的参考影像路径
    """
    if gdal is None:
        raise ImportError("未安装 GDAL 库，无法保存带地理坐标的 TIFF。请使用 pip install gdal 安装。")
        
    # 打开参考影像，获取地理变换和投影信息
    ref_ds = gdal.Open(reference_tiff_path, gdal.GA_ReadOnly)
    if ref_ds is None:
        raise FileNotFoundError(f"无法打开参考影像以获取地理信息: {reference_tiff_path}")
        
    geo_transform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    ref_ds = None
    
    # 确定数据类型
    if array.dtype == np.float32:
        gdal_dtype = gdal.GDT_Float32
    elif array.dtype == np.int32:
        gdal_dtype = gdal.GDT_Int32
    elif array.dtype == np.float64:
        gdal_dtype = gdal.GDT_Float64
    else:
        # 默认回退
        array = array.astype(np.float32)
        gdal_dtype = gdal.GDT_Float32

    # 创建新的 TIFF 文件
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, array.shape[1], array.shape[0], 1, gdal_dtype)
    
    # 写入地理信息
    if geo_transform:
        out_ds.SetGeoTransform(geo_transform)
    if projection:
        out_ds.SetProjection(projection)
        
    # 写入数据
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.FlushCache()
    
    # 清理引用
    out_ds = None


def validate_file_exists(filepath: str, file_description: str = "文件") -> None:
    """
    验证文件是否存在
    
    参数:
        filepath (str): 文件路径
        file_description (str): 文件描述，用于错误信息
    
    抛出:
        FileNotFoundError: 当文件不存在时
    
    需求: 15.3
    """
    if not os.path.exists(filepath):
        abs_path = os.path.abspath(filepath)
        raise FileNotFoundError(
            f"{file_description}不存在: {abs_path}\n"
            f"请检查路径是否正确，或文件是否已被删除。"
        )


def validate_directory_exists(dirpath: str, dir_description: str = "目录") -> None:
    """
    验证目录是否存在
    
    参数:
        dirpath (str): 目录路径
        dir_description (str): 目录描述，用于错误信息
    
    抛出:
        FileNotFoundError: 当目录不存在时
    
    需求: 15.3
    """
    if not os.path.exists(dirpath):
        abs_path = os.path.abspath(dirpath)
        raise FileNotFoundError(
            f"{dir_description}不存在: {abs_path}\n"
            f"请检查路径是否正确，或使用 os.makedirs() 创建目录。"
        )
    
    if not os.path.isdir(dirpath):
        abs_path = os.path.abspath(dirpath)
        raise NotADirectoryError(
            f"路径不是目录: {abs_path}\n"
            f"该路径指向一个文件，而不是目录。"
        )


def validate_file_extension(filepath: str, expected_extensions: list, 
                            file_description: str = "文件") -> None:
    """
    验证文件扩展名
    
    参数:
        filepath (str): 文件路径
        expected_extensions (list): 期望的扩展名列表（例如 ['.yaml', '.yml']）
        file_description (str): 文件描述，用于错误信息
    
    抛出:
        ValueError: 当文件扩展名不匹配时
    
    需求: 15.3
    """
    _, ext = os.path.splitext(filepath)
    if ext.lower() not in [e.lower() for e in expected_extensions]:
        raise ValueError(
            f"{file_description}扩展名错误: {ext}\n"
            f"期望的扩展名: {', '.join(expected_extensions)}\n"
            f"文件路径: {os.path.abspath(filepath)}"
        )


def ensure_directory_exists(dirpath: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    参数:
        dirpath (str): 目录路径
    
    需求: 15.3
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
        print(f"已创建目录: {os.path.abspath(dirpath)}")
