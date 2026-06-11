"""
Logging configuration module.
日志配置模块。
"""

import logging
import sys
from typing import Optional, Union


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Set up logging configuration for the application.
    为应用程序设置日志配置。
    
    Args:
        level: Logging level (default: INFO). Can be int or string ('DEBUG', 'INFO', etc.)
              日志级别 (默认: INFO)。可以是整数或字符串
        log_file: Optional path to log file. If None, logs to console only.
                 可选的日志文件路径。如果为None，仅记录到控制台。
        format_string: Optional custom format string. If None, uses default format.
                      可选的自定义格式字符串。如果为None，使用默认格式。
    """
    # Convert string level to int if needed
    # 如果需要，将字符串级别转换为整数
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if format_string is None:
        format_string = (
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        )
    
    # Create formatter
    # 创建格式化器
    formatter = logging.Formatter(
        format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure root logger
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    # 移除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    # 如果指定，添加文件处理器
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set logging level for third-party libraries to WARNING
    # 将第三方库的日志级别设置为WARNING
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('rasterio').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    获取具有指定名称的日志器实例。
    
    Args:
        name: Logger name (typically __name__ of the module)
             日志器名称 (通常是模块的__name__)
             
    Returns:
        Logger instance (日志器实例)
    """
    return logging.getLogger(name)
