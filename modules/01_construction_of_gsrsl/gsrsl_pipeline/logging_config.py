"""
Logging configuration for GSRSL pipeline
GSRSL管道的日志配置
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_file: str = "gsrsl_pipeline.log", level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for the GSRSL pipeline.
    为GSRSL管道配置日志
    
    Args:
        log_file: Path to log file (default: gsrsl_pipeline.log)
                 日志文件路径（默认：gsrsl_pipeline.log）
        level: Logging level (default: logging.INFO)
              日志级别（默认：logging.INFO）
    
    Returns:
        Configured logger instance
        配置的日志记录器实例
    """
    # Create logger
    logger = logging.getLogger('gsrsl_pipeline')
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - simpler logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = 'gsrsl_pipeline') -> logging.Logger:
    """
    Get a logger instance.
    获取日志记录器实例
    
    Args:
        name: Logger name (default: gsrsl_pipeline)
             日志记录器名称（默认：gsrsl_pipeline）
    
    Returns:
        Logger instance
        日志记录器实例
    """
    return logging.getLogger(name)
