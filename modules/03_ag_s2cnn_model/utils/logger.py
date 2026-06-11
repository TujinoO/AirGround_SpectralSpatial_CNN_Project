"""
日志记录系统

实现训练过程的日志记录功能，包括控制台输出和文件记录。

需求: 14.1, 14.2, 14.7
"""

import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime


class TrainingLogger:
    """
    训练日志记录器
    
    功能:
        1. 配置 Python logging 模块
        2. 同时输出到控制台和文件
        3. 记录训练损失、验证损失和评估指标
    
    参数:
        log_dir (str): 日志文件保存目录
        log_name (str): 日志文件名（不含扩展名）
        console_level (int): 控制台日志级别
        file_level (int): 文件日志级别
    
    需求: 14.1, 14.2, 14.7
    """
    
    def __init__(
        self,
        log_dir: str = 'logs',
        log_name: Optional[str] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        """
        初始化日志记录器
        
        参数:
            log_dir: 日志文件保存目录
            log_name: 日志文件名（不含扩展名），默认使用时间戳
            console_level: 控制台日志级别
            file_level: 文件日志级别
        """
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名
        if log_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_name = f'training_{timestamp}'
        
        self.log_file = os.path.join(log_dir, f'{log_name}.log')
        
        # 创建 logger
        self.logger = logging.getLogger('AG_S2CNN_Training')
        self.logger.setLevel(logging.DEBUG)
        
        # 清除已有的 handlers（避免重复添加）
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 创建格式化器
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 记录日志系统初始化信息
        self.logger.info("="*60)
        self.logger.info("日志系统初始化完成")
        self.logger.info(f"日志文件: {self.log_file}")
        self.logger.info("="*60)
    
    def info(self, message: str):
        """记录 INFO 级别日志"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """记录 DEBUG 级别日志"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """记录 WARNING 级别日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录 ERROR 级别日志"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """记录 CRITICAL 级别日志"""
        self.logger.critical(message)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """
        记录 epoch 开始信息
        
        参数:
            epoch: 当前 epoch 数
            total_epochs: 总 epoch 数
        """
        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info(f"Epoch {epoch}/{total_epochs}")
        self.logger.info("="*60)
    
    def log_training_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        记录训练指标
        
        参数:
            epoch: 当前 epoch 数
            metrics: 训练指标字典
        """
        self.logger.info(f"[Epoch {epoch}] 训练指标:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_validation_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        记录验证指标
        
        参数:
            epoch: 当前 epoch 数
            metrics: 验证指标字典
        """
        self.logger.info(f"[Epoch {epoch}] 验证指标:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_learning_rate(self, epoch: int, lr: float):
        """
        记录学习率
        
        参数:
            epoch: 当前 epoch 数
            lr: 当前学习率
        """
        self.logger.info(f"[Epoch {epoch}] 学习率: {lr:.8f}")
    
    def log_model_save(self, filepath: str, metrics: Optional[Dict[str, float]] = None):
        """
        记录模型保存信息
        
        参数:
            filepath: 模型保存路径
            metrics: 模型对应的指标（可选）
        """
        self.logger.info(f"模型已保存: {filepath}")
        if metrics:
            self.logger.info(f"  指标: {metrics}")
    
    def log_early_stopping(self, epoch: int, patience: int, best_score: float):
        """
        记录早停信息
        
        参数:
            epoch: 触发早停的 epoch 数
            patience: 早停容忍度
            best_score: 最佳分数
        """
        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info(f"早停触发 (Epoch {epoch})")
        self.logger.info(f"  容忍度: {patience} epochs")
        self.logger.info(f"  最佳分数: {best_score:.6f}")
        self.logger.info("="*60)
    
    def log_training_complete(self, total_epochs: int, best_metrics: Dict[str, float]):
        """
        记录训练完成信息
        
        参数:
            total_epochs: 总训练 epoch 数
            best_metrics: 最佳指标
        """
        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info("训练完成")
        self.logger.info(f"  总 Epochs: {total_epochs}")
        self.logger.info("  最佳指标:")
        for key, value in best_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"    {key}: {value:.6f}")
            else:
                self.logger.info(f"    {key}: {value}")
        self.logger.info("="*60)
    
    def log_config(self, config: Dict[str, Any]):
        """
        记录配置信息
        
        参数:
            config: 配置字典
        """
        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info("配置信息")
        self.logger.info("="*60)
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("="*60)
    
    def log_device_info(self, device: str, device_name: Optional[str] = None):
        """
        记录设备信息
        
        参数:
            device: 设备类型 (cuda/cpu)
            device_name: 设备名称（可选）
        """
        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info("设备信息")
        self.logger.info("="*60)
        self.logger.info(f"  设备类型: {device}")
        if device_name:
            self.logger.info(f"  设备名称: {device_name}")
        self.logger.info("="*60)
    
    def log_model_info(self, num_params: int, model_size_mb: float):
        """
        记录模型信息
        
        参数:
            num_params: 模型参数量
            model_size_mb: 模型大小（MB）
        """
        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info("模型信息")
        self.logger.info("="*60)
        self.logger.info(f"  参数量: {num_params:,} ({num_params/1e6:.2f}M)")
        self.logger.info(f"  模型大小: {model_size_mb:.2f} MB")
        self.logger.info("="*60)
    
    def log_exception(self, exception: Exception):
        """
        记录异常信息
        
        参数:
            exception: 异常对象
        """
        self.logger.error("="*60)
        self.logger.error("发生异常")
        self.logger.error("="*60)
        self.logger.error(f"异常类型: {type(exception).__name__}")
        self.logger.error(f"异常信息: {str(exception)}")
        self.logger.error("="*60)
        # 记录完整的堆栈跟踪
        import traceback
        self.logger.error(traceback.format_exc())
    
    def close(self):
        """关闭日志记录器"""
        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info("日志记录器关闭")
        self.logger.info("="*60)
        
        # 关闭所有 handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def create_logger(
    log_dir: str = 'logs',
    log_name: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> TrainingLogger:
    """
    创建训练日志记录器的便捷函数
    
    参数:
        log_dir: 日志文件保存目录
        log_name: 日志文件名（不含扩展名）
        console_level: 控制台日志级别
        file_level: 文件日志级别
    
    返回:
        TrainingLogger: 日志记录器实例
    """
    return TrainingLogger(
        log_dir=log_dir,
        log_name=log_name,
        console_level=console_level,
        file_level=file_level
    )
