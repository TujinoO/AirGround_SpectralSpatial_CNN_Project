"""
日志和可视化模块测试

测试日志记录系统和可视化功能的基本功能。
"""

import os
import pytest
import numpy as np
import tempfile
import shutil
from utils.logger import TrainingLogger, create_logger
from utils.visualization import TrainingVisualizer, create_visualizer


class TestTrainingLogger:
    """测试训练日志记录器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_logger_initialization(self):
        """测试日志记录器初始化"""
        logger = TrainingLogger(log_dir=self.test_dir, log_name='test_log')
        
        # 验证日志文件是否创建
        assert os.path.exists(logger.log_file)
        assert logger.log_file.endswith('test_log.log')
        
        logger.close()
    
    def test_logger_basic_logging(self):
        """测试基本日志记录功能"""
        logger = TrainingLogger(log_dir=self.test_dir, log_name='test_log')
        
        # 测试不同级别的日志
        logger.info("这是一条 INFO 日志")
        logger.debug("这是一条 DEBUG 日志")
        logger.warning("这是一条 WARNING 日志")
        logger.error("这是一条 ERROR 日志")
        
        # 验证日志文件存在且有内容
        assert os.path.exists(logger.log_file)
        with open(logger.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'INFO' in content
            assert 'WARNING' in content
            assert 'ERROR' in content
        
        logger.close()
    
    def test_logger_epoch_logging(self):
        """测试 epoch 日志记录"""
        logger = TrainingLogger(log_dir=self.test_dir, log_name='test_log')
        
        # 记录 epoch 开始
        logger.log_epoch_start(1, 10)
        
        # 记录训练指标
        train_metrics = {'loss': 0.5, 'accuracy': 0.85}
        logger.log_training_metrics(1, train_metrics)
        
        # 记录验证指标
        val_metrics = {'loss': 0.6, 'OA': 0.82, 'Kappa': 0.75}
        logger.log_validation_metrics(1, val_metrics)
        
        # 记录学习率
        logger.log_learning_rate(1, 0.001)
        
        # 验证日志文件内容
        with open(logger.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'Epoch 1/10' in content
            assert 'loss: 0.5' in content or 'loss: 0.500000' in content
            assert 'accuracy: 0.85' in content or 'accuracy: 0.850000' in content
        
        logger.close()
    
    def test_logger_model_save(self):
        """测试模型保存日志"""
        logger = TrainingLogger(log_dir=self.test_dir, log_name='test_log')
        
        logger.log_model_save('model.pth', {'OA': 0.9})
        
        with open(logger.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'model.pth' in content
        
        logger.close()
    
    def test_create_logger_function(self):
        """测试便捷创建函数"""
        logger = create_logger(log_dir=self.test_dir, log_name='test_log')
        
        assert isinstance(logger, TrainingLogger)
        assert os.path.exists(logger.log_file)
        
        logger.close()


class TestTrainingVisualizer:
    """测试训练可视化器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_visualizer_initialization(self):
        """测试可视化器初始化"""
        visualizer = TrainingVisualizer(save_dir=self.test_dir)
        
        # 验证保存目录是否创建
        assert os.path.exists(self.test_dir)
        assert visualizer.save_dir == self.test_dir
        
        visualizer.close_all()
    
    def test_plot_loss_curves(self):
        """测试损失曲线绘制"""
        visualizer = TrainingVisualizer(save_dir=self.test_dir)
        
        # 生成模拟数据
        train_losses = [0.8, 0.6, 0.5, 0.4, 0.35]
        val_losses = [0.85, 0.65, 0.55, 0.45, 0.4]
        
        # 绘制损失曲线
        fig = visualizer.plot_loss_curves(train_losses, val_losses)
        
        # 验证图像文件是否保存
        save_path = os.path.join(self.test_dir, 'loss_curves.png')
        assert os.path.exists(save_path)
        
        visualizer.close_all()
    
    def test_plot_learning_rate_curve(self):
        """测试学习率曲线绘制"""
        visualizer = TrainingVisualizer(save_dir=self.test_dir)
        
        # 生成模拟数据
        learning_rates = [0.001, 0.0008, 0.0006, 0.0004, 0.0002]
        
        # 绘制学习率曲线
        fig = visualizer.plot_learning_rate_curve(learning_rates)
        
        # 验证图像文件是否保存
        save_path = os.path.join(self.test_dir, 'learning_rate_curve.png')
        assert os.path.exists(save_path)
        
        visualizer.close_all()
    
    def test_plot_confusion_matrix(self):
        """测试混淆矩阵绘制"""
        visualizer = TrainingVisualizer(save_dir=self.test_dir)
        
        # 生成模拟混淆矩阵
        cm = np.array([
            [50, 5, 3, 2],
            [4, 45, 6, 5],
            [2, 3, 48, 7],
            [1, 2, 4, 53]
        ])
        
        class_names = ['类别0', '类别1', '类别2', '类别3']
        
        # 绘制混淆矩阵
        fig = visualizer.plot_confusion_matrix(cm, class_names)
        
        # 验证图像文件是否保存
        save_path = os.path.join(self.test_dir, 'confusion_matrix.png')
        assert os.path.exists(save_path)
        
        visualizer.close_all()
    
    def test_plot_confusion_matrix_normalized(self):
        """测试归一化混淆矩阵绘制"""
        visualizer = TrainingVisualizer(save_dir=self.test_dir)
        
        # 生成模拟混淆矩阵
        cm = np.array([
            [50, 5, 3, 2],
            [4, 45, 6, 5],
            [2, 3, 48, 7],
            [1, 2, 4, 53]
        ])
        
        # 绘制归一化混淆矩阵
        fig = visualizer.plot_confusion_matrix(
            cm,
            save_name='confusion_matrix_normalized.png',
            normalize=True
        )
        
        # 验证图像文件是否保存
        save_path = os.path.join(self.test_dir, 'confusion_matrix_normalized.png')
        assert os.path.exists(save_path)
        
        visualizer.close_all()
    
    def test_plot_metrics_comparison(self):
        """测试指标对比图绘制"""
        visualizer = TrainingVisualizer(save_dir=self.test_dir)
        
        # 生成模拟数据
        metrics_dict = {
            'OA': [0.7, 0.75, 0.8, 0.82, 0.85],
            'AA': [0.68, 0.73, 0.78, 0.8, 0.83],
            'Kappa': [0.6, 0.65, 0.7, 0.72, 0.75]
        }
        
        # 绘制指标对比图
        fig = visualizer.plot_metrics_comparison(metrics_dict)
        
        # 验证图像文件是否保存
        save_path = os.path.join(self.test_dir, 'metrics_comparison.png')
        assert os.path.exists(save_path)
        
        visualizer.close_all()
    
    def test_plot_class_performance(self):
        """测试类别性能图绘制"""
        visualizer = TrainingVisualizer(save_dir=self.test_dir)
        
        # 生成模拟数据
        precision = np.array([0.85, 0.78, 0.82, 0.88])
        recall = np.array([0.83, 0.75, 0.80, 0.86])
        f1_scores = np.array([0.84, 0.76, 0.81, 0.87])
        class_names = ['类别0', '类别1', '类别2', '类别3']
        
        # 绘制类别性能图
        fig = visualizer.plot_class_performance(
            precision, recall, f1_scores, class_names
        )
        
        # 验证图像文件是否保存
        save_path = os.path.join(self.test_dir, 'class_performance.png')
        assert os.path.exists(save_path)
        
        visualizer.close_all()
    
    def test_create_visualizer_function(self):
        """测试便捷创建函数"""
        visualizer = create_visualizer(save_dir=self.test_dir)
        
        assert isinstance(visualizer, TrainingVisualizer)
        assert visualizer.save_dir == self.test_dir
        
        visualizer.close_all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
