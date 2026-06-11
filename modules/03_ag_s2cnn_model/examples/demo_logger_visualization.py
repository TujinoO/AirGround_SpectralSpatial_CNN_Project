"""
日志和可视化功能演示

演示如何使用日志记录系统和可视化功能。

需求: 14.1, 14.2, 14.3, 14.4, 14.5, 14.7
"""

import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import create_logger
from utils.visualization import create_visualizer


def demo_logger():
    """演示日志记录功能"""
    print("\n" + "="*60)
    print("日志记录功能演示")
    print("="*60)
    
    # 创建日志记录器
    logger = create_logger(log_dir='logs', log_name='demo_training')
    
    # 记录配置信息
    config = {
        'num_bands': 290,
        'spatial_size': 13,
        'num_classes': 4,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 100
    }
    logger.log_config(config)
    
    # 记录设备信息
    logger.log_device_info('cuda:0', 'NVIDIA RTX 5060')
    
    # 记录模型信息
    logger.log_model_info(num_params=790000, model_size_mb=3.2)
    
    # 模拟训练过程
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        # 记录 epoch 开始
        logger.log_epoch_start(epoch, num_epochs)
        
        # 模拟训练指标
        train_metrics = {
            'loss': 0.8 - epoch * 0.1,
            'accuracy': 0.6 + epoch * 0.05
        }
        logger.log_training_metrics(epoch, train_metrics)
        
        # 模拟验证指标
        val_metrics = {
            'loss': 0.85 - epoch * 0.1,
            'OA': 0.58 + epoch * 0.05,
            'AA': 0.55 + epoch * 0.05,
            'Kappa': 0.50 + epoch * 0.05,
            'F1_macro': 0.56 + epoch * 0.05
        }
        logger.log_validation_metrics(epoch, val_metrics)
        
        # 记录学习率
        lr = 0.001 * (0.9 ** epoch)
        logger.log_learning_rate(epoch, lr)
        
        # 模拟保存最佳模型
        if epoch == 3:
            logger.log_model_save('checkpoints/best_model.pth', val_metrics)
    
    # 记录训练完成
    best_metrics = {
        'OA': 0.83,
        'AA': 0.80,
        'Kappa': 0.75,
        'F1_macro': 0.81
    }
    logger.log_training_complete(num_epochs, best_metrics)
    
    # 关闭日志记录器
    logger.close()
    
    print(f"\n日志文件已保存到: {logger.log_file}")


def demo_visualization():
    """演示可视化功能"""
    print("\n" + "="*60)
    print("可视化功能演示")
    print("="*60)
    
    # 创建可视化器
    visualizer = create_visualizer(save_dir='results', dpi=300)
    
    # 1. 绘制损失曲线
    print("\n1. 绘制损失曲线...")
    train_losses = [0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38, 0.35, 0.33, 0.32]
    val_losses = [0.85, 0.75, 0.65, 0.55, 0.50, 0.45, 0.43, 0.41, 0.40, 0.39]
    visualizer.plot_loss_curves(train_losses, val_losses)
    
    # 2. 绘制学习率曲线
    print("\n2. 绘制学习率曲线...")
    learning_rates = [0.001 * (0.95 ** i) for i in range(10)]
    visualizer.plot_learning_rate_curve(learning_rates)
    
    # 3. 绘制混淆矩阵
    print("\n3. 绘制混淆矩阵...")
    confusion_matrix = np.array([
        [85, 8, 5, 2],
        [6, 78, 10, 6],
        [4, 7, 82, 7],
        [3, 5, 8, 84]
    ])
    class_names = ['锂辉石伟晶岩', '贫矿伟晶岩', '围岩', '背景']
    visualizer.plot_confusion_matrix(
        confusion_matrix,
        class_names,
        title='AG-S²CNN 混淆矩阵'
    )
    
    # 4. 绘制归一化混淆矩阵
    print("\n4. 绘制归一化混淆矩阵...")
    visualizer.plot_confusion_matrix(
        confusion_matrix,
        class_names,
        save_name='confusion_matrix_normalized.png',
        title='AG-S²CNN 混淆矩阵（归一化）',
        normalize=True
    )
    
    # 5. 绘制指标对比图
    print("\n5. 绘制指标对比图...")
    metrics_dict = {
        'OA': [0.70, 0.75, 0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.85, 0.86],
        'AA': [0.68, 0.72, 0.75, 0.77, 0.79, 0.80, 0.81, 0.82, 0.82, 0.83],
        'Kappa': [0.60, 0.65, 0.68, 0.70, 0.72, 0.73, 0.74, 0.75, 0.75, 0.76],
        'F1_macro': [0.69, 0.73, 0.76, 0.78, 0.80, 0.81, 0.82, 0.83, 0.83, 0.84]
    }
    visualizer.plot_metrics_comparison(metrics_dict)
    
    # 6. 绘制类别性能图
    print("\n6. 绘制类别性能图...")
    precision = np.array([0.85, 0.78, 0.82, 0.88])
    recall = np.array([0.83, 0.75, 0.80, 0.86])
    f1_scores = np.array([0.84, 0.76, 0.81, 0.87])
    visualizer.plot_class_performance(
        precision,
        recall,
        f1_scores,
        class_names,
        title='AG-S²CNN 各类别性能指标'
    )
    
    # 关闭所有图形
    visualizer.close_all()
    
    print(f"\n所有图表已保存到: {visualizer.save_dir}")


def demo_integrated_usage():
    """演示日志和可视化的集成使用"""
    print("\n" + "="*60)
    print("集成使用演示")
    print("="*60)
    
    # 创建日志记录器和可视化器
    logger = create_logger(log_dir='logs', log_name='integrated_demo')
    visualizer = create_visualizer(save_dir='results')
    
    logger.info("开始模拟训练流程...")
    
    # 模拟训练数据收集
    train_losses = []
    val_losses = []
    learning_rates = []
    oa_scores = []
    
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        # 模拟训练
        train_loss = 0.8 - epoch * 0.05
        val_loss = 0.85 - epoch * 0.05
        lr = 0.001 * (0.95 ** epoch)
        oa = 0.6 + epoch * 0.025
        
        # 记录数据
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        learning_rates.append(lr)
        oa_scores.append(oa)
        
        # 记录日志
        logger.log_epoch_start(epoch, num_epochs)
        logger.log_training_metrics(epoch, {'loss': train_loss})
        logger.log_validation_metrics(epoch, {'loss': val_loss, 'OA': oa})
        logger.log_learning_rate(epoch, lr)
    
    # 训练结束后生成可视化
    logger.info("生成可视化图表...")
    
    visualizer.plot_loss_curves(train_losses, val_losses, save_name='integrated_loss.png')
    visualizer.plot_learning_rate_curve(learning_rates, save_name='integrated_lr.png')
    
    # 生成最终混淆矩阵
    cm = np.array([
        [85, 8, 5, 2],
        [6, 78, 10, 6],
        [4, 7, 82, 7],
        [3, 5, 8, 84]
    ])
    class_names = ['锂辉石伟晶岩', '贫矿伟晶岩', '围岩', '背景']
    visualizer.plot_confusion_matrix(cm, class_names, save_name='integrated_cm.png')
    
    # 记录完成信息
    best_metrics = {'OA': max(oa_scores), 'final_loss': train_losses[-1]}
    logger.log_training_complete(num_epochs, best_metrics)
    
    # 清理
    logger.close()
    visualizer.close_all()
    
    print(f"\n日志文件: {logger.log_file}")
    print(f"图表目录: {visualizer.save_dir}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("AG-S²CNN 日志和可视化功能演示")
    print("="*60)
    
    # 创建输出目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 运行演示
    demo_logger()
    demo_visualization()
    demo_integrated_usage()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print("\n请查看以下目录:")
    print("  - logs/     : 日志文件")
    print("  - results/  : 可视化图表")
    print()


if __name__ == '__main__':
    main()
