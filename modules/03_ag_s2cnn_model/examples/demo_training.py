"""
训练示例脚本

演示如何使用 AG-S²CNN 模型进行完整的训练流程。

使用方法:
    python examples/demo_training.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ag_s2cnn import AG_S2CNN
from utils.dataset import HyperspectralDataset
from utils.loss import WeightedCrossEntropyLoss
from utils.metrics import MetricsCalculator
from utils.logger import setup_logger
from utils.visualization import plot_training_curves, plot_confusion_matrix
from config import Config


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=20, mode='max', delta=0.0):
        """
        参数:
            patience (int): 容忍的 epoch 数
            mode (str): 'min' 或 'max'
            delta (float): 最小改善量
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric, model, save_path):
        """
        检查是否应该早停
        
        参数:
            metric (float): 当前指标值
            model (nn.Module): 模型
            save_path (str): 模型保存路径
        
        返回:
            bool: 是否应该停止训练
        """
        score = metric if self.mode == 'max' else -metric
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"早停计数: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, save_path)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, model, save_path):
        """保存模型检查点"""
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存到 {save_path}")


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    训练一个 epoch
    
    参数:
        model: 模型
        dataloader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch 数
    
    返回:
        avg_loss: 平均损失
        metrics: 训练指标
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [训练]")
    for batch_idx, (x_sat, x_ref, labels) in enumerate(pbar):
        # 移动数据到设备
        x_sat = x_sat.to(device)
        x_ref = x_ref.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(x_sat, x_ref)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 记录
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return avg_loss, all_preds, all_labels


def validate(model, dataloader, criterion, device, metrics_calculator):
    """
    验证模型
    
    参数:
        model: 模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        metrics_calculator: 指标计算器
    
    返回:
        avg_loss: 平均损失
        metrics: 验证指标
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="验证")
        for x_sat, x_ref, labels in pbar:
            # 移动数据到设备
            x_sat = x_sat.to(device)
            x_ref = x_ref.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(x_sat, x_ref)
            loss = criterion(outputs, labels)
            
            # 记录
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = metrics_calculator.calculate_all_metrics(all_labels, all_preds)
    
    return avg_loss, metrics


def train_model(config_path='config.yaml'):
    """
    完整的训练流程
    
    参数:
        config_path (str): 配置文件路径
    """
    # 加载配置
    print("=" * 60)
    print("加载配置...")
    config = Config.from_yaml(config_path)
    config.validate()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = config['data']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger('training', os.path.join(output_dir, 'training.log'))
    logger.info("开始训练流程")
    
    # 加载数据（这里使用模拟数据作为示例）
    print("\n" + "=" * 60)
    print("准备数据...")
    
    # 注意: 在实际使用中，应该从文件加载真实数据
    # image_cube = np.load(config['data']['image_path'])
    # ground_truth = np.load(config['data']['ground_truth_path'])
    # gsrsl = np.load(config['data']['gsrsl_path'], allow_pickle=True).item()
    
    # 这里使用模拟数据作为示例
    print("警告: 使用模拟数据进行演示")
    H, W = 100, 100
    num_bands = config['model']['num_bands']
    image_cube = np.random.randn(H, W, num_bands).astype(np.float32)
    ground_truth = np.random.randint(0, config['model']['num_classes'], size=(H, W))
    
    # 创建模拟的 GSRSL
    gsrsl = {
        i: np.random.randn(num_bands).astype(np.float32)
        for i in range(config['model']['num_classes'])
    }
    
    # 创建数据集
    dataset = HyperspectralDataset(
        image_cube=image_cube,
        ground_truth=ground_truth,
        gsrsl=gsrsl,
        spatial_size=config['model']['spatial_size'],
        augmentation=config['augmentation']['enabled']
    )
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0  # Windows 上设置为 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # 创建模型
    print("\n" + "=" * 60)
    print("创建模型...")
    model = AG_S2CNN(
        num_bands=config['model']['num_bands'],
        spatial_size=config['model']['spatial_size'],
        num_classes=config['model']['num_classes']
    )
    model.to(device)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 创建损失函数
    print("\n" + "=" * 60)
    print("创建损失函数...")
    
    # 计算类别权重
    train_labels = []
    for _, _, label in train_dataset:
        train_labels.append(label)
    train_labels = np.array(train_labels)
    
    class_weights = WeightedCrossEntropyLoss.calculate_class_weights(
        train_labels,
        config['model']['num_classes']
    )
    print(f"类别权重: {class_weights}")
    
    criterion = WeightedCrossEntropyLoss(class_weights=class_weights.to(device))
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # 创建指标计算器
    metrics_calculator = MetricsCalculator(num_classes=config['model']['num_classes'])
    
    # 创建早停机制
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        mode='max'
    )
    
    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    
    best_val_acc = 0.0
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        print("-" * 60)
        
        # 训练
        train_loss, train_preds, train_labels = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        train_losses.append(train_loss)
        
        # 验证
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, metrics_calculator
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_metrics['OA'])
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 打印结果
        print(f"\n训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        print(f"验证精度 (OA): {val_metrics['OA']:.4f}")
        print(f"验证 Kappa: {val_metrics['Kappa']:.4f}")
        print(f"学习率: {current_lr:.6f}")
        
        # 记录日志
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                   f"Val OA={val_metrics['OA']:.4f}, Val Kappa={val_metrics['Kappa']:.4f}")
        
        # 更新学习率
        scheduler.step()
        
        # 早停检查
        if early_stopping(val_metrics['OA'], model, os.path.join(output_dir, 'best_model.pth')):
            print("\n早停触发，停止训练")
            logger.info("早停触发，停止训练")
            break
        
        # 更新最佳精度
        if val_metrics['OA'] > best_val_acc:
            best_val_acc = val_metrics['OA']
    
    # 训练完成
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证精度: {best_val_acc:.4f}")
    print("=" * 60)
    
    # 生成可视化
    print("\n生成可视化...")
    
    # 损失曲线
    plot_training_curves(
        train_losses,
        val_losses,
        save_path=os.path.join(output_dir, 'loss_curves.png')
    )
    
    # 学习率曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
    plt.close()
    
    # 加载最佳模型并生成混淆矩阵
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    val_loss, val_metrics = validate(model, val_loader, criterion, device, metrics_calculator)
    
    plot_confusion_matrix(
        val_metrics['confusion_matrix'],
        class_names=[f'Class {i}' for i in range(config['model']['num_classes'])],
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # 打印最终指标
    print("\n最终验证指标:")
    metrics_calculator.print_metrics(
        val_metrics,
        class_names=[f'Class {i}' for i in range(config['model']['num_classes'])]
    )
    
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == '__main__':
    # 运行训练
    train_model('config.yaml')
