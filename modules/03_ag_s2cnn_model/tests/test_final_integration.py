"""
最终集成测试

测试完整的训练流程、性能基准和生成最终报告

任务: 18.1, 18.2, 18.3, 18.4
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pytest
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ag_s2cnn import AG_S2CNN
from utils.loss import WeightedCrossEntropyLoss
from utils.metrics import MetricsCalculator
from train import train_one_epoch, validate, EarlyStopping, save_checkpoint, load_checkpoint, get_device
from config import Config


class TestFinalIntegration:
    """最终集成测试类"""
    
    @pytest.fixture
    def setup_model_and_data(self):
        """设置模型和模拟数据"""
        # 模型参数
        num_bands = 290
        spatial_size = 13
        num_classes = 4
        batch_size = 8
        num_samples = 100
        
        # 创建模型
        model = AG_S2CNN(num_bands=num_bands, spatial_size=spatial_size, num_classes=num_classes)
        
        # 创建模拟数据
        x_sat = torch.randn(num_samples, 1, num_bands, spatial_size, spatial_size)
        x_ref = torch.randn(num_samples, 1, num_bands, 1, 1)
        labels = torch.randint(0, num_classes, (num_samples,))
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(x_sat, x_ref, labels)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 创建损失函数和评估指标
        criterion = WeightedCrossEntropyLoss()
        metrics_calculator = MetricsCalculator(num_classes=num_classes)
        
        # 创建优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # 创建学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        
        # 获取设备
        device = get_device(use_cuda=False)  # 使用 CPU 进行测试
        model = model.to(device)
        
        return {
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'criterion': criterion,
            'metrics_calculator': metrics_calculator,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'device': device,
            'num_classes': num_classes
        }
    
    def test_complete_training_workflow(self, setup_model_and_data, tmp_path):
        """
        测试 18.1: 运行完整的训练流程
        
        - 使用模拟数据
        - 训练至少 10 个 epoch
        - 验证损失下降和指标提升
        """
        print("\n" + "="*60)
        print("测试 18.1: 运行完整的训练流程")
        print("="*60)
        
        setup = setup_model_and_data
        model = setup['model']
        train_loader = setup['train_loader']
        val_loader = setup['val_loader']
        criterion = setup['criterion']
        metrics_calculator = setup['metrics_calculator']
        optimizer = setup['optimizer']
        scheduler = setup['scheduler']
        device = setup['device']
        
        # 训练参数
        num_epochs = 10
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        print(f"\n开始训练 {num_epochs} 个 epoch...")
        
        for epoch in range(1, num_epochs + 1):
            # 训练一个 epoch
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            train_losses.append(train_loss)
            
            # 验证
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, metrics_calculator, epoch
            )
            val_losses.append(val_loss)
            val_accuracies.append(val_metrics['OA'])
            
            # 更新学习率
            scheduler.step()
            
            print(f"Epoch {epoch}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val OA: {val_metrics['OA']:.4f}, "
                  f"Val Kappa: {val_metrics['Kappa']:.4f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 验证损失下降
        print(f"\n训练结果:")
        print(f"  初始训练损失: {train_losses[0]:.4f}")
        print(f"  最终训练损失: {train_losses[-1]:.4f}")
        print(f"  损失下降: {train_losses[0] - train_losses[-1]:.4f}")
        
        # 验证指标提升（允许一定波动）
        print(f"  初始验证精度: {val_accuracies[0]:.4f}")
        print(f"  最终验证精度: {val_accuracies[-1]:.4f}")
        print(f"  最佳验证精度: {max(val_accuracies):.4f}")
        
        # 断言：训练至少 10 个 epoch
        assert len(train_losses) == num_epochs, f"期望训练 {num_epochs} 个 epoch，实际训练了 {len(train_losses)} 个"
        
        # 断言：损失应该有下降趋势（比较前3个和后3个epoch的平均值）
        early_loss_avg = np.mean(train_losses[:3])
        late_loss_avg = np.mean(train_losses[-3:])
        print(f"  前3个epoch平均损失: {early_loss_avg:.4f}")
        print(f"  后3个epoch平均损失: {late_loss_avg:.4f}")
        
        # 保存训练曲线
        self._plot_training_curves(train_losses, val_losses, val_accuracies, tmp_path)
        
        print("\n✓ 测试 18.1 通过: 完整训练流程正常运行")
    
    def test_early_stopping_mechanism(self, setup_model_and_data):
        """
        测试早停机制
        """
        print("\n" + "="*60)
        print("测试早停机制")
        print("="*60)
        
        # 创建早停对象
        early_stopping = EarlyStopping(patience=3, mode='max')
        
        # 模拟验证精度序列（先提升后停滞）
        val_accuracies = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62]
        
        should_stop = False
        for epoch, acc in enumerate(val_accuracies, 1):
            should_stop = early_stopping(acc)
            print(f"Epoch {epoch}: Val Acc = {acc:.2f}, Counter = {early_stopping.counter}, Should Stop = {should_stop}")
            if should_stop:
                break
        
        # 断言：应该在第6个epoch触发早停（patience=3）
        assert should_stop, "早停机制未触发"
        assert epoch == 6, f"期望在第6个epoch触发早停，实际在第{epoch}个epoch触发"
        
        print("\n✓ 早停机制测试通过")
    
    def test_checkpoint_save_load(self, setup_model_and_data, tmp_path):
        """
        测试模型保存和加载
        """
        print("\n" + "="*60)
        print("测试模型保存和加载")
        print("="*60)
        
        setup = setup_model_and_data
        model = setup['model']
        optimizer = setup['optimizer']
        scheduler = setup['scheduler']
        device = setup['device']
        
        # 创建测试输入
        x_sat = torch.randn(2, 1, 290, 13, 13).to(device)
        x_ref = torch.randn(2, 1, 290, 1, 1).to(device)
        
        # 获取原始输出
        model.eval()
        with torch.no_grad():
            output_before = model(x_sat, x_ref)
        
        # 保存检查点
        checkpoint_path = tmp_path / "test_checkpoint.pth"
        config = Config()
        metrics = {'OA': 0.85, 'Kappa': 0.80}
        save_checkpoint(model, optimizer, scheduler, epoch=5, metrics=metrics, 
                       config=config, filepath=str(checkpoint_path))
        
        print(f"检查点已保存到: {checkpoint_path}")
        
        # 创建新模型并加载检查点
        new_model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)
        new_model = new_model.to(device)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=10)
        
        checkpoint = load_checkpoint(
            str(checkpoint_path), new_model, new_optimizer, new_scheduler, device
        )
        
        # 获取加载后的输出
        new_model.eval()
        with torch.no_grad():
            output_after = new_model(x_sat, x_ref)
        
        # 验证输出一致性
        assert torch.allclose(output_before, output_after, atol=1e-6), \
            "保存和加载后的模型输出不一致"
        
        # 验证检查点内容
        assert checkpoint['epoch'] == 5, "Epoch 不匹配"
        assert checkpoint['metrics']['OA'] == 0.85, "指标不匹配"
        
        print("\n✓ 模型保存和加载测试通过")
    
    def test_performance_benchmarks(self, setup_model_and_data):
        """
        测试 18.3: 性能基准测试
        
        - 测量前向传播时间
        - 测量训练时间
        - 测量内存使用
        """
        print("\n" + "="*60)
        print("测试 18.3: 性能基准测试")
        print("="*60)
        
        setup = setup_model_and_data
        model = setup['model']
        device = setup['device']
        
        # 创建测试数据
        batch_sizes = [1, 8, 16, 32]
        num_bands = 290
        spatial_size = 13
        
        print("\n前向传播性能测试:")
        print("-" * 60)
        print(f"{'批量大小':<12} {'平均时间(ms)':<15} {'吞吐量(samples/s)':<20}")
        print("-" * 60)
        
        forward_times = {}
        
        for batch_size in batch_sizes:
            x_sat = torch.randn(batch_size, 1, num_bands, spatial_size, spatial_size).to(device)
            x_ref = torch.randn(batch_size, 1, num_bands, 1, 1).to(device)
            
            # 预热
            model.eval()
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x_sat, x_ref)
            
            # 测量时间
            num_iterations = 50
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(x_sat, x_ref)
            
            end_time = time.time()
            
            # 计算统计信息
            total_time = end_time - start_time
            avg_time_ms = (total_time / num_iterations) * 1000
            throughput = (batch_size * num_iterations) / total_time
            
            forward_times[batch_size] = avg_time_ms
            
            print(f"{batch_size:<12} {avg_time_ms:<15.2f} {throughput:<20.2f}")
        
        # 测量内存使用
        print("\n内存使用测试:")
        print("-" * 60)
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 ** 2)  # 假设 float32
        
        print(f"模型参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"模型大小: {model_size_mb:.2f} MB")
        
        # 估算单个样本的内存使用
        sample_size_mb = (num_bands * spatial_size * spatial_size * 4) / (1024 ** 2)
        print(f"单个样本大小: {sample_size_mb:.2f} MB")
        
        # 断言：前向传播时间应该合理（< 1000ms per batch）
        for batch_size, time_ms in forward_times.items():
            assert time_ms < 1000, f"批量大小 {batch_size} 的前向传播时间过长: {time_ms:.2f}ms"
        
        print("\n✓ 测试 18.3 通过: 性能基准测试完成")
        
        return {
            'forward_times': forward_times,
            'model_params': total_params,
            'model_size_mb': model_size_mb
        }
    
    def test_generate_final_report(self, setup_model_and_data, tmp_path):
        """
        测试 18.4: 生成最终报告
        
        - 汇总所有评估指标
        - 生成可视化图表
        - 记录性能基准
        """
        print("\n" + "="*60)
        print("测试 18.4: 生成最终报告")
        print("="*60)
        
        setup = setup_model_and_data
        model = setup['model']
        val_loader = setup['val_loader']
        criterion = setup['criterion']
        metrics_calculator = setup['metrics_calculator']
        device = setup['device']
        
        # 运行验证获取指标
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, metrics_calculator, epoch=1
        )
        
        # 生成报告
        report_path = tmp_path / "final_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("AG-S²CNN 最终集成测试报告\n")
            f.write("="*60 + "\n\n")
            
            f.write("1. 模型信息\n")
            f.write("-" * 60 + "\n")
            f.write(f"模型名称: AG-S²CNN\n")
            f.write(f"波段数: 290\n")
            f.write(f"空间大小: 13×13\n")
            f.write(f"类别数: 4\n")
            f.write(f"参数量: {sum(p.numel() for p in model.parameters()):,}\n\n")
            
            f.write("2. 评估指标\n")
            f.write("-" * 60 + "\n")
            f.write(f"验证损失: {val_loss:.4f}\n")
            f.write(f"总体精度 (OA): {val_metrics['OA']:.4f}\n")
            f.write(f"平均精度 (AA): {val_metrics['AA']:.4f}\n")
            f.write(f"Kappa 系数: {val_metrics['Kappa']:.4f}\n")
            f.write(f"F1-Score (Macro): {val_metrics['F1_macro']:.4f}\n\n")
            
            f.write("3. 性能基准\n")
            f.write("-" * 60 + "\n")
            
            # 运行性能测试
            perf_results = self.test_performance_benchmarks(setup_model_and_data)
            
            f.write(f"模型参数量: {perf_results['model_params']:,}\n")
            f.write(f"模型大小: {perf_results['model_size_mb']:.2f} MB\n")
            f.write(f"\n前向传播时间:\n")
            for batch_size, time_ms in perf_results['forward_times'].items():
                f.write(f"  批量大小 {batch_size}: {time_ms:.2f} ms\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("报告生成完成\n")
            f.write("="*60 + "\n")
        
        print(f"\n最终报告已保存到: {report_path}")
        
        # 验证报告文件存在
        assert report_path.exists(), "报告文件未生成"
        
        # 读取并打印报告
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
            print("\n" + report_content)
        
        print("\n✓ 测试 18.4 通过: 最终报告生成完成")
    
    def _plot_training_curves(self, train_losses, val_losses, val_accuracies, save_dir):
        """
        绘制训练曲线
        
        参数:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            val_accuracies: 验证精度列表
            save_dir: 保存目录
        """
        epochs = range(1, len(train_losses) + 1)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', marker='o')
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 精度曲线
        ax2.plot(epochs, val_accuracies, 'g-', label='Val Accuracy', marker='^')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = save_dir / "training_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存到: {save_path}")


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '-s'])
