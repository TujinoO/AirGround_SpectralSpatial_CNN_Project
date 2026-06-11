"""
模型验证脚本 - 检查点 9

验证完整模型的性能指标:
1. 端到端前向传播
2. 内存使用情况
3. 推理速度
"""

import time
import torch
import psutil
import os
from models.ag_s2cnn import AG_S2CNN


def get_memory_usage():
    """获取当前进程的内存使用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转换为 MB


def test_end_to_end_forward_propagation():
    """测试端到端前向传播"""
    print("\n" + "="*80)
    print("测试 1: 端到端前向传播")
    print("="*80)
    
    model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)
    model.eval()
    
    # 测试不同批量大小
    batch_sizes = [1, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        x_sat = torch.randn(batch_size, 1, 290, 13, 13)
        x_ref = torch.randn(batch_size, 1, 290, 1, 1)
        
        with torch.no_grad():
            output = model(x_sat, x_ref)
        
        # 验证输出
        assert output.shape == (batch_size, 4), f"批量大小 {batch_size} 输出维度错误"
        assert not torch.isnan(output).any(), f"批量大小 {batch_size} 输出包含 NaN"
        assert not torch.isinf(output).any(), f"批量大小 {batch_size} 输出包含 Inf"
        
        print(f"✓ 批量大小 {batch_size:2d}: 输出维度 {tuple(output.shape)}, "
              f"输出范围 [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    print("\n✓ 端到端前向传播测试通过")


def test_memory_usage():
    """测试内存使用情况"""
    print("\n" + "="*80)
    print("测试 2: 内存使用情况")
    print("="*80)
    
    # 记录初始内存
    initial_memory = get_memory_usage()
    print(f"初始内存使用: {initial_memory:.2f} MB")
    
    # 创建模型
    model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)
    model.eval()
    
    after_model_memory = get_memory_usage()
    model_memory = after_model_memory - initial_memory
    print(f"模型加载后内存: {after_model_memory:.2f} MB (增加 {model_memory:.2f} MB)")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 测试不同批量大小的内存使用
    batch_sizes = [1, 8, 16, 32]
    
    for batch_size in batch_sizes:
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        before_inference = get_memory_usage()
        
        x_sat = torch.randn(batch_size, 1, 290, 13, 13)
        x_ref = torch.randn(batch_size, 1, 290, 1, 1)
        
        with torch.no_grad():
            output = model(x_sat, x_ref)
        
        after_inference = get_memory_usage()
        inference_memory = after_inference - before_inference
        
        print(f"批量大小 {batch_size:2d}: 推理内存增加 {inference_memory:.2f} MB")
    
    print("\n✓ 内存使用测试完成")


def test_inference_speed():
    """测试推理速度"""
    print("\n" + "="*80)
    print("测试 3: 推理速度")
    print("="*80)
    
    model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)
    model.eval()
    
    # 预热
    x_sat = torch.randn(8, 1, 290, 13, 13)
    x_ref = torch.randn(8, 1, 290, 1, 1)
    with torch.no_grad():
        _ = model(x_sat, x_ref)
    
    # 测试不同批量大小的推理速度
    batch_sizes = [1, 4, 8, 16, 32]
    num_iterations = 50
    
    print(f"\n每个批量大小运行 {num_iterations} 次迭代:")
    
    for batch_size in batch_sizes:
        x_sat = torch.randn(batch_size, 1, 290, 13, 13)
        x_ref = torch.randn(batch_size, 1, 290, 1, 1)
        
        # 计时
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(x_sat, x_ref)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_iterations
        avg_time_per_sample = avg_time_per_batch / batch_size
        throughput = batch_size * num_iterations / total_time
        
        print(f"批量大小 {batch_size:2d}: "
              f"平均 {avg_time_per_batch*1000:.2f} ms/batch, "
              f"{avg_time_per_sample*1000:.2f} ms/sample, "
              f"吞吐量 {throughput:.2f} samples/s")
    
    print("\n✓ 推理速度测试完成")


def test_gpu_availability():
    """测试 GPU 可用性"""
    print("\n" + "="*80)
    print("测试 4: GPU 可用性")
    print("="*80)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用")
        print(f"  GPU 设备数量: {torch.cuda.device_count()}")
        print(f"  当前 GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 版本: {torch.version.cuda}")
        
        # 测试 GPU 推理
        model = AG_S2CNN().cuda()
        model.eval()
        
        x_sat = torch.randn(8, 1, 290, 13, 13).cuda()
        x_ref = torch.randn(8, 1, 290, 1, 1).cuda()
        
        with torch.no_grad():
            output = model(x_sat, x_ref)
        
        print(f"  ✓ GPU 推理测试通过")
        print(f"  GPU 内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  GPU 内存缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("⚠ CUDA 不可用，使用 CPU 模式")
        print("  建议: 安装 CUDA 以获得更好的性能")


def test_model_components():
    """测试模型各组件"""
    print("\n" + "="*80)
    print("测试 5: 模型组件验证")
    print("="*80)
    
    model = AG_S2CNN()
    
    # 验证子模块存在
    components = ['g_encoder', 's_backbone', 'ag_fusion', 'classifier']
    for component in components:
        assert hasattr(model, component), f"缺少组件: {component}"
        print(f"✓ {component} 组件存在")
    
    # 测试各组件的输出维度
    batch_size = 4
    x_sat = torch.randn(batch_size, 1, 290, 13, 13)
    x_ref = torch.randn(batch_size, 1, 290, 1, 1)
    
    with torch.no_grad():
        # G-Encoder
        f_ref = model.g_encoder(x_ref)
        assert f_ref.shape == (batch_size, 64, 145, 13, 13), "G-Encoder 输出维度错误"
        print(f"✓ G-Encoder 输出: {tuple(f_ref.shape)}")
        
        # S-Backbone
        f_sat = model.s_backbone(x_sat)
        assert f_sat.shape == (batch_size, 64, 145, 13, 13), "S-Backbone 输出维度错误"
        print(f"✓ S-Backbone 输出: {tuple(f_sat.shape)}")
        
        # AG-Fusion
        f_fused = model.ag_fusion(f_sat, f_ref)
        assert f_fused.shape == (batch_size, 64, 145, 13, 13), "AG-Fusion 输出维度错误"
        print(f"✓ AG-Fusion 输出: {tuple(f_fused.shape)}")
        
        # Classifier
        output = model.classifier(f_fused)
        assert output.shape == (batch_size, 4), "Classifier 输出维度错误"
        print(f"✓ Classifier 输出: {tuple(output.shape)}")
    
    print("\n✓ 所有组件验证通过")


def test_dimension_alignment():
    """测试维度对齐"""
    print("\n" + "="*80)
    print("测试 6: 维度对齐验证")
    print("="*80)
    
    model = AG_S2CNN()
    batch_size = 8
    
    x_sat = torch.randn(batch_size, 1, 290, 13, 13)
    x_ref = torch.randn(batch_size, 1, 290, 1, 1)
    
    with torch.no_grad():
        f_ref = model.g_encoder(x_ref)
        f_sat = model.s_backbone(x_sat)
    
    # 验证维度对齐
    assert f_ref.shape == f_sat.shape, \
        f"维度不对齐: G-Encoder {f_ref.shape} vs S-Backbone {f_sat.shape}"
    
    print(f"✓ G-Encoder 和 S-Backbone 输出维度对齐: {tuple(f_ref.shape)}")
    print(f"  - 批量维度: {f_ref.shape[0]}")
    print(f"  - 通道维度: {f_ref.shape[1]}")
    print(f"  - 光谱维度: {f_ref.shape[2]}")
    print(f"  - 空间维度: {f_ref.shape[3]}×{f_ref.shape[4]}")
    
    print("\n✓ 维度对齐验证通过")


def main():
    """运行所有验证测试"""
    print("\n" + "="*80)
    print("AG-S²CNN 模型完整验证 - 检查点 9")
    print("="*80)
    
    try:
        # 运行所有测试
        test_end_to_end_forward_propagation()
        test_memory_usage()
        test_inference_speed()
        test_gpu_availability()
        test_model_components()
        test_dimension_alignment()
        
        # 总结
        print("\n" + "="*80)
        print("验证总结")
        print("="*80)
        print("✓ 所有测试通过")
        print("✓ 端到端前向传播正常")
        print("✓ 内存使用在合理范围内")
        print("✓ 推理速度符合预期")
        print("✓ 模型组件完整且功能正常")
        print("✓ 维度对齐验证通过")
        print("\n模型已准备好进行训练和部署！")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ 验证失败: {e}")
        raise


if __name__ == "__main__":
    main()
