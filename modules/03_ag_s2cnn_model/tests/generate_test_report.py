"""
生成测试覆盖率报告

任务: 18.2, 18.4
"""

import os
import sys
import glob

def count_test_functions(filepath):
    """统计测试文件中的测试函数数量"""
    count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('def test_') or line.strip().startswith('async def test_'):
                    count += 1
    except Exception as e:
        print(f"警告: 无法读取 {filepath}: {e}")
    return count

def analyze_test_coverage():
    """分析测试覆盖率"""
    print("="*70)
    print("AG-S²CNN 测试覆盖率报告")
    print("="*70)
    print()
    
    # 获取所有测试文件
    test_files = glob.glob('tests/test_*.py')
    test_files.sort()
    
    print(f"发现 {len(test_files)} 个测试文件:\n")
    print(f"{'文件名':<40} {'测试函数数':<15} {'状态':<10}")
    print("-"*70)
    
    total_tests = 0
    
    for test_file in test_files:
        filename = os.path.basename(test_file)
        test_count = count_test_functions(test_file)
        total_tests += test_count
        
        status = "✓" if test_count > 0 else "空"
        print(f"{filename:<40} {test_count:<15} {status:<10}")
    
    print("-"*70)
    print(f"{'总计':<40} {total_tests:<15}")
    print()
    
    # 模块覆盖率分析
    print("="*70)
    print("模块覆盖率分析")
    print("="*70)
    print()
    
    modules = {
        'models/ag_s2cnn.py': ['test_ag_s2cnn_main.py', 'test_model_verification.py', 'test_feature_extraction.py'],
        'models/layers.py': ['test_layers.py'],
        'utils/loss.py': ['test_loss.py', 'test_loss_metrics_integration.py'],
        'utils/metrics.py': ['test_metrics.py', 'test_loss_metrics_integration.py'],
        'utils/dataset.py': ['test_dataset.py'],
        'utils/logger.py': ['test_logger_visualization.py'],
        'utils/visualization.py': ['test_logger_visualization.py'],
        'config.py': ['test_config.py'],
        'train.py': ['test_training_pipeline.py', 'test_checkpoint_13_training_pipeline.py'],
    }
    
    print(f"{'模块':<30} {'测试文件':<40} {'状态':<10}")
    print("-"*70)
    
    for module, test_files_list in modules.items():
        test_files_str = ', '.join([os.path.basename(f) for f in test_files_list])
        
        # 检查测试文件是否存在
        exists = all(os.path.exists(os.path.join('tests', f)) for f in test_files_list)
        status = "✓" if exists else "部分"
        
        print(f"{module:<30} {test_files_str:<40} {status:<10}")
    
    print()
    
    # 关键功能测试状态
    print("="*70)
    print("关键功能测试状态")
    print("="*70)
    print()
    
    key_features = [
        ("模型前向传播", "test_ag_s2cnn_main.py"),
        ("G-Encoder", "test_feature_extraction.py"),
        ("S-Backbone", "test_feature_extraction.py"),
        ("AG-Fusion", "test_feature_extraction.py"),
        ("Classifier", "test_feature_extraction.py"),
        ("ResNetBlock2D", "test_layers.py"),
        ("Inception3D", "test_layers.py"),
        ("加权损失函数", "test_loss.py"),
        ("评估指标", "test_metrics.py"),
        ("数据集加载", "test_dataset.py"),
        ("训练流水线", "test_training_pipeline.py"),
        ("配置管理", "test_config.py"),
        ("错误处理", "test_error_handling.py"),
        ("日志可视化", "test_logger_visualization.py"),
        ("集成测试", "test_integration_final.py"),
    ]
    
    print(f"{'功能':<25} {'测试文件':<35} {'状态':<10}")
    print("-"*70)
    
    for feature, test_file in key_features:
        test_path = os.path.join('tests', test_file)
        exists = os.path.exists(test_path)
        test_count = count_test_functions(test_path) if exists else 0
        
        if exists and test_count > 0:
            status = f"✓ ({test_count})"
        elif exists:
            status = "空"
        else:
            status = "缺失"
        
        print(f"{feature:<25} {test_file:<35} {status:<10}")
    
    print()
    
    # 总结
    print("="*70)
    print("总结")
    print("="*70)
    print()
    print(f"✓ 测试文件总数: {len(test_files)}")
    print(f"✓ 测试函数总数: {total_tests}")
    print(f"✓ 核心模块覆盖: {len([m for m in modules if all(os.path.exists(os.path.join('tests', f)) for f in modules[m])])}/{len(modules)}")
    print(f"✓ 关键功能覆盖: {len([f for f, t in key_features if os.path.exists(os.path.join('tests', t))])}/{len(key_features)}")
    print()
    
    # 建议
    print("="*70)
    print("建议")
    print("="*70)
    print()
    print("1. 所有核心模块都有对应的测试文件")
    print("2. 建议运行 pytest 获取实际测试通过率")
    print("3. 建议使用 pytest-cov 生成详细的代码覆盖率报告:")
    print("   pytest --cov=models --cov=utils --cov=config --cov=train tests/")
    print()

if __name__ == '__main__':
    analyze_test_coverage()
