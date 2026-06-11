"""
推理示例脚本

演示如何使用训练好的 AG-S²CNN 模型进行推理预测。

使用方法:
    python examples/demo_inference.py --model outputs/best_model.pth --image data/test_image.npy
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ag_s2cnn import AG_S2CNN
from config import Config


def load_model(model_path, config):
    """
    加载训练好的模型
    
    参数:
        model_path (str): 模型权重路径
        config (Config): 配置对象
    
    返回:
        model: 加载好的模型
    """
    print(f"加载模型: {model_path}")
    
    # 创建模型
    model = AG_S2CNN(
        num_bands=config['model']['num_bands'],
        spatial_size=config['model']['spatial_size'],
        num_classes=config['model']['num_classes']
    )
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print("模型加载成功")
    return model


def extract_patches(image_cube, spatial_size=13):
    """
    从全景影像中提取所有邻域切片
    
    参数:
        image_cube (numpy.ndarray): 高光谱影像 (H, W, Bands)
        spatial_size (int): 邻域大小
    
    返回:
        patches (numpy.ndarray): 切片数组 (N, Bands, S, S)
        positions (list): 切片中心坐标列表
    """
    H, W, Bands = image_cube.shape
    half_size = spatial_size // 2
    
    patches = []
    positions = []
    
    print(f"提取邻域切片 (邻域大小: {spatial_size}×{spatial_size})...")
    
    for i in range(half_size, H - half_size):
        for j in range(half_size, W - half_size):
            # 提取邻域
            patch = image_cube[
                i - half_size:i + half_size + 1,
                j - half_size:j + half_size + 1,
                :
            ]
            
            # 转换为 (Bands, S, S)
            patch = np.transpose(patch, (2, 0, 1))
            patches.append(patch)
            positions.append((i, j))
    
    patches = np.array(patches, dtype=np.float32)
    print(f"提取了 {len(patches)} 个切片")
    
    return patches, positions


def predict_image(model, image_cube, gsrsl, config, device='cpu', batch_size=32):
    """
    对整幅影像进行预测
    
    参数:
        model: 训练好的模型
        image_cube (numpy.ndarray): 高光谱影像 (H, W, Bands)
        gsrsl (dict): 地面标准参考光谱库
        config (Config): 配置对象
        device (str): 设备
        batch_size (int): 批量大小
    
    返回:
        prediction_map (numpy.ndarray): 预测结果图 (H, W)
        confidence_map (numpy.ndarray): 置信度图 (H, W)
    """
    H, W, Bands = image_cube.shape
    spatial_size = config['model']['spatial_size']
    num_classes = config['model']['num_classes']
    
    # 初始化预测图
    prediction_map = np.zeros((H, W), dtype=np.int32)
    confidence_map = np.zeros((H, W), dtype=np.float32)
    
    # 提取所有切片
    patches, positions = extract_patches(image_cube, spatial_size)
    
    # 准备地面流（使用背景类的参考光谱，或者可以根据需要选择）
    # 这里简化处理，使用第一个类别的参考光谱
    ref_spectrum = gsrsl[0]  # 可以根据实际情况调整
    
    # 移动模型到设备
    model = model.to(device)
    
    # 批量推理
    print(f"\n开始推理 (批量大小: {batch_size})...")
    num_batches = (len(patches) + batch_size - 1) // batch_size
    
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="推理进度"):
            # 获取批量数据
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(patches))
            batch_patches = patches[start_idx:end_idx]
            
            # 转换为张量
            x_sat = torch.from_numpy(batch_patches).unsqueeze(1)  # (B, 1, Bands, S, S)
            x_ref = torch.from_numpy(ref_spectrum).float().view(1, 1, -1, 1, 1)
            x_ref = x_ref.repeat(x_sat.size(0), 1, 1, 1, 1)  # (B, 1, Bands, 1, 1)
            
            # 移动到设备
            x_sat = x_sat.to(device)
            x_ref = x_ref.to(device)
            
            # 推理
            outputs = model(x_sat, x_ref)
            probabilities = torch.softmax(outputs, dim=1)
            
            # 获取预测和置信度
            confidences, predictions = torch.max(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # 填充预测图
    print("\n生成预测图...")
    for idx, (i, j) in enumerate(positions):
        prediction_map[i, j] = all_predictions[idx]
        confidence_map[i, j] = all_confidences[idx]
    
    return prediction_map, confidence_map


def predict_single_sample(model, x_sat, x_ref, device='cpu'):
    """
    对单个样本进行预测
    
    参数:
        model: 训练好的模型
        x_sat (numpy.ndarray): 卫星流 (Bands, S, S)
        x_ref (numpy.ndarray): 地面流 (Bands,)
        device (str): 设备
    
    返回:
        prediction (int): 预测类别
        probabilities (numpy.ndarray): 类别概率
    """
    model = model.to(device)
    model.eval()
    
    # 转换为张量
    x_sat = torch.from_numpy(x_sat).float().unsqueeze(0).unsqueeze(0)  # (1, 1, Bands, S, S)
    x_ref = torch.from_numpy(x_ref).float().view(1, 1, -1, 1, 1)  # (1, 1, Bands, 1, 1)
    
    # 移动到设备
    x_sat = x_sat.to(device)
    x_ref = x_ref.to(device)
    
    # 推理
    with torch.no_grad():
        output = model(x_sat, x_ref)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    
    return prediction.item(), probabilities.cpu().numpy()[0]


def visualize_prediction(prediction_map, confidence_map, save_dir='outputs'):
    """
    可视化预测结果
    
    参数:
        prediction_map (numpy.ndarray): 预测结果图
        confidence_map (numpy.ndarray): 置信度图
        save_dir (str): 保存目录
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 预测结果图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(prediction_map, cmap='tab10')
    plt.colorbar(label='Class')
    plt.title('Prediction Map')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Confidence')
    plt.title('Confidence Map')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_result.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"预测结果已保存到: {os.path.join(save_dir, 'prediction_result.png')}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AG-S²CNN 推理示例')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--model', type=str, default='outputs/best_model.pth', help='模型权重路径')
    parser.add_argument('--image', type=str, default=None, help='测试影像路径 (.npy)')
    parser.add_argument('--gsrsl', type=str, default=None, help='GSRSL 路径 (.npy)')
    parser.add_argument('--output', type=str, default='outputs', help='输出目录')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AG-S²CNN 推理示例")
    print("=" * 60)
    
    # 加载配置
    print("\n加载配置...")
    config = Config.from_yaml(args.config)
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(args.model, config)
    
    # 如果提供了影像路径，进行全图推理
    if args.image is not None:
        print("\n" + "=" * 60)
        print("全图推理模式")
        print("=" * 60)
        
        # 加载影像
        print(f"\n加载影像: {args.image}")
        image_cube = np.load(args.image)
        print(f"影像形状: {image_cube.shape}")
        
        # 加载 GSRSL
        if args.gsrsl is not None:
            print(f"加载 GSRSL: {args.gsrsl}")
            gsrsl = np.load(args.gsrsl, allow_pickle=True).item()
        else:
            # 使用模拟数据
            print("警告: 未提供 GSRSL，使用模拟数据")
            gsrsl = {
                i: np.random.randn(config['model']['num_bands']).astype(np.float32)
                for i in range(config['model']['num_classes'])
            }
        
        # 推理
        prediction_map, confidence_map = predict_image(
            model, image_cube, gsrsl, config,
            device=device, batch_size=args.batch_size
        )
        
        # 保存结果
        print("\n保存结果...")
        os.makedirs(args.output, exist_ok=True)
        np.save(os.path.join(args.output, 'prediction_map.npy'), prediction_map)
        np.save(os.path.join(args.output, 'confidence_map.npy'), confidence_map)
        
        # 可视化
        visualize_prediction(prediction_map, confidence_map, args.output)
        
        # 统计信息
        print("\n" + "=" * 60)
        print("预测统计:")
        print("-" * 60)
        for class_id in range(config['model']['num_classes']):
            count = np.sum(prediction_map == class_id)
            percentage = count / prediction_map.size * 100
            print(f"Class {class_id}: {count:6d} 像素 ({percentage:5.2f}%)")
        print("=" * 60)
        
    else:
        # 单样本推理示例
        print("\n" + "=" * 60)
        print("单样本推理示例")
        print("=" * 60)
        
        # 创建模拟数据
        print("\n使用模拟数据进行演示...")
        x_sat = np.random.randn(
            config['model']['num_bands'],
            config['model']['spatial_size'],
            config['model']['spatial_size']
        ).astype(np.float32)
        
        x_ref = np.random.randn(config['model']['num_bands']).astype(np.float32)
        
        # 推理
        prediction, probabilities = predict_single_sample(model, x_sat, x_ref, device)
        
        # 打印结果
        print("\n推理结果:")
        print(f"预测类别: {prediction}")
        print("\n类别概率:")
        for i, prob in enumerate(probabilities):
            print(f"  Class {i}: {prob:.4f}")
        
        print("\n提示: 使用 --image 参数进行全图推理")
    
    print("\n推理完成!")


if __name__ == '__main__':
    main()
