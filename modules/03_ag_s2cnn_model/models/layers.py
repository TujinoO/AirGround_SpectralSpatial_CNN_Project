"""
基础层和工具模块

包含 AG-S²CNN 模型的基础卷积层和工具组件
"""

import torch
import torch.nn as nn


class ResNetBlock2D(nn.Module):
    """
    2D 残差卷积块
    
    功能:
        - 包含两层 3×3 卷积
        - 残差连接（跳跃连接）
        - BatchNorm 和 ReLU 激活
    
    参数:
        channels (int): 输入和输出通道数
    
    需求: 5.2
    """
    
    def __init__(self, channels):
        super(ResNetBlock2D, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二层卷积
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入特征 (Batch, Channels, H, W)
        
        返回:
            Tensor: 输出特征 (Batch, Channels, H, W)
        """
        identity = x
        
        # 第一层卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二层卷积
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        out += identity
        out = self.relu(out)
        
        return out


class Inception3D(nn.Module):
    """
    Inception-3D 多尺度并行模块
    
    功能:
        - 局部细节分支（3×3×3 卷积）
        - 宏观上下文分支（5×5×3 卷积）
        - 在通道维度拼接两个分支的输出
    
    参数:
        in_channels (int): 输入通道数
    
    需求: 3.2, 3.3, 3.4
    """
    
    def __init__(self, in_channels):
        super(Inception3D, self).__init__()
        
        # 局部细节分支 (小核)
        self.branch_local = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(0.2)
        )
        
        # 宏观上下文分支 (大核)
        # 注意: PyTorch Conv3d kernel_size 是 (D, H, W)，所以 5x5x3 (空x空x谱) 对应 (3, 5, 5)
        self.branch_context = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入特征 (Batch, in_channels, D, H, W)
        
        返回:
            Tensor: 输出特征 (Batch, 48, D, H, W)
        """
        # 局部细节特征
        local_features = self.branch_local(x)
        
        # 宏观上下文特征
        context_features = self.branch_context(x)
        
        # 通道维度拼接
        output = torch.cat([local_features, context_features], dim=1)
        
        return output
