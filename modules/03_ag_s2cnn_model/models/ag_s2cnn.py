"""
AG-S²CNN 主模型模块

包含地面光谱编码器、卫星特征主干、差分融合模块和分类器
"""

import torch
import torch.nn as nn


class GEncoder(nn.Module):
    """
    地面光谱特征编码器 (G-Encoder)
    
    功能:
        1. 提取地面光谱的深层语义特征
        2. 执行 Spatial Broadcasting 扩展到空间维度
    
    参数:
        num_bands (int): 光谱波段数，默认 290
        spatial_size (int): 空间邻域大小，默认 13
    
    需求: 2.1, 2.2, 2.3, 2.4, 2.5
    """
    
    def __init__(self, num_bands=290, spatial_size=25):
        super(GEncoder, self).__init__()
        self.num_bands = num_bands
        self.spatial_size = spatial_size
        
        # Layer G1: 光谱局部特征提取
        # kernel_size=7×1×1, stride=2×1×1 用于光谱降维
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=16,
            kernel_size=(7, 1, 1),
            stride=(2, 1, 1),
            padding=(3, 0, 0)
        )
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.LeakyReLU(0.2)
        
        # Layer G2: 语义空间映射
        # kernel_size=1×1×1 用于通道映射
        self.conv2 = nn.Conv3d(
            in_channels=16,
            out_channels=64,
            kernel_size=(1, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.LeakyReLU(0.2)
    
    def forward(self, x_ref):
        """
        前向传播
        
        参数:
            x_ref (Tensor): 地面流输入 (Batch, 1, Bands, 1, 1)
        
        返回:
            Tensor: 广播后的特征 (Batch, 64, D_new, S, S)
        """
        # 第一层卷积 - 光谱降维
        x = self.conv1(x_ref)  # (Batch, 16, D_new, 1, 1)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # 第二层卷积 - 通道映射
        x = self.conv2(x)  # (Batch, 64, D_new, 1, 1)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Spatial Broadcasting - 扩展到空间维度
        x = self._spatial_broadcast(x)  # (Batch, 64, D_new, S, S)
        
        return x
    
    def _spatial_broadcast(self, x):
        """
        空间广播机制: 将 (N, C, D, 1, 1) 扩展为 (N, C, D, S, S)
        
        参数:
            x (Tensor): 输入特征 (N, C, D, 1, 1)
        
        返回:
            Tensor: 广播后的特征 (N, C, D, S, S)
        
        需求: 2.4
        """
        batch, channels, depth, _, _ = x.shape
        # 使用 repeat 在空间维度复制
        x = x.repeat(1, 1, 1, self.spatial_size, self.spatial_size)
        return x


class SBackbone(nn.Module):
    """
    多尺度卫星特征提取主干 (S-Backbone)
    
    功能:
        1. 光谱降维
        2. Inception-3D 多尺度并行特征提取
        3. 特征融合与通道映射
    
    参数:
        num_bands (int): 光谱波段数，默认 290
        spatial_size (int): 空间邻域大小，默认 13
    
    需求: 3.1, 3.6
    """
    
    def __init__(self, num_bands=290, spatial_size=25):
        super(SBackbone, self).__init__()
        self.num_bands = num_bands
        self.spatial_size = spatial_size
        
        # 光谱降维层
        # kernel_size=(7, 1, 1), stride=(2, 1, 1) 用于光谱降维（第一个维度是光谱维度）
        self.spectral_reduction = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(7, 1, 1), stride=(2, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2)
        )
        
        # Inception-3D 模块 (需要从 layers.py 导入)
        from models.layers import Inception3D
        self.inception = Inception3D(in_channels=16)
        
        # 通道映射层 (48 → 64)
        self.channel_mapping = nn.Sequential(
            nn.Conv3d(48, 64, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x_sat):
        """
        前向传播
        
        参数:
            x_sat (Tensor): 卫星流输入 (Batch, 1, Bands, 13, 13)
        
        返回:
            Tensor: 特征输出 (Batch, 64, D_new, 13, 13)
        """
        # 光谱降维
        x = self.spectral_reduction(x_sat)  # (Batch, 16, D_new, 13, 13)
        
        # 多尺度特征提取
        x = self.inception(x)  # (Batch, 48, D_new, 13, 13)
        
        # 通道映射
        x = self.channel_mapping(x)  # (Batch, 64, D_new, 13, 13)
        
        return x



class AGDifferenceFusion(nn.Module):
    """
    空地光谱差分融合模块 (AG-Difference Fusion) - 注意力增强版
    
    功能:
        1. 计算观测特征与标准特征的绝对差分
        2. 拼接原始特征和差分特征
        3. 1x1x1卷积初步融合
        4. 通道注意力机制加权
    
    需求: 4.1, 4.2, 4.3, 4.4
    """
    
    def __init__(self):
        super(AGDifferenceFusion, self).__init__()
        
        # 自适应融合层 (128 → 64)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2)
        )
        
        # 通道注意力层 (SE Block)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(64, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 64, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, f_sat, f_ref):
        """
        前向传播
        """
        # 验证维度一致性 (需求 4.1)
        assert f_sat.shape == f_ref.shape, \
            f"维度不匹配: f_sat {f_sat.shape} vs f_ref {f_ref.shape}"
        
        # 计算绝对差分 (需求 4.2)
        f_diff = torch.abs(f_sat - f_ref)  # (Batch, 64, D, H, W)
        
        # 拼接原始特征和差分特征 (需求 4.3)
        f_combined = torch.cat([f_sat, f_diff], dim=1)  # (Batch, 128, D, H, W)
        
        # 初步自适应融合
        f_fused = self.fusion_conv(f_combined)  # (Batch, 64, D, H, W)
        
        # 计算并应用通道注意力权重
        attention_weight = self.attention(f_fused)
        f_fused = f_fused * attention_weight
        
        return f_fused



class Classifier2D(nn.Module):
    """
    2D 语义抽象与分类模块
    
    功能:
        1. 3D 到 2D 的维度重塑
        2. 残差卷积块提取空间语义
        3. 全局平均池化
        4. 分类输出 (单通道置信度)
    
    参数:
        reduced_bands (int): 降维后的光谱维度
        spatial_size (int): 空间邻域大小，默认 13
        num_classes (int): 分类类别数，默认 1 (二分类置信度)
    
    需求: 5.1, 5.2, 5.3, 5.4
    """
    
    def __init__(self, reduced_bands, spatial_size=25, num_classes=1, embedding_channels=64):
        super(Classifier2D, self).__init__()
        self.reduced_bands = reduced_bands
        self.spatial_size = spatial_size
        self.num_classes = num_classes
        self.embedding_channels = embedding_channels
        
        # 计算 2D 特征通道数
        # 将 3D 特征 (Batch, 64, D, H, W) 重塑为 2D 特征 (Batch, 64*D, H, W)
        self.feature_channels = 64 * reduced_bands
        
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(self.feature_channels, embedding_channels, kernel_size=1),
            nn.BatchNorm2d(embedding_channels),
            nn.ReLU(inplace=True)
        )

        from models.layers import ResNetBlock2D
        # 仅保留一个残差块，防止在小样本上过拟合
        self.resnet_block1 = ResNetBlock2D(embedding_channels)
        
        # 全局平均池化 (需求 5.3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 加大 Dropout 比例防过拟合
        self.dropout = nn.Dropout(p=0.5)
        
        # 分类器 (需求 5.4)
        self.fc = nn.Linear(embedding_channels, num_classes)
    
    def forward(self, x):
        """
        前向传播
        """
        batch_size = x.size(0)
        
        # Reshape: 3D → 2D (合并光谱维度到通道维度) (需求 5.1)
        x = x.view(batch_size, -1, self.spatial_size, self.spatial_size)

        x = self.channel_reduction(x)
        
        # 残差卷积块 (需求 5.2)
        x = self.resnet_block1(x)
        
        # 全局平均池化 (需求 5.3)
        x = self.gap(x)  # (Batch, Channels, 1, 1)
        x = x.view(batch_size, -1)  # (Batch, Channels)
        
        # Dropout
        x = self.dropout(x)
        
        # 分类 (需求 5.4)
        x = self.fc(x)  # (Batch, num_classes)
        
        return x



class AG_S2CNN(nn.Module):
    """
    AG-S²CNN 主模型类
    
    空地协同光谱-空间卷积神经网络，用于高海拔荒漠区伟晶岩型锂矿预测。
    
    架构:
        1. G-Encoder: 地面光谱特征编码器
        2. S-Backbone: 多尺度卫星特征提取主干
        3. AG-Fusion: 空地光谱差分融合模块
        4. Classifier: 2D 语义抽象与分类模块
    
    参数:
        num_bands (int): 光谱波段数，默认 290
        spatial_size (int): 空间邻域大小，默认 13
        num_classes (int): 分类类别数，默认 4
    
    需求: 1.1, 1.2, 1.3
    """
    
    def __init__(self, num_bands=290, spatial_size=25, num_classes=4):
        super(AG_S2CNN, self).__init__()
        self.num_bands = num_bands
        self.spatial_size = spatial_size
        self.num_classes = num_classes
        
        # 计算降维后的光谱维度
        # 经过 stride=2 的卷积后: D_new = (D + 2*padding - kernel_size) / stride + 1
        # 对于 kernel_size=7, stride=2, padding=3: D_new = (290 + 6 - 7) / 2 + 1 = 145
        self.reduced_bands = self._calculate_reduced_bands(num_bands)
        
        # 初始化子模块
        self.g_encoder = GEncoder(num_bands, spatial_size)
        self.s_backbone = SBackbone(num_bands, spatial_size)
        self.ag_fusion = AGDifferenceFusion()
        self.classifier = Classifier2D(self.reduced_bands, spatial_size, num_classes)
    
    def forward(self, x_sat, x_ref):
        """
        前向传播
        
        参数:
            x_sat (Tensor): 卫星流输入 (Batch, 1, Bands, 13, 13)
            x_ref (Tensor): 地面流输入 (Batch, 1, Bands, 1, 1)
        
        返回:
            Tensor: 类别概率分布 (Batch, num_classes)
        
        需求: 1.1, 1.2, 1.3
        """
        # 验证输入维度 (需求 1.1, 1.2, 1.3)
        self._validate_input(x_sat, x_ref)
        
        # 地面流编码 (需求 2.1-2.6)
        f_ref = self.g_encoder(x_ref)  # (Batch, 64, D_new, 13, 13)
        
        # 卫星流特征提取 (需求 3.1-3.8)
        f_sat = self.s_backbone(x_sat)  # (Batch, 64, D_new, 13, 13)
        
        # 差分融合 (需求 4.1-4.5)
        f_fused = self.ag_fusion(f_sat, f_ref)  # (Batch, 64, D_new, 13, 13)
        
        # 分类 (需求 5.1-5.6)
        output = self.classifier(f_fused)  # (Batch, num_classes)
        
        return output
    
    def _validate_input(self, x_sat, x_ref):
        """
        验证输入数据的维度和合法性
        
        参数:
            x_sat (Tensor): 卫星流输入
            x_ref (Tensor): 地面流输入
        
        抛出:
            ValueError: 当输入维度不匹配或包含非法值时
        
        需求: 1.1, 1.2, 1.3, 1.4
        """
        # 验证卫星流维度 (需求 1.1)
        expected_sat_shape = (x_sat.size(0), 1, self.num_bands, self.spatial_size, self.spatial_size)
        if x_sat.shape != expected_sat_shape:
            raise ValueError(
                f"维度错误: 卫星流输入期望维度为 (Batch, 1, {self.num_bands}, {self.spatial_size}, {self.spatial_size})，"
                f"实际接收到 {tuple(x_sat.shape)}"
            )
        
        # 验证地面流维度 (需求 1.2)
        expected_ref_shape = (x_ref.size(0), 1, self.num_bands, 1, 1)
        if x_ref.shape != expected_ref_shape:
            raise ValueError(
                f"维度错误: 地面流输入期望维度为 (Batch, 1, {self.num_bands}, 1, 1)，"
                f"实际接收到 {tuple(x_ref.shape)}"
            )
        
        # 验证批量大小一致性
        if x_sat.size(0) != x_ref.size(0):
            raise ValueError(
                f"批量大小不一致: 卫星流批量大小为 {x_sat.size(0)}，"
                f"地面流批量大小为 {x_ref.size(0)}"
            )
        
        # 验证批量大小范围 (需求 1.5)
        batch_size = x_sat.size(0)
        if batch_size < 1 or batch_size > 512:
            raise ValueError(
                f"批量大小超出范围: 期望 1-512，实际为 {batch_size}"
            )
        
        # 检查 NaN 值 (需求 1.4)
        if torch.isnan(x_sat).any():
            raise ValueError("卫星流输入包含 NaN 值")
        if torch.isnan(x_ref).any():
            raise ValueError("地面流输入包含 NaN 值")
        
        # 检查 Inf 值 (需求 1.4)
        if torch.isinf(x_sat).any():
            raise ValueError("卫星流输入包含 Inf 值")
        if torch.isinf(x_ref).any():
            raise ValueError("地面流输入包含 Inf 值")
    
    def _calculate_reduced_bands(self, num_bands):
        """
        计算降维后的光谱维度
        
        参数:
            num_bands (int): 原始波段数
        
        返回:
            int: 降维后的波段数
        
        公式: D_new = (D + 2*padding - kernel_size) / stride + 1
        对于 kernel_size=7, stride=2, padding=3:
        D_new = (num_bands + 2*3 - 7) / 2 + 1
        """
        kernel_size = 7
        stride = 2
        padding = 3
        reduced = (num_bands + 2 * padding - kernel_size) // stride + 1
        return reduced
