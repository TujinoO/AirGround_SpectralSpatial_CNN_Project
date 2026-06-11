from typing import Callable, Dict

import torch
import torch.nn as nn

from models.ag_s2cnn import AG_S2CNN, Classifier2D, GEncoder, SBackbone


class Standard3DCNNBaseline(nn.Module):
    """
    去除 G-Encoder 与 AG 差分融合，仅保留常规 3D 特征 + 分类头。
    """

    def __init__(self, num_bands: int, spatial_size: int):
        super().__init__()
        self.s_backbone = SBackbone(num_bands=num_bands, spatial_size=spatial_size)
        reduced_bands = (num_bands + 2 * 3 - 7) // 2 + 1
        self.classifier = Classifier2D(
            reduced_bands=reduced_bands,
            spatial_size=spatial_size,
            num_classes=1,
        )

    def forward(self, x_sat: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        del x_ref
        f_sat = self.s_backbone(x_sat)
        return self.classifier(f_sat)


class SingleScaleBackbone(nn.Module):
    def __init__(self, num_bands: int, kernel_hw: int):
        super().__init__()
        self.spectral_reduction = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(7, 1, 1), stride=(2, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2),
        )
        self.single_scale = nn.Sequential(
            nn.Conv3d(16, 24, kernel_size=(3, kernel_hw, kernel_hw), padding=(1, kernel_hw // 2, kernel_hw // 2)),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(0.2),
        )
        self.channel_mapping = nn.Sequential(
            nn.Conv3d(24, 64, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x_sat: torch.Tensor) -> torch.Tensor:
        x = self.spectral_reduction(x_sat)
        x = self.single_scale(x)
        x = self.channel_mapping(x)
        return x


class FusionAblationNet(nn.Module):
    """
    可配置的消融模型：
    - backbone_mode: full | small_only | large_only
    - fusion_mode: full | no_physics | concat | diff_no_att
    """

    def __init__(
        self,
        num_bands: int,
        spatial_size: int,
        backbone_mode: str = "full",
        fusion_mode: str = "full",
        head_mode: str = "lightweight",
    ):
        super().__init__()
        self.backbone_mode = backbone_mode
        self.fusion_mode = fusion_mode
        self.head_mode = head_mode

        self.g_encoder = GEncoder(num_bands=num_bands, spatial_size=spatial_size)
        if backbone_mode == "small_only":
            self.s_backbone = SingleScaleBackbone(num_bands=num_bands, kernel_hw=3)
        elif backbone_mode == "large_only":
            self.s_backbone = SingleScaleBackbone(num_bands=num_bands, kernel_hw=5)
        else:
            self.s_backbone = SBackbone(num_bands=num_bands, spatial_size=spatial_size)

        self.concat_proj = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
        )
        self.diff_no_att_proj = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
        )
        self.full_fusion = AG_S2CNN(num_bands=num_bands, spatial_size=spatial_size, num_classes=1).ag_fusion

        reduced_bands = (num_bands + 2 * 3 - 7) // 2 + 1
        if head_mode == "fc":
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * reduced_bands * spatial_size * spatial_size, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 1),
            )
            self._fc_mode = True
        else:
            self.classifier = Classifier2D(
                reduced_bands=reduced_bands,
                spatial_size=spatial_size,
                num_classes=1,
            )
            self._fc_mode = False

    def _fuse(self, f_sat: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "no_physics":
            return f_sat
        if self.fusion_mode == "concat":
            return self.concat_proj(torch.cat([f_sat, f_ref], dim=1))
        if self.fusion_mode == "diff_no_att":
            f_diff = torch.abs(f_sat - f_ref)
            return self.diff_no_att_proj(torch.cat([f_sat, f_diff], dim=1))
        return self.full_fusion(f_sat, f_ref)

    def forward(self, x_sat: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        f_sat = self.s_backbone(x_sat)
        f_ref = self.g_encoder(x_ref)
        fused = self._fuse(f_sat, f_ref)
        if self._fc_mode:
            return self.classifier(fused)
        return self.classifier(fused)


def get_comparative_deep_builders() -> Dict[str, Callable[[int, int], nn.Module]]:
    return {
        "AG-S2CNN": lambda bands, spatial: AG_S2CNN(num_bands=bands, spatial_size=spatial, num_classes=1),
        "3D-CNN-Standard": lambda bands, spatial: Standard3DCNNBaseline(num_bands=bands, spatial_size=spatial),
    }


def get_ablation_builders() -> Dict[str, Callable[[int, int], nn.Module]]:
    return {
        "Full": lambda b, s: FusionAblationNet(b, s, backbone_mode="full", fusion_mode="full", head_mode="lightweight"),
        "SmallOnly": lambda b, s: FusionAblationNet(b, s, backbone_mode="small_only", fusion_mode="full", head_mode="lightweight"),
        "LargeOnly": lambda b, s: FusionAblationNet(b, s, backbone_mode="large_only", fusion_mode="full", head_mode="lightweight"),
        "NoPhysics": lambda b, s: FusionAblationNet(b, s, backbone_mode="full", fusion_mode="no_physics", head_mode="lightweight"),
        "PhysicsConcat": lambda b, s: FusionAblationNet(b, s, backbone_mode="full", fusion_mode="concat", head_mode="lightweight"),
        "PhysicsDiffNoAtt": lambda b, s: FusionAblationNet(b, s, backbone_mode="full", fusion_mode="diff_no_att", head_mode="lightweight"),
        "FCHead": lambda b, s: FusionAblationNet(b, s, backbone_mode="full", fusion_mode="full", head_mode="fc"),
        "LightHead": lambda b, s: FusionAblationNet(b, s, backbone_mode="full", fusion_mode="full", head_mode="lightweight"),
    }

