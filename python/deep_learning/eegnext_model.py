"""
EEG-NeXt: ConvNeXt-based architecture for EEG classification
Based on: Demir et al., "EEG-NEXT: A MODERNIZED CONVNET FOR THE
CLASSIFICATION OF COGNITIVE ACTIVITY FROM EEG" (2022)

Implementation follows Table 1 and Table 2 from the paper exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import numpy as np


class CNBlock(nn.Module):
    """
    ConvNeXt Block (CNBlock) - Table 2 from paper

    Architecture:
    1. Depthwise Conv2D (7x7)
    2. Permute (C,H,W) → (H,W,C)
    3. LayerNorm
    4. Linear (expansion: dim → 4*dim)
    5. GELU
    6. Linear (compression: 4*dim → dim)
    7. Permute (H,W,C) → (C,H,W)
    8. Residual connection
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()

        # Depthwise convolution (groups=dim makes it depthwise)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # LayerNorm (applied in permuted space)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Inverted bottleneck: dim → 4*dim → dim
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Optional: DropPath for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            output: (B, C, H, W)
        """
        input_tensor = x

        # Depthwise convolution
        x = self.dwconv(x)

        # Permute: (B, C, H, W) → (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        # LayerNorm
        x = self.norm(x)

        # Pointwise convolutions (MLP)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Permute back: (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Residual connection
        x = input_tensor + self.drop_path(x)

        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class EEGNeXt(nn.Module):
    """
    EEG-NeXt: Modernized ConvNet for EEG Classification

    Based on Table 1 from the paper.

    Input: EEG Scalogram (C, S, T)
        C = number of EEG channels
        S = number of scales (frequencies)
        T = number of time samples

    Output: Class logits (num_classes,)

    Architecture follows the paper exactly:
    1. Initial Conv2D (3x(7,7)) + GELU
    2. NearestInterpolation to 64x64
    3. Patchify Conv2D (96x(4,4), stride=4)
    4. Stage 1: 3x CNBlock (dim=96)
    5. Downsample: Conv2D (192x(2,2), stride=2)
    6. Stage 2: 3x CNBlock (dim=192)
    7. Downsample: Conv2D (384x(2,2), stride=2)
    8. Stage 3: 9x CNBlock (dim=384)
    9. Downsample: Conv2D (768x(2,2), stride=2)
    10. Stage 4: 3x CNBlock (dim=768)
    11. Global average pooling
    12. LayerNorm + Linear classifier
    """

    def __init__(self,
                 in_channels,  # C: number of EEG channels
                 num_scales,  # S: number of CWT scales (frequencies)
                 num_times,  # T: number of time points
                 num_classes,  # L: number of target classes
                 drop_path_rate=0.0,
                 pretrained=False):
        """
        Args:
            in_channels: Number of EEG channels (e.g., 4 for PO7, Pz, PO8, Fz)
            num_scales: Number of CWT scales (e.g., 40 for 4-42 Hz)
            num_times: Number of time samples (e.g., 5120 for 10s at 512 Hz)
            num_classes: Number of output classes (e.g., 2 for On-Task/MW)
            drop_path_rate: Stochastic depth rate
            pretrained: Use pretrained ConvNeXt weights (ImageNet)
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_scales = num_scales
        self.num_times = num_times
        self.num_classes = num_classes

        # Stage depths (number of CNBlocks per stage)
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]

        # ====================================================================
        # STEM: Initial convolution
        # ====================================================================
        # Input: (B, C, S, T) → (B, 3, S, T)
        self.stem_conv = nn.Conv2d(
            in_channels, 3,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=True
        )
        self.stem_act = nn.GELU()

        # Interpolate to fixed size 64x64 (matching ImageNet pretraining)
        # This allows us to use pretrained weights
        self.interpolate_size = (64, 64)

        # ====================================================================
        # PATCHIFY: Downsampling via strided convolution
        # ====================================================================
        # (B, 3, 64, 64) → (B, 96, 16, 16)
        self.downsample_layers = nn.ModuleList()

        # First downsample (patchify)
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                nn.LayerNorm([dims[0], 16, 16], eps=1e-6)
            )
        )

        # Subsequent downsamples
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.LayerNorm([dims[i], 16 // (2 ** i), 16 // (2 ** i)], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        # ====================================================================
        # STAGES: CNBlocks
        # ====================================================================
        self.stages = nn.ModuleList()

        # Stochastic depth decay rule
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[CNBlock(dim=dims[i], drop_path=dp_rates[cur + j])
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # ====================================================================
        # HEAD: Classification head
        # ====================================================================
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        # Initialize weights
        self.apply(self._init_weights)

        # Load pretrained ConvNeXt weights if requested
        if pretrained:
            self._load_pretrained_weights()

    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_weights(self):
        """
        Load pretrained ConvNeXt-Tiny weights from torchvision

        Only loads compatible layers (stages, downsamples)
        Stem is NOT loaded because input channels differ
        """
        print("Loading pretrained ConvNeXt-Tiny weights...")

        # Load pretrained model
        pretrained_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.state_dict()

        # Filter out incompatible keys
        # Skip 'stem' (different input channels)
        # Skip 'head' (different output classes)
        pretrained_dict_filtered = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict
               and 'features.0.0' not in k  # Skip stem conv
               and 'classifier' not in k  # Skip classifier
               and model_dict[k].shape == v.shape
        }

        # Update model dict
        model_dict.update(pretrained_dict_filtered)
        self.load_state_dict(model_dict)

        print(f"Loaded {len(pretrained_dict_filtered)} pretrained layers")

    def forward_features(self, x):
        """
        Extract features (before classification head)

        Args:
            x: (B, C, S, T) - EEG scalogram
        Returns:
            features: (B, 768) - Global features
        """
        # Stem
        x = self.stem_conv(x)  # (B, C, S, T) → (B, 3, S, T)
        x = self.stem_act(x)

        # Interpolate to 64x64
        x = F.interpolate(x, size=self.interpolate_size, mode='nearest')

        # 4 stages with downsampling
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        # Global average pooling
        x = x.mean([-2, -1])  # (B, 768, 2, 2) → (B, 768)

        return x

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (B, C, S, T) - EEG scalogram
               B = batch size
               C = number of EEG channels
               S = number of scales (frequencies)
               T = number of time points

        Returns:
            logits: (B, num_classes)
        """
        # Extract features
        x = self.forward_features(x)  # (B, C, S, T) → (B, 768)

        # Normalization
        x = self.norm(x)

        # Classification
        x = self.head(x)  # (B, 768) → (B, num_classes)

        return x


def create_eegnext(in_channels=4, num_scales=40, num_times=5120,
                   num_classes=2, pretrained=True):
    """
    Create EEG-NeXt model

    Args:
        in_channels: Number of EEG channels (default: 4 for PO7, Pz, PO8, Fz)
        num_scales: Number of CWT scales (default: 40)
        num_times: Number of time points (default: 5120 for 10s at 512Hz)
        num_classes: Number of classes (default: 2 for binary classification)
        pretrained: Use pretrained ImageNet weights (default: True)

    Returns:
        model: EEGNeXt model

    Example:
    >>> model = create_eegnext(in_channels=4, num_classes=2, pretrained=True)
    >>> x = torch.randn(8, 4, 40, 5120)  # Batch of 8 scalograms
    >>> output = model(x)  # (8, 2)
    """
    model = EEGNeXt(
        in_channels=in_channels,
        num_scales=num_scales,
        num_times=num_times,
        num_classes=num_classes,
        drop_path_rate=0.1,
        pretrained=pretrained
    )
    return model


if __name__ == "__main__":
    """
    Test EEG-NeXt model
    """
    print("=" * 70)
    print("TESTING EEG-NeXt MODEL")
    print("=" * 70)

    # Create model
    model = create_eegnext(
        in_channels=4,  # PO7, Pz, PO8, Fz
        num_scales=40,  # 40 frequencies (4-42 Hz)
        num_times=5120,  # 10 seconds at 512 Hz
        num_classes=2,  # On-Task vs Mind Wandering
        pretrained=True
    )

    print(f"\nModel created:")
    print(f"  Input: (batch, 4, 40, 5120)")
    print(f"  Output: (batch, 2)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 4, 40, 5120)

    print(f"\nTesting forward pass...")
    print(f"  Input shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[0]}")

    # Test feature extraction
    with torch.no_grad():
        features = model.forward_features(x)

    print(f"\nFeature extraction:")
    print(f"  Features shape: {features.shape}")
    print(f"  Features mean: {features.mean():.4f}")
    print(f"  Features std: {features.std():.4f}")

    print("\n" + "=" * 70)
    print("✓ MODEL TEST COMPLETE")
    print("=" * 70)