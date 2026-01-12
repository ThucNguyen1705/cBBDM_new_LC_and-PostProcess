"""
Phase 2: Texture Refinement Module (TRM)
=========================================

Inspired by RLI-DM's CMRM (Color Map Refinement Module), this module refines 
coarse RGB output from Phase 1 (LBBDM) to produce sharper textures and finer details.

Architecture Design:
    Input: Coarse RGB (3) + SAR Raw (1) + Edge Map (1) = 5 channels
    Output: Refined RGB (3 channels)
    
    Uses residual learning: refined_rgb = coarse_rgb + learned_residual
    
Key Components:
    1. Feature Extraction: Extract multi-scale features from all inputs
    2. MARM Processing: Use MARM blocks for texture-aware processing
    3. Residual Refinement: Learn texture residual to add to coarse output
    4. Skip Connections: Preserve high-frequency details from SAR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BrownianBridge.base.modules.marm import MARM, ATFM


class ResidualBlock(nn.Module):
    """Residual block with Instance Normalization"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
        )
        
    def forward(self, x):
        return x + self.block(x)


class DownBlock(nn.Module):
    """Downsample block with conv stride 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True),
        )
        
    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample block with transposed conv"""
    def __init__(self, in_ch, out_ch, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        
        skip_ch = out_ch if use_skip else 0
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True),
        )
        
    def forward(self, x, skip=None):
        x = self.up(x)
        if self.use_skip and skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TextureAttentionModule(nn.Module):
    """
    Cross-attention module to extract texture information from SAR
    and apply it to coarse RGB features.
    """
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, rgb_feat, sar_feat):
        """
        Args:
            rgb_feat: (B, C, H, W) - Features from coarse RGB
            sar_feat: (B, C, H, W) - Features from SAR (texture source)
        Returns:
            refined_feat: (B, C, H, W) - RGB features refined with SAR texture
        """
        B, C, H, W = rgb_feat.shape
        
        Q = self.query(rgb_feat).view(B, C, -1)  # (B, C, H*W)
        K = self.key(sar_feat).view(B, C, -1)     # (B, C, H*W)
        V = self.value(sar_feat).view(B, C, -1)   # (B, C, H*W)
        
        # Attention: (B, H*W, H*W)
        attn = torch.softmax(torch.bmm(Q.permute(0, 2, 1), K) / (C ** 0.5), dim=-1)
        
        # Apply attention to SAR values
        out = torch.bmm(V, attn.permute(0, 2, 1)).view(B, C, H, W)
        
        # Residual connection with learnable weight
        return rgb_feat + self.gamma * out


class TextureRefinementModule(nn.Module):
    """
    Phase 2: Texture Refinement Module
    
    Refines coarse RGB output from Phase 1 LBBDM by:
    1. Extracting texture cues from SAR Raw
    2. Using Edge Map for structural guidance
    3. Learning texture residual via MARM blocks
    
    Input:
        - coarse_rgb: (B, 3, H, W) - Coarse output from Phase 1
        - sar_raw: (B, 1, H, W) - Original SAR input
        - edge_map: (B, 1, H, W) - Edge map for structure
        
    Output:
        - refined_rgb: (B, 3, H, W) - Refined output with better texture
    """
    
    def __init__(self,
                 rgb_channels=3,
                 sar_channels=1,
                 edge_channels=1,
                 base_dim=64,
                 n_marms=3,
                 n_res_blocks=4):
        super().__init__()
        
        self.base_dim = base_dim
        input_channels = rgb_channels + sar_channels + edge_channels  # 5
        
        # =====================================================
        # Initial Feature Extraction (no downsampling)
        # =====================================================
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_dim, 7, padding=3),
            nn.InstanceNorm2d(base_dim),
            nn.ReLU(True),
        )
        
        # =====================================================
        # Encoder Path (downsample 2x to preserve detail)
        # =====================================================
        self.down1 = DownBlock(base_dim, base_dim * 2)      # 64 -> 128
        self.down2 = DownBlock(base_dim * 2, base_dim * 4)  # 128 -> 256
        
        # =====================================================
        # SAR Texture Extractor (separate path for texture)
        # =====================================================
        self.sar_enc = nn.Sequential(
            nn.Conv2d(sar_channels, base_dim, 7, padding=3),
            nn.ReLU(True),
            DownBlock(base_dim, base_dim * 2),
            DownBlock(base_dim * 2, base_dim * 4),
        )
        
        # =====================================================
        # Edge Structure Extractor
        # =====================================================
        self.edge_enc = nn.Sequential(
            nn.Conv2d(edge_channels, 32, 3, padding=1),
            nn.ReLU(True),
            DownBlock(32, base_dim),
            DownBlock(base_dim, base_dim * 2),
        )
        
        # =====================================================
        # Texture Attention: Cross-attend SAR texture to RGB
        # =====================================================
        self.texture_attn = TextureAttentionModule(base_dim * 4)
        
        # =====================================================
        # MARM Processing (texture-aware refinement)
        # =====================================================
        self.marms = nn.Sequential(*[MARM(channels=base_dim * 4) for _ in range(n_marms)])
        
        # =====================================================
        # Residual Blocks (fine-grained processing)
        # =====================================================
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(base_dim * 4) for _ in range(n_res_blocks)
        ])
        
        # =====================================================
        # Decoder Path (upsample back to original size)
        # =====================================================
        self.up1 = UpBlock(base_dim * 4, base_dim * 2, use_skip=True)  # 256 -> 128
        self.up2 = UpBlock(base_dim * 2, base_dim, use_skip=True)       # 128 -> 64
        
        # Final Refinement (predict residual)
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.InstanceNorm2d(base_dim),
            nn.ReLU(True),
            nn.Conv2d(base_dim, rgb_channels, 7, padding=3),
            nn.Tanh(),  # Residual in [-1, 1] range
        )
        
        # FIXED residual scaling factor (not learnable to prevent color shift)
        self.register_buffer('residual_scale', torch.tensor(0.1))
        
        print(f"[TextureRefinementModule] Initialized with {n_marms} MARM blocks, "
              f"{n_res_blocks} ResBlocks, base_dim={base_dim}")
    
    def forward(self, coarse_rgb, sar_raw, edge_map):
        """
        Args:
            coarse_rgb: (B, 3, H, W) - Coarse RGB from Phase 1, range [-1, 1]
            sar_raw: (B, 1, H, W) - Original SAR, range [-1, 1]
            edge_map: (B, 1, H, W) - Edge map, range [0, 1]
            
        Returns:
            refined_rgb: (B, 3, H, W) - Refined RGB, range [-1, 1]
        """
        # Normalize inputs to [0, 1] for processing
        rgb_norm = (coarse_rgb + 1.0) / 2.0
        sar_norm = (sar_raw + 1.0) / 2.0
        
        # Concatenate inputs and extract initial features
        x = torch.cat([rgb_norm, sar_norm, edge_map], dim=1)  # (B, 5, H, W)
        x0 = self.init_conv(x)  # (B, 64, H, W)
        
        # Encoder with skip connections
        x1 = self.down1(x0)  # (B, 128, H/2, W/2)
        x2 = self.down2(x1)  # (B, 256, H/4, W/4)
        
        # Extract SAR texture features
        sar_feat = self.sar_enc(sar_norm)  # (B, 256, H/4, W/4)
        
        # Cross-attention: Apply SAR texture to RGB features
        x2 = self.texture_attn(x2, sar_feat)
        
        # MARM Processing
        x2 = self.marms(x2)
        
        # Residual Block Processing
        x2 = self.res_blocks(x2)
        
        # Decoder with skip connections
        x1_up = self.up1(x2, x1)  # (B, 128, H/2, W/2)
        x0_up = self.up2(x1_up, x0)  # (B, 64, H, W)
        
        # Predict residual and apply to coarse RGB
        residual = self.final_conv(x0_up)  # (B, 3, H, W) in [-1, 1]
        
        # Zero-center the residual to prevent global color shift
        # This ensures residual adds texture, not color bias
        residual = residual - residual.mean(dim=[2, 3], keepdim=True)
        
        # Apply scaled residual (scale is fixed at 0.1)
        refined_rgb = coarse_rgb + self.residual_scale * residual
        
        # Clamp to valid range
        refined_rgb = torch.clamp(refined_rgb, -1.0, 1.0)
        
        return refined_rgb


class TextureRefinementModuleLite(nn.Module):
    """
    Lightweight version of TRM for faster training.
    Uses fewer MARM blocks and smaller base dimension.
    """
    
    def __init__(self,
                 rgb_channels=3,
                 sar_channels=1,
                 edge_channels=1,
                 base_dim=32,
                 n_marms=2,
                 n_res_blocks=2):
        super().__init__()
        
        input_channels = rgb_channels + sar_channels + edge_channels
        
        # Single-scale processing (no encoder-decoder)
        self.feat_ext = nn.Sequential(
            nn.Conv2d(input_channels, base_dim, 7, padding=3),
            nn.InstanceNorm2d(base_dim),
            nn.ReLU(True),
            nn.Conv2d(base_dim, base_dim * 2, 3, padding=1),
            nn.InstanceNorm2d(base_dim * 2),
            nn.ReLU(True),
        )
        
        # MARM + ResBlocks
        self.process = nn.Sequential(
            *[MARM(channels=base_dim * 2) for _ in range(n_marms)],
            *[ResidualBlock(base_dim * 2) for _ in range(n_res_blocks)],
        )
        
        # Output
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim, 3, padding=1),
            nn.InstanceNorm2d(base_dim),
            nn.ReLU(True),
            nn.Conv2d(base_dim, rgb_channels, 7, padding=3),
            nn.Tanh(),
        )
        
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, coarse_rgb, sar_raw, edge_map):
        rgb_norm = (coarse_rgb + 1.0) / 2.0
        sar_norm = (sar_raw + 1.0) / 2.0
        
        x = torch.cat([rgb_norm, sar_norm, edge_map], dim=1)
        feat = self.feat_ext(x)
        feat = self.process(feat)
        residual = self.out_conv(feat)
        
        refined_rgb = coarse_rgb + self.residual_scale * residual
        return torch.clamp(refined_rgb, -1.0, 1.0)
