import torch
import torch.nn as nn
import torch.nn.functional as F

class ZPool(nn.Module):
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        return torch.cat([max_pool, avg_pool], dim=1)

class AxialAttention(nn.Module):
    def __init__(self, dim, mode='height'):
        super().__init__()
        self.mode = mode
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        input_shape = x.shape
        if input_shape[1] == self.dim: x_perm = x.permute(0, 2, 3, 1)
        elif input_shape[2] == self.dim: x_perm = x.permute(0, 1, 3, 2)
        elif input_shape[3] == self.dim: x_perm = x.permute(0, 1, 2, 3)
        else: x_perm = x.permute(0, 2, 3, 1)

        B, D1, D2, C = x_perm.shape
        if self.mode == 'height': x_in = x_perm.permute(0, 2, 1, 3).reshape(B * D2, D1, C)
        else: x_in = x_perm.reshape(B * D1, D2, C)

        qkv = self.qkv(x_in)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = C ** -0.5
        attn = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        out = self.proj(attn @ v)

        if self.mode == 'height': out = out.view(B, D2, D1, C).permute(0, 2, 1, 3)
        else: out = out.view(B, D1, D2, C)

        if input_shape[1] == self.dim: out = out.permute(0, 3, 1, 2)
        elif input_shape[2] == self.dim: out = out.permute(0, 1, 3, 2)
        elif input_shape[3] == self.dim: out = out.permute(0, 1, 2, 3)
        return out

class ATFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.zpool = ZPool()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.axial_h = AxialAttention(channels, mode='height')
        self.axial_w = AxialAttention(channels, mode='width')

    def forward(self, x):
        z = self.zpool(x)
        w = torch.sigmoid(self.conv(z))
        x_res = w * x 
        return self.axial_w(self.axial_h(x_res)) + x_res

class MARM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branchA = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.InstanceNorm2d(channels), nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.InstanceNorm2d(channels), nn.Dropout(0.2),
        )
        self.atfm_hw = ATFM(channels)
        self.atfm_ch = ATFM(channels)
        self.atfm_cw = ATFM(channels) 

    def rotate_c_h(self, x): return x.permute(0, 2, 1, 3)
    def unrotate_c_h(self, x): return x.permute(0, 2, 1, 3)
    def rotate_c_w(self, x): return x.permute(0, 3, 2, 1) 
    def unrotate_c_w(self, x): return x.permute(0, 3, 2, 1)

    def forward(self, X):
        A = self.branchA(X)
        B = self.atfm_hw(X)
        C = self.unrotate_c_h(self.atfm_ch(self.rotate_c_h(X)))
        D = self.unrotate_c_w(self.atfm_cw(self.rotate_c_w(X)))
        return A + (B + C + D) / 3

class MARMConditionModel(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, base_dim=64, n_marms=4, **kwargs):
        super().__init__()
        if 'n_stages' in kwargs: print(f"Warning: 'n_stages' ignored.")

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 7, padding=3), nn.InstanceNorm2d(base_dim), nn.ReLU(True),
            nn.Conv2d(base_dim, base_dim * 2, 4, stride=2, padding=1), nn.InstanceNorm2d(base_dim * 2), nn.ReLU(True),
            nn.Conv2d(base_dim * 2, base_dim * 4, 4, stride=2, padding=1), nn.InstanceNorm2d(base_dim * 4), nn.ReLU(True),
        )
        marm_dim = base_dim * 4 
        self.marms = nn.Sequential(*[MARM(channels=marm_dim) for _ in range(n_marms)])
        self.final_proj = nn.Conv2d(marm_dim, out_channels, 1, padding=0)

    def forward(self, x):
        return self.final_proj(self.marms(self.enc(x)))


class MARMHierarchical(nn.Module):
    """
    Hierarchical-Lite MARM: LC (+ Edge) guides SAR at EACH processing stage.
    
    Architecture:
        LC/Edge -> Lightweight Stem -> LC_Features (extracted once)
        SAR -> Encoder -> MARM_1 <- inject(LC_Features)
                             |
                          MARM_2 <- inject(LC_Features)
                             |
                          MARM_3 <- inject(LC_Features)
                             |
                         Output
    
    Benefits:
        - LC semantics guide SAR at every stage (not just fused at end)
        - Fewer MARM blocks than parallel processing (3-4 vs 10)
        - Faster training (15-20 min/epoch vs 40 min)
    """
    def __init__(self, 
                 sar_in_channels=3,
                 lc_in_channels=64,
                 edge_in_channels=1,
                 out_channels=128,
                 base_dim=64,
                 n_marms=3,
                 use_edge=True,
                 injection_type='add',  # 'add', 'concat', 'attention'
                 **kwargs):
        super().__init__()
        
        self.use_edge = use_edge
        self.injection_type = injection_type
        self.n_marms = n_marms
        self.marm_dim = base_dim * 4  # 256 channels after encoder
        
        # =====================================================
        # SAR Encoder (downsample 4x: 256 -> 64)
        # =====================================================
        self.sar_enc = nn.Sequential(
            nn.Conv2d(sar_in_channels, base_dim, 7, padding=3),
            nn.InstanceNorm2d(base_dim), nn.ReLU(True),
            nn.Conv2d(base_dim, base_dim * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_dim * 2), nn.ReLU(True),
            nn.Conv2d(base_dim * 2, self.marm_dim, 4, stride=2, padding=1),
            nn.InstanceNorm2d(self.marm_dim), nn.ReLU(True),
        )
        
        # =====================================================
        # LC Feature Extractor (lightweight, no MARM)
        # =====================================================
        self.lc_enc = nn.Sequential(
            nn.Conv2d(lc_in_channels, base_dim, 3, padding=1),
            nn.InstanceNorm2d(base_dim), nn.ReLU(True),
            nn.Conv2d(base_dim, base_dim * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_dim * 2), nn.ReLU(True),
            nn.Conv2d(base_dim * 2, self.marm_dim, 4, stride=2, padding=1),
            nn.InstanceNorm2d(self.marm_dim), nn.ReLU(True),
        )
        
        # =====================================================
        # Edge Feature Extractor (optional, lightweight)
        # =====================================================
        if self.use_edge:
            self.edge_enc = nn.Sequential(
                nn.Conv2d(edge_in_channels, 32, 3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, base_dim, 4, stride=2, padding=1),
                nn.InstanceNorm2d(base_dim), nn.ReLU(True),
                nn.Conv2d(base_dim, self.marm_dim, 4, stride=2, padding=1),
                nn.InstanceNorm2d(self.marm_dim), nn.ReLU(True),
            )
        
        # =====================================================
        # LC Injection Projections (one per MARM stage)
        # Project LC features to match MARM input at each stage
        # =====================================================
        if self.injection_type == 'concat':
            # Concat doubles channels, so we need to reduce back
            inject_in = self.marm_dim * 2
            if self.use_edge:
                inject_in = self.marm_dim * 3  # SAR + LC + Edge
        else:
            inject_in = self.marm_dim
            
        self.lc_inject_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.marm_dim, self.marm_dim, 1),
                nn.InstanceNorm2d(self.marm_dim),
            ) for _ in range(n_marms)
        ])
        
        if self.use_edge:
            self.edge_inject_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.marm_dim, self.marm_dim, 1),
                    nn.InstanceNorm2d(self.marm_dim),
                ) for _ in range(n_marms)
            ])
        
        # For concat injection, we need fusion after each injection
        if self.injection_type == 'concat':
            self.inject_fusions = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(inject_in, self.marm_dim, 1),
                    nn.InstanceNorm2d(self.marm_dim),
                    nn.ReLU(True),
                ) for _ in range(n_marms)
            ])
        
        # MARM Blocks (shared, process SAR with LC injection)
        self.marms = nn.ModuleList([MARM(channels=self.marm_dim) for _ in range(n_marms)])
        
        # Final Projection to output channels
        self.final_proj = nn.Conv2d(self.marm_dim, out_channels, 1)
        
        print(f"[MARMHierarchical] Initialized with {n_marms} MARM blocks, "
              f"injection_type='{injection_type}', use_edge={use_edge}")
    
    def forward(self, x_sar, x_lc, x_edge=None):
        """
        Args:
            x_sar: Raw SAR image (B, 3, H, W) in [-1, 1]
            x_lc: LC features from embedding (B, embed_dim, H, W)
            x_edge: Edge map (B, 1, H, W) in [0, 1], optional
            
        Returns:
            context: (B, out_channels, H', W') for UNet cross-attention
        """
        # Normalize SAR from [-1, 1] to [0, 1]
        x_sar_norm = (x_sar + 1.0) / 2.0
        
        # =====================================================
        # Extract features (done once, reused at each stage)
        # =====================================================
        sar_feat = self.sar_enc(x_sar_norm)  # (B, marm_dim, H/4, W/4)
        lc_feat = self.lc_enc(x_lc)           # (B, marm_dim, H/4, W/4)
        
        if self.use_edge and x_edge is not None:
            # Resize edge to match LC size if needed
            if x_edge.shape[-2:] != x_lc.shape[-2:]:
                x_edge = F.interpolate(x_edge, size=x_lc.shape[-2:], 
                                       mode='bilinear', align_corners=False)
            edge_feat = self.edge_enc(x_edge)  # (B, marm_dim, H/4, W/4)
        else:
            edge_feat = None
        
        # =====================================================
        # Hierarchical Processing: LC/Edge inject at each stage
        # =====================================================
        x = sar_feat
        
        for i in range(self.n_marms):
            # Get projected LC guidance for this stage
            lc_inject = self.lc_inject_projs[i](lc_feat)
            
            # Get projected Edge guidance for this stage (if available)
            if self.use_edge and edge_feat is not None:
                edge_inject = self.edge_inject_projs[i](edge_feat)
            else:
                edge_inject = None
            
            # Apply injection based on type
            if self.injection_type == 'add':
                # Additive injection: x = x + LC + Edge
                x = x + lc_inject
                if edge_inject is not None:
                    x = x + edge_inject
                    
            elif self.injection_type == 'concat':
                # Concatenation injection: x = Fusion([x, LC, Edge])
                if edge_inject is not None:
                    x = torch.cat([x, lc_inject, edge_inject], dim=1)
                else:
                    x = torch.cat([x, lc_inject], dim=1)
                x = self.inject_fusions[i](x)
                
            elif self.injection_type == 'attention':
                # Gate-based injection: x = x + sigmoid(LC) * Edge
                gate = torch.sigmoid(lc_inject)
                if edge_inject is not None:
                    x = x + gate * edge_inject
                else:
                    x = x + gate * lc_inject
            
            # Process through MARM block
            x = self.marms[i](x)
        
        # Final projection
        return self.final_proj(x)