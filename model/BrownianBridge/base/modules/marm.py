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