"""
Phase 2: Texture Refinement Runner
===================================

Runner class for training Phase 2 Texture Refinement Module.
Loads frozen Phase 1 LBBDM, generates coarse RGB, then trains refinement module.

Training Flow:
    1. Load frozen Phase 1 LBBDM (no grad)
    2. For each batch:
        - Generate coarse RGB from LBBDM
        - Pass coarse_rgb + sar_raw + edge_map to TRM
        - Compute loss (L1 + Perceptual)
        - Backprop through TRM only
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.autonotebook import tqdm
import torchvision.utils as vutils

from Register import Registers
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from model.RefinementModule.refinement import TextureRefinementModule, TextureRefinementModuleLite
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel


# =====================================================
# SSIM Loss - Better structure preservation than L1
# =====================================================
class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""
    
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)
        
    def _create_window(self, window_size, channel):
        """Create Gaussian window for SSIM"""
        import math
        sigma = 1.5
        gauss = torch.tensor([
            math.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, pred, target):
        """Compute 1 - SSIM as loss (higher SSIM = lower loss)"""
        # Move window to same device
        window = self.window.to(pred.device).type_as(pred)
        
        # Normalize to [0, 1]
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        
        mu1 = F.conv2d(pred, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(target, window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()


# =====================================================
# Gradient Loss - Preserve edges and texture details
# =====================================================
class GradientLoss(nn.Module):
    """Edge-preserving gradient loss using Sobel filters"""
    
    def __init__(self):
        super().__init__()
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
    
    def forward(self, pred, target):
        # Compute gradients for prediction
        pred_grad_x = F.conv2d(pred, self.sobel_x.to(pred.device), padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, self.sobel_y.to(pred.device), padding=1, groups=3)
        
        # Compute gradients for target
        target_grad_x = F.conv2d(target, self.sobel_x.to(target.device), padding=1, groups=3)
        target_grad_y = F.conv2d(target, self.sobel_y.to(target.device), padding=1, groups=3)
        
        # L1 loss on gradients
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y


# =====================================================
# Multi-Layer Perceptual Loss - Richer feature matching
# =====================================================
class MultiLayerPerceptualLoss(nn.Module):
    """
    Multi-layer VGG Perceptual Loss
    Uses features from multiple VGG layers for richer texture matching
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features
        
        # Extract features at multiple layers
        # relu1_2, relu2_2, relu3_4, relu4_4, relu5_4
        self.layer_indices = [3, 8, 17, 26, 35]
        self.layer_weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # Equal weights
        
        self.layers = nn.ModuleList()
        prev_idx = 0
        for idx in self.layer_indices:
            self.layers.append(nn.Sequential(*list(vgg.children())[prev_idx:idx+1]))
            prev_idx = idx + 1
        
        self.layers = self.layers.to(device).eval()
        
        for param in self.parameters():
            param.requires_grad = False
            
        # VGG normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        x = (x + 1.0) / 2.0
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std
    
    def forward(self, pred, target):
        pred = self.normalize(pred)
        target = self.normalize(target)
        
        total_loss = 0.0
        pred_feat = pred
        target_feat = target
        
        for i, layer in enumerate(self.layers):
            pred_feat = layer(pred_feat)
            target_feat = layer(target_feat)
            total_loss += self.layer_weights[i] * F.l1_loss(pred_feat, target_feat)
        
        return total_loss / len(self.layers)


# =====================================================
# Charbonnier Loss - Smooth L1 variant, better for details
# =====================================================
class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (smooth L1)
    Better than L1 for fine details - less penalty for small errors
    L(x) = sqrt(x^2 + eps^2)
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


class PerceptualLoss(nn.Module):
    """VGG-based Perceptual Loss for texture refinement"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features
        
        # Use features up to relu3_4 (layer 16)
        self.layers = nn.Sequential(*list(vgg.children())[:16]).to(device).eval()
        
        for param in self.layers.parameters():
            param.requires_grad = False
            
        # VGG normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """Normalize from [-1, 1] to VGG input range"""
        x = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        # Move mean/std to same device as input
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std
    
    def forward(self, pred, target):
        """Compute perceptual loss between pred and target"""
        pred_feat = self.layers(self.normalize(pred))
        target_feat = self.layers(self.normalize(target))
        return F.l1_loss(pred_feat, target_feat)


@Registers.runners.register_with_name('RefinementRunner')
class RefinementRunner(DiffusionBaseRunner):
    """
    Runner for Phase 2 Texture Refinement training.
    """
    
    def __init__(self, config):
        # Initialize these BEFORE calling super().__init__ 
        # because super() calls initialize_model which sets these
        self.phase1_model = None
        self.perceptual_loss = None
        super().__init__(config)
    
    def initialize_model(self, config):
        """Initialize Phase 2 Refinement Module"""
        
        # =====================================================
        # 1. Load frozen Phase 1 LBBDM
        # =====================================================
        if hasattr(config.model, 'phase1_model_path') and config.model.phase1_model_path:
            print(f"[RefinementRunner] Loading Phase 1 LBBDM from: {config.model.phase1_model_path}")
            
            # Load Phase 1 config (assuming similar structure)
            phase1_model = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
            
            # Load Phase 1 weights
            phase1_states = torch.load(config.model.phase1_model_path, map_location='cpu')
            if 'model' in phase1_states:
                phase1_model.load_state_dict(phase1_states['model'], strict=False)
            else:
                phase1_model.load_state_dict(phase1_states, strict=False)
            
            # Freeze Phase 1
            for param in phase1_model.parameters():
                param.requires_grad = False
            phase1_model.eval()
            
            self.phase1_model = phase1_model
            print("[RefinementRunner] Phase 1 LBBDM loaded and frozen")
        else:
            print("[RefinementRunner] WARNING: No Phase 1 model path provided!")
            self.phase1_model = None
        
        # =====================================================
        # 2. Initialize Phase 2 Refinement Module
        # =====================================================
        refinement_config = getattr(config.model, 'refinement', None)
        if refinement_config is None:
            # Default config if not provided
            use_lite = False
            base_dim = 64
            n_marms = 3
            n_res_blocks = 4
        else:
            use_lite = getattr(refinement_config, 'use_lite', False)
            base_dim = getattr(refinement_config, 'base_dim', 64 if not use_lite else 32)
            n_marms = getattr(refinement_config, 'n_marms', 3 if not use_lite else 2)
            n_res_blocks = getattr(refinement_config, 'n_res_blocks', 4 if not use_lite else 2)
        
        if use_lite:
            net = TextureRefinementModuleLite(
                rgb_channels=3,
                sar_channels=1,
                edge_channels=1,
                base_dim=base_dim,
                n_marms=n_marms,
                n_res_blocks=n_res_blocks,
            ).to(config.training.device[0])
        else:
            net = TextureRefinementModule(
                rgb_channels=3,
                sar_channels=1,
                edge_channels=1,
                base_dim=base_dim,
                n_marms=n_marms,
                n_res_blocks=n_res_blocks,
            ).to(config.training.device[0])
        
        net.apply(weights_init)
        
        # =====================================================
        # 3. Initialize Loss Functions
        # =====================================================
        device = config.training.device[0]
        
        # Choose loss type from config
        loss_type = getattr(config.model, 'loss_type', 'advanced')
        
        if loss_type == 'advanced':
            # Advanced multi-component loss
            self.perceptual_loss = MultiLayerPerceptualLoss(device=device)
            self.ssim_loss = SSIMLoss()
            self.gradient_loss = GradientLoss()
            self.charbonnier_loss = CharbonnierLoss()
            print("[RefinementRunner] Using ADVANCED loss: Charbonnier + SSIM + MultiPerceptual + Gradient")
        else:
            # Original simple loss
            self.perceptual_loss = PerceptualLoss(device=device)
            self.ssim_loss = None
            self.gradient_loss = None
            self.charbonnier_loss = None
            print("[RefinementRunner] Using SIMPLE loss: L1 + Perceptual")
        
        return net
    
    def print_model_summary(self, net):
        """Print model parameter summary"""
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("=" * 50)
        print(f"[Phase 2 TRM] Total Parameters: {total_num / 1e6:.2f}M")
        print(f"[Phase 2 TRM] Trainable Parameters: {trainable_num / 1e6:.2f}M")
        
        if self.phase1_model is not None:
            total_p1, trainable_p1 = get_parameter_number(self.phase1_model)
            print(f"[Phase 1 LBBDM] Total Parameters: {total_p1 / 1e6:.2f}M (frozen)")
        print("=" * 50)
    
    def initialize_optimizer_scheduler(self, net, config):
        """Initialize optimizer and scheduler for TRM"""
        optimizer = get_optimizer(config.model.optimizer, net.parameters())
        print(f'[RefinementRunner] Optimizer: {config.model.optimizer.optimizer}')
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.training.n_epochs,
            eta_min=1e-6
        )
        return [optimizer], [scheduler]
    
    def _unpack_batch(self, batch):
        """
        Unpack batch data.
        
        Supports two formats:
        - 4 elements: (optical, sar, lc_label, edge_map) - on-the-fly generation
        - 5 elements: (optical, sar, lc_label, edge_map, coarse_rgb) - cached mode
        """
        x = x_name = x_cond_sar = x_cond_sar_name = x_cond_lc = x_cond_edge = coarse_rgb = None

        if not (isinstance(batch, (list, tuple)) and len(batch) >= 3):
            if isinstance(batch, (list, tuple)):
                first = batch[0]
                second = batch[1] if len(batch) > 1 else None
            else:
                x = batch; x_name = None
                return x, x_name, None, None, None, x_cond_name, None
            
            if isinstance(first, (list, tuple)): x = first[0]; x_name = first[1]
            else: x = first; x_name = None
            
            if isinstance(second, (list, tuple)): x_cond_sar = second[0]; x_cond_name = second[1]
            else: x_cond_sar = second; x_cond_name = None
            
            return x, x_name, x_cond_sar, None, None, x_cond_name, None

        if isinstance(batch[0], (list, tuple)): x = batch[0][0]; x_name = batch[0][1]
        else: x = batch[0]; x_name = None

        if isinstance(batch[1], (list, tuple)): x_cond_sar = batch[1][0]; x_cond_sar_name = batch[1][1]
        else: x_cond_sar = batch[1]; x_cond_sar_name = None

        if isinstance(batch[2], (list, tuple)): x_cond_lc = batch[2][0]
        else: x_cond_lc = batch[2]
        
        if len(batch) >= 4:
            if isinstance(batch[3], (list, tuple)): x_cond_edge = batch[3][0]
            else: x_cond_edge = batch[3]
        
        # Check for cached coarse_rgb (5th element)
        if len(batch) >= 5:
            if isinstance(batch[4], (list, tuple)): coarse_rgb = batch[4][0]
            else: coarse_rgb = batch[4]
            
        return x, x_name, x_cond_sar, x_cond_lc, x_cond_edge, x_cond_sar_name, coarse_rgb
    
    @torch.no_grad()
    def generate_coarse_rgb(self, x_cond_sar, x_cond_lc, x_cond_edge):
        """Generate coarse RGB from Phase 1 LBBDM (no grad)"""
        if self.phase1_model is None:
            raise ValueError("Phase 1 model not loaded!")
        
        self.phase1_model.eval()
        coarse_rgb = self.phase1_model.sample(
            x_cond_sar, x_cond_lc, x_cond_edge,
            clip_denoised=True
        )
        return coarse_rgb
    
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        """
        Compute refinement loss.
        
        Loss = λ1 * L1_loss + λ2 * Perceptual_loss
        
        Supports two modes:
        - Cached mode: coarse_rgb loaded from disk (fast)
        - On-the-fly mode: coarse_rgb generated from Phase 1 (slow)
        """
        x, x_name, x_cond_sar, x_cond_lc, x_cond_edge, x_cond_name, coarse_rgb_cached = self._unpack_batch(batch)

        # Move to device
        x = x.to(self.config.training.device[0])  # Ground truth RGB
        x_cond_sar = x_cond_sar.to(self.config.training.device[0])
        
        if x_cond_lc is None:
            x_cond_lc = torch.zeros_like(x_cond_sar)[:, 0, :, :].long()
        else:
            x_cond_lc = x_cond_lc.to(self.config.training.device[0])
        
        if x_cond_edge is not None:
            x_cond_edge = x_cond_edge.to(self.config.training.device[0])
        else:
            # Create dummy edge map if not available
            x_cond_edge = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], 
                                      device=self.config.training.device[0])
        
        # =====================================================
        # 1. Get coarse RGB (cached or generated)
        # =====================================================
        if coarse_rgb_cached is not None:
            # Cached mode: use pre-generated coarse RGB
            coarse_rgb = coarse_rgb_cached.to(self.config.training.device[0])
        else:
            # On-the-fly mode: generate from Phase 1 (slow)
            coarse_rgb = self.generate_coarse_rgb(x_cond_sar, x_cond_lc, x_cond_edge)
        
        # =====================================================
        # 2. Prepare SAR for refinement (use single channel)
        # =====================================================
        if x_cond_sar.shape[1] == 3:
            sar_raw = x_cond_sar[:, 0:1, :, :]  # Take first channel
        else:
            sar_raw = x_cond_sar
        
        # =====================================================
        # 3. Pass through Refinement Module
        # =====================================================
        refined_rgb = net(coarse_rgb, sar_raw, x_cond_edge)
        
        # =====================================================
        # 4. Compute Losses
        # =====================================================
        loss_type = getattr(self.config.model, 'loss_type', 'advanced')
        
        if loss_type == 'advanced' and self.charbonnier_loss is not None:
            # =====================================================
            # ADVANCED LOSS FUNCTION
            # =====================================================
            
            # Charbonnier Loss (smooth L1, better for details)
            charbonnier_loss = self.charbonnier_loss(refined_rgb, x)
            
            # SSIM Loss (structural similarity)
            ssim_loss = self.ssim_loss(refined_rgb, x)
            
            # Multi-layer Perceptual Loss
            perceptual_loss = self.perceptual_loss(refined_rgb, x)
            
            # Gradient Loss (edge preservation)
            gradient_loss = self.gradient_loss(refined_rgb, x)
            
            # Color Histogram Loss - match color distribution with GT (not coarse!)
            # This encourages learning GT colors, not just preserving coarse colors
            refined_mean = refined_rgb.mean(dim=[2, 3])  # (B, 3)
            target_mean = x.mean(dim=[2, 3])             # (B, 3) - use GT!
            refined_std = refined_rgb.std(dim=[2, 3])
            target_std = x.std(dim=[2, 3])
            color_loss = F.l1_loss(refined_mean, target_mean) + F.l1_loss(refined_std, target_std)
            
            # Get weights from config
            lambda_charb = getattr(self.config.model, 'lambda_charbonnier', 1.0)
            lambda_ssim = getattr(self.config.model, 'lambda_ssim', 0.5)
            lambda_perceptual = getattr(self.config.model, 'lambda_perceptual', 0.1)
            lambda_gradient = getattr(self.config.model, 'lambda_gradient', 0.1)
            lambda_color = getattr(self.config.model, 'lambda_color', 0.2)
            
            total_loss = (lambda_charb * charbonnier_loss + 
                         lambda_ssim * ssim_loss + 
                         lambda_perceptual * perceptual_loss + 
                         lambda_gradient * gradient_loss +
                         lambda_color * color_loss)
            
            # Logging
            if write:
                self.writer.add_scalar(f'loss/{stage}', total_loss, step)
                self.writer.add_scalar(f'charbonnier_loss/{stage}', charbonnier_loss, step)
                self.writer.add_scalar(f'ssim_loss/{stage}', ssim_loss, step)
                self.writer.add_scalar(f'perceptual_loss/{stage}', perceptual_loss, step)
                self.writer.add_scalar(f'gradient_loss/{stage}', gradient_loss, step)
                self.writer.add_scalar(f'color_loss/{stage}', color_loss, step)
        else:
            # =====================================================
            # SIMPLE LOSS FUNCTION (original)
            # =====================================================
            l1_loss = F.l1_loss(refined_rgb, x)
            perceptual_loss = self.perceptual_loss(refined_rgb, x)
            
            # Color consistency with coarse
            refined_mean = refined_rgb.mean(dim=[2, 3])
            coarse_mean = coarse_rgb.mean(dim=[2, 3])
            color_loss = F.l1_loss(refined_mean, coarse_mean)
            
            lambda_l1 = getattr(self.config.model, 'lambda_l1', 1.0)
            lambda_perceptual = getattr(self.config.model, 'lambda_perceptual', 0.1)
            lambda_color = getattr(self.config.model, 'lambda_color', 0.5)
            
            total_loss = lambda_l1 * l1_loss + lambda_perceptual * perceptual_loss + lambda_color * color_loss
            
            if write:
                self.writer.add_scalar(f'loss/{stage}', total_loss, step)
                self.writer.add_scalar(f'l1_loss/{stage}', l1_loss, step)
                self.writer.add_scalar(f'perceptual_loss/{stage}', perceptual_loss, step)
                self.writer.add_scalar(f'color_loss/{stage}', color_loss, step)
        
        return total_loss
    
    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        """Generate and save sample images"""
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        
        x, x_name, x_cond_sar, x_cond_lc, x_cond_edge, x_cond_name, coarse_rgb_cached = self._unpack_batch(batch)

        batch_size = min(x.shape[0], 4)

        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond_sar = x_cond_sar[0:batch_size].to(self.config.training.device[0])
        
        if x_cond_lc is None:
            x_cond_lc = torch.zeros_like(x_cond_sar)[:, 0, :, :].long()
        else:
            x_cond_lc = x_cond_lc[0:batch_size].to(self.config.training.device[0])
        
        if x_cond_edge is not None:
            x_cond_edge = x_cond_edge[0:batch_size].to(self.config.training.device[0])
        else:
            x_cond_edge = torch.zeros(batch_size, 1, x.shape[2], x.shape[3], 
                                      device=self.config.training.device[0])

        grid_size = 4
        to_normal = self.config.data.dataset_config.to_normal

        # Get coarse RGB (cached or generated)
        if coarse_rgb_cached is not None:
            coarse_rgb = coarse_rgb_cached[0:batch_size].to(self.config.training.device[0])
        else:
            coarse_rgb = self.generate_coarse_rgb(x_cond_sar, x_cond_lc, x_cond_edge)
        
        # Prepare SAR input
        if x_cond_sar.shape[1] == 3:
            sar_raw = x_cond_sar[:, 0:1, :, :]
        else:
            sar_raw = x_cond_sar
        
        # Generate refined RGB
        net.eval()
        refined_rgb = net(coarse_rgb, sar_raw, x_cond_edge)
        net.train()

        # Save images
        # 1. Ground Truth
        image_grid = get_image_grid(x.cpu(), grid_size, to_normal=to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')

        # 2. SAR Condition
        image_grid_sar = get_image_grid(x_cond_sar.cpu(), grid_size, to_normal=to_normal)
        im = Image.fromarray(image_grid_sar)
        im.save(os.path.join(sample_path, 'condition_sar.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition_sar', image_grid_sar, self.global_step, dataformats='HWC')

        # 3. Coarse RGB (Phase 1 output)
        image_grid_coarse = get_image_grid(coarse_rgb.cpu(), grid_size, to_normal=to_normal)
        im = Image.fromarray(image_grid_coarse)
        im.save(os.path.join(sample_path, 'coarse_rgb.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_coarse_rgb', image_grid_coarse, self.global_step, dataformats='HWC')

        # 4. Refined RGB (Phase 2 output)
        image_grid_refined = get_image_grid(refined_rgb.cpu(), grid_size, to_normal=to_normal)
        im = Image.fromarray(image_grid_refined)
        im.save(os.path.join(sample_path, 'refined_rgb.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_refined_rgb', image_grid_refined, self.global_step, dataformats='HWC')

        # 5. Edge Map
        if x_cond_edge is not None:
            edge_vis = x_cond_edge.repeat(1, 3, 1, 1)
            edge_vis = (edge_vis - 0.5) * 2.0
            image_grid_edge = get_image_grid(edge_vis.cpu(), grid_size, to_normal=to_normal)
            im = Image.fromarray(image_grid_edge)
            im.save(os.path.join(sample_path, 'condition_edge.png'))
            if stage != 'test':
                self.writer.add_image(f'{stage}_condition_edge', image_grid_edge, self.global_step, dataformats='HWC')

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        """Generate samples for evaluation"""
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))
        coarse_path = make_dir(os.path.join(sample_path, 'coarse'))
        refined_path = make_dir(os.path.join(sample_path, 'refined'))
        sar_path = make_dir(os.path.join(sample_path, 'sar'))

        to_normal = self.config.data.dataset_config.to_normal
        
        net.eval()
        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        
        for idx, test_batch in enumerate(pbar):
            x, x_name, x_cond_sar, x_cond_lc, x_cond_edge, _, coarse_rgb_cached = self._unpack_batch(test_batch)

            x = x.to(self.config.training.device[0])
            x_cond_sar = x_cond_sar.to(self.config.training.device[0])
            
            if x_cond_lc is None:
                x_cond_lc = torch.zeros_like(x_cond_sar)[:, 0, :, :].long()
            else:
                x_cond_lc = x_cond_lc.to(self.config.training.device[0])
            
            if x_cond_edge is not None:
                x_cond_edge = x_cond_edge.to(self.config.training.device[0])
            else:
                x_cond_edge = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], 
                                          device=self.config.training.device[0])

            # Get coarse RGB (cached or generated)
            if coarse_rgb_cached is not None:
                coarse_rgb = coarse_rgb_cached.to(self.config.training.device[0])
            else:
                coarse_rgb = self.generate_coarse_rgb(x_cond_sar, x_cond_lc, x_cond_edge)
            
            if x_cond_sar.shape[1] == 3:
                sar_raw = x_cond_sar[:, 0:1, :, :]
            else:
                sar_raw = x_cond_sar
            
            refined_rgb = net(coarse_rgb, sar_raw, x_cond_edge)

            # Save images
            batch_size = x.shape[0]
            for i in range(batch_size):
                name = x_name[i] if x_name is not None else f"sample_{idx * batch_size + i}"
                
                save_single_image(x[i], gt_path, f'{name}.png', to_normal=to_normal)
                save_single_image(coarse_rgb[i], coarse_path, f'{name}.png', to_normal=to_normal)
                save_single_image(refined_rgb[i], refined_path, f'{name}.png', to_normal=to_normal)
                save_single_image(x_cond_sar[i], sar_path, f'{name}.png', to_normal=to_normal)
