import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.BrownianBridge.base.modules.marm import MARMConditionModel 
from model.VQGAN.vqgan import VQModel

def disabled_train(self, mode=True): return self

class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.vqvae = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqvae.train = disabled_train
        for param in self.vqvae.parameters(): param.requires_grad = False
        print(f"Loaded VQ-VAE from {model_config.VQGAN.params.ckpt_path}")

        if self.condition_key == 'cross_attention':
            self.num_classes = getattr(model_config.CondStageParams, 'num_classes', 11)
            self.embed_dim = getattr(model_config.CondStageParams, 'embed_dim', 64)
            self.context_out_dim = getattr(model_config.CondStageParams, 'out_channels', 128)

            # Backward-compatible switches (default False)
            # - use_onehot: sharper semantics (hard class separation)
            # - use_boundary: add boundary/edge cue from LC label map
            # - use_dual_marm: add MARM branch for raw SAR + fusion layer
            self.lc_use_onehot = bool(getattr(model_config.CondStageParams, 'use_onehot', False))
            self.lc_use_boundary = bool(getattr(model_config.CondStageParams, 'use_boundary', False))
            self.use_dual_marm = bool(getattr(model_config.CondStageParams, 'use_dual_marm', False))

            # --- LC Branch (MARM_LC) ---
            if self.lc_use_onehot:
                self.lc_proj = nn.Conv2d(self.num_classes, self.embed_dim, kernel_size=1)
                self.lc_embedding = None
            else:
                self.lc_embedding = nn.Embedding(self.num_classes, self.embed_dim)
                self.lc_proj = None

            stem_in = self.embed_dim + (1 if self.lc_use_boundary else 0)
            self.lc_stem = nn.Sequential(
                nn.Conv2d(stem_in, self.embed_dim, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1),
            )

            # MARM for LC context
            self.cond_stage_model = MARMConditionModel(**vars(model_config.CondStageParams))

            # --- SAR Branch (MARM_SAR) + Fusion Layer (Dual-MARM mode) ---
            if self.use_dual_marm:
                print("[Dual-MARM] Initializing MARM_SAR branch for raw SAR image...")
                # SAR MARM: input is raw SAR image (3 channels), output is context_out_dim
                sar_marm_params = {
                    'in_channels': 3,  # Raw SAR RGB
                    'out_channels': self.context_out_dim,
                    'base_dim': getattr(model_config.CondStageParams, 'base_dim', 64),
                    'n_marms': getattr(model_config.CondStageParams, 'n_marms', 4),
                }
                self.marm_sar = MARMConditionModel(**sar_marm_params)

                # Fusion Layer: Concat(256) -> Conv1x1 -> ReLU -> (128)
                self.fusion_proj = nn.Sequential(
                    nn.Conv2d(self.context_out_dim * 2, self.context_out_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
                print(f"[Dual-MARM] Fusion: {self.context_out_dim * 2} -> {self.context_out_dim}")
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqvae
        else:
            self.cond_stage_model = None

    def get_parameters(self):
        if self.condition_key == 'cross_attention':
            parts = [
                self.denoise_fn.parameters(),
                self.cond_stage_model.parameters(),
                self.lc_stem.parameters(),
            ]
            if getattr(self, 'lc_embedding', None) is not None:
                parts.append(self.lc_embedding.parameters())
            if getattr(self, 'lc_proj', None) is not None:
                parts.append(self.lc_proj.parameters())
            # Dual-MARM: add MARM_SAR and fusion_proj
            if getattr(self, 'use_dual_marm', False):
                parts.append(self.marm_sar.parameters())
                parts.append(self.fusion_proj.parameters())
            return itertools.chain(*parts)
        elif self.condition_key == 'SpatialRescaler':
            return itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        return self.denoise_fn.parameters()

    def forward(self, x, x_cond_sar, x_cond_herringbone, context=None):
        with torch.no_grad():
            x_latent = self.encode(x, cond=False)
            x_cond_sar_latent = self.encode(x_cond_sar, cond=True)

        # Get fused context (LC + optionally SAR via Dual-MARM)
        context_guidance = self.get_cond_stage_context(x_cond_herringbone, x_cond_sar)
        return super().forward(x_latent.detach(), x_cond_sar_latent.detach(), context=context_guidance)

    def get_cond_stage_context(self, x_cond_lc, x_cond_sar_raw=None):
        """
        Compute context for cross-attention.
        
        Args:
            x_cond_lc: LC label map (B, H, W) long tensor
            x_cond_sar_raw: Raw SAR image (B, 3, H, W) float tensor, only used if use_dual_marm=True
        
        Returns:
            context: (B, context_dim, H', W') tensor for UNet cross-attention
        """
        if self.cond_stage_model is None:
            return None

        if x_cond_lc.dim() == 4:
            x_cond_lc = x_cond_lc.squeeze(1)

        x_cond_lc = x_cond_lc.long()
        if hasattr(self, 'num_classes'):
            x_cond_lc = torch.clamp(x_cond_lc, 0, self.num_classes - 1)

        if getattr(self, 'condition_key', None) == 'cross_attention':
            # --- LC Branch ---
            if getattr(self, 'lc_use_onehot', False):
                # (B,H,W) -> (B,num_classes,H,W)
                x_onehot = F.one_hot(x_cond_lc, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
                x_feat = self.lc_proj(x_onehot)
            else:
                # embedding: (B,H,W) -> (B,C,H,W)
                x_feat = self.lc_embedding(x_cond_lc).permute(0, 3, 1, 2)

            if getattr(self, 'lc_use_boundary', False):
                # simple boundary cue from label changes (B,1,H,W)
                bnd = torch.zeros((x_cond_lc.shape[0], 1, x_cond_lc.shape[1], x_cond_lc.shape[2]), device=x_cond_lc.device)
                bnd[:, :, :, 1:] |= (x_cond_lc[:, :, 1:] != x_cond_lc[:, :, :-1]).unsqueeze(1)
                bnd[:, :, 1:, :] |= (x_cond_lc[:, 1:, :] != x_cond_lc[:, :-1, :]).unsqueeze(1)
                x_feat = torch.cat([x_feat, bnd.float()], dim=1)

            x_feat = self.lc_stem(x_feat)
            context_lc = self.cond_stage_model(x_feat)  # (B, 128, H', W')

            # --- Dual-MARM: SAR Branch + Fusion ---
            if getattr(self, 'use_dual_marm', False) and x_cond_sar_raw is not None:
                # Normalize SAR from [-1,1] to [0,1] for MARM_SAR (designed for [0,1] input)
                x_sar_normalized = (x_cond_sar_raw + 1.0) / 2.0
                context_sar = self.marm_sar(x_sar_normalized)  # (B, 128, H', W')
                # Concat & Project
                context_fused = torch.cat([context_sar, context_lc], dim=1)  # (B, 256, H', W')
                context = self.fusion_proj(context_fused)  # (B, 128, H', W')
            else:
                context = context_lc
        else:
            context = self.cond_stage_model(x_cond_lc)

        if self.condition_key == 'first_stage':
            context = context.detach()
        return context

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        x_latent = self.vqvae.encoder(x)
        if not self.model_config.latent_before_quant_conv: x_latent = self.vqvae.quant_conv(x_latent)
        if normalize:
            if cond: x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            else: x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent

    # Ban hay danh gia phuong phap, neu no tot hon thi ap dung
    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        if normalize:
            if cond: x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            else: x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        if self.model_config.latent_before_quant_conv: x_latent = self.vqvae.quant_conv(x_latent)
        x_latent_quant, _, _ = self.vqvae.quantize(x_latent)
        return self.vqvae.decode(x_latent_quant)

    @torch.no_grad()
    def sample(self, x_cond_sar, x_cond_herringbone, clip_denoised=False, sample_mid_step=False):
        x_cond_sar_latent = self.encode(x_cond_sar, cond=True)
        # Pass raw SAR for Dual-MARM fusion
        context_guidance = self.get_cond_stage_context(x_cond_herringbone, x_cond_sar)
        b, _, h, w = x_cond_sar_latent.shape
        target_shape = (b, 3, h, w) 
        
        if sample_mid_step:
            temp, _ = self.p_sample_loop(y=x_cond_sar_latent, x_T_shape=target_shape, context=context_guidance, clip_denoised=clip_denoised, sample_mid_step=sample_mid_step)
            out_samples = []
            for i in tqdm(range(len(temp))):
                with torch.no_grad(): out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))
            return out_samples, []
        else:
            x_latent = self.p_sample_loop(y=x_cond_sar_latent, x_T_shape=target_shape, context=context_guidance, clip_denoised=clip_denoised, sample_mid_step=sample_mid_step)
            return self.decode(x_latent, cond=False)