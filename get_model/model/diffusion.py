import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchmetrics
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig
from torch.nn.init import trunc_normal_

from get_model.model.modules import (
    BaseConfig, BaseModule, RegionEmbed, RegionEmbedConfig
)
from get_model.model.position_encoding import AbsolutePositionalEncoding
from get_model.model.model import BaseGETModel, BaseGETModelConfig, GETLoss, RegressionMetrics

# -----------------------------------------------------------------------------
# Core Diffusion Modules (Embeddings & Blocks)
# -----------------------------------------------------------------------------

def modulate(x, shift, scale):
    """
    Modulate the input x using the shift and scale parameters from adaLN.
    x: (N, L, D)
    shift, scale: (N, D) -> broadcast to (N, 1, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Attention(nn.Module):
    """
    Standard Self-Attention Module.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    From transformer.py
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class DiTBlock(nn.Module):
    """
    A Diffusion Transformer Block with Adaptive Layer Norm (adaLN).
    Conditioning (timestep embedding) is injected via adaLN.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
        
        # adaLN modulation: regress 6 parameters (shift, scale, gate) for 2 sub-blocks
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # c is the timestep embedding: (N, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # 1. Self-Attention Block
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # 2. Feed-Forward Block
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# -----------------------------------------------------------------------------
# Diffusion Transformer for Region-based Models
# -----------------------------------------------------------------------------

class GETRegionDiTEncoder(nn.Module):
    """
    DiT-style Transformer encoder for region-based inputs.
    Similar to GETTransformer but with timestep conditioning via adaLN.
    """
    def __init__(self, 
                 embed_dim: int = 768, 
                 depth: int = 12, 
                 num_heads: int = 12, 
                 mlp_ratio: float = 4.0, 
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Time Embedding
        self.t_embedder = TimestepEmbedder(embed_dim)

        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final Layer Norm (with affine for final output)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                # Only init if affine parameters exist
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)

        # Zero-out adaLN modulation layers for identity init (training stability)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t):
        """
        x: (B, N, embed_dim) - embedded region features
        t: (B,) - timestep indices
        Returns: (B, N, embed_dim)
        """
        # Embed timestep
        c = self.t_embedder(t)  # (B, embed_dim)

        # Apply DiT blocks
        for blk in self.blocks:
            x = blk(x, c)

        x = self.norm(x)
        return x


# -----------------------------------------------------------------------------
# Main Model: GETRegionDiffusion (matches GETRegionPretrain structure)
# -----------------------------------------------------------------------------

@dataclass
class GETRegionDiffusionConfig(BaseGETModelConfig):
    """
    Configuration for region-based diffusion model.
    Matches GETRegionPretrain structure but uses diffusion for masked prediction.
    """
    num_regions: int = 900
    num_motif: int = 283
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    output_dim: int = 283  # Predict masked motif features
    flash_attn: bool = False
    pool_method: str = 'mean'
    
    # Region embedding config
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    
    # Encoder config (for compatibility, but we use DiT blocks)
    encoder: dict = field(default_factory=lambda: {
        'num_heads': 12,
        'embed_dim': 768,
        'num_layers': 12,
        'drop_path_rate': 0.1,
        'drop_rate': 0,
        'attn_drop_rate': 0,
        'use_mean_pooling': False,
        'flash_attn': False
    })
    
    # Head for masked prediction
    head_mask: dict = field(default_factory=lambda: {'in_features': 768, 'out_features': 283})
    
    # Mask token config
    mask_token: dict = field(default_factory=lambda: {'embed_dim': 768, 'std': 0.02})
    
    # Diffusion config
    diffusion: dict = field(default_factory=lambda: {
        'num_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'hidden_dim': 256
    })


class GETRegionDiffusion(BaseGETModel):
    """
    Region-based Diffusion Model for masked motif prediction.
    
    Architecture mirrors GETRegionPretrain:
    - Input: region_motif (B, N, num_motif)
    - RegionEmbed -> DiT Encoder with timestep conditioning -> Head
    
    Training: Diffusion-based denoising of masked regions.
    """
    def __init__(self, cfg: GETRegionDiffusionConfig):
        super().__init__(cfg)
        
        # 1. Region Embedding (same as GETRegionPretrain)
        self.region_embed = RegionEmbed(cfg.region_embed)
        
        # 2. DiT-style Encoder with timestep conditioning
        self.encoder = GETRegionDiTEncoder(
            embed_dim=cfg.embed_dim,
            depth=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_ratio=4.0,
            dropout=cfg.dropout
        )
        
        # 3. Prediction head
        self.head_mask = nn.Linear(**cfg.head_mask)
        
        # 4. Mask and CLS tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.mask_token['embed_dim']))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        trunc_normal_(self.mask_token, std=cfg.mask_token['std'])
        trunc_normal_(self.cls_token, std=0.02)
        
        # 5. Diffusion schedule
        self._setup_diffusion(cfg.diffusion)
        
        # Note: Don't call self.apply(self._init_weights) here because
        # the encoder already initializes its weights, and we override _init_weights below

    def _init_weights(self, m):
        """
        Override BaseGETModel._init_weights to handle LayerNorm with elementwise_affine=False.
        DiT uses LayerNorm without affine parameters for adaLN.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Only initialize if affine parameters exist
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _setup_diffusion(self, diff_cfg):
        """Setup noise schedule for diffusion."""
        num_timesteps = diff_cfg.get('num_timesteps', 1000)
        beta_start = diff_cfg.get('beta_start', 0.0001)
        beta_end = diff_cfg.get('beta_end', 0.02)
        
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.num_timesteps = num_timesteps

    def get_input(self, batch):
        """Get input from batch - matches GETRegionPretrain."""
        return {
            'region_motif': batch['region_motif'],
            'mask': batch['mask'].unsqueeze(-1).bool()
        }

    def forward(self, region_motif, mask):
        """
        Forward pass with diffusion-based masked prediction.
        
        Args:
            region_motif: (B, N, num_motif) - Input motif features
            mask: (B, N, 1) - Boolean mask for which regions to predict
        
        Returns:
            x_masked: predictions for masked positions
            region_motif: original input
            mask: the mask used
        """
        # 1. Embed regions
        x = self.region_embed(region_motif)
        B, N, C = x.shape
        
        # 2. Apply mask tokens to masked positions
        cls_tokens = self.cls_token.expand(B, -1, -1)
        mask_token = self.mask_token.expand(B, N, -1)
        w = mask.type_as(mask_token)
        x = x * (1 - w) + mask_token * w
        
        # 3. Add CLS token
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 4. Sample timestep and run through DiT encoder
        device = x.device
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        
        # Run encoder with timestep conditioning
        x = self.encoder(x, t)
        
        # 5. Get predictions for masked positions
        x = x[:, 1:]  # Remove CLS token
        x_masked = self.head_mask(x)
        
        return x_masked, region_motif, mask

    def before_loss(self, output, batch):
        """Prepare predictions and targets for loss computation."""
        x_masked, x_original, mask = output
        
        # Apply mask to both pred and target
        pred = {'masked': x_masked * mask}
        obs = {'masked': x_original * mask}
        return pred, obs

    def generate_dummy_data(self):
        """Generate dummy data for testing."""
        B, R, M = 2, 900, 283
        return {
            'region_motif': torch.randn(B, R, M).float().abs(),
            'mask': torch.randint(0, 2, (B, R)).bool()
        }

    @torch.no_grad()
    def predict(self, batch, num_steps=50):
        """
        Inference: iterative denoising for generation.
        Uses fewer steps than training for efficiency.
        """
        self.eval()
        region_motif = batch['region_motif']
        mask = batch['mask'].unsqueeze(-1).bool()
        
        # Encode unmasked regions
        x = self.region_embed(region_motif)
        B, N, C = x.shape
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        mask_token = self.mask_token.expand(B, N, -1)
        w = mask.type_as(mask_token)
        x = x * (1 - w) + mask_token * w
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Use a subset of timesteps for faster inference
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]
        
        for t_val in timesteps:
            t = torch.full((B,), t_val, device=x.device, dtype=torch.long)
            x_out = self.encoder(x, t)
        
        # Final prediction
        x_out = x_out[:, 1:]
        predictions = self.head_mask(x_out)
        
        # Merge with original (keep unmasked, use predictions for masked)
        output = region_motif.clone()
        output[mask.squeeze(-1)] = predictions[mask.squeeze(-1)]
        
        return output


# Alias for backward compatibility with config
GETDiffusion = GETRegionDiffusion
