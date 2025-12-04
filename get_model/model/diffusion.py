import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchmetrics
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig
from torch.nn.init import trunc_normal_
from timm.models.layers import DropPath

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
    Standard Self-Attention Module (matching GETTransformer's Attention).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention_mask=None, attention_bias=None):
        B, N, C = x.shape
        
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attn_dtype = attn.dtype
            attn_mask_value = -65504 if attn_dtype in [torch.float16, torch.float32, torch.bfloat16] else -1e9
            attn = attn.masked_fill(attention_mask, attn_mask_value)
        
        # Apply attention bias if provided
        if attention_bias is not None:
            if attention_bias.dim() == 1:
                attention_bias = attention_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif attention_bias.dim() == 2:
                attention_bias = attention_bias.unsqueeze(1).unsqueeze(1)
            elif attention_bias.dim() == 3:
                attention_bias = attention_bias.unsqueeze(1)
            attn = attn + attention_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
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
    Enhanced with DropPath for better training stability (matching GETTransformer).
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0.0, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, 
            num_heads=num_heads, 
            qkv_bias=True, 
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size, 
            hidden_features=mlp_hidden_dim, 
            act_layer=nn.GELU,
            drop=proj_drop
        )
        
        # DropPath for stochastic depth (matching GETTransformer)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # adaLN modulation: regress 6 parameters (shift, scale, gate) for 2 sub-blocks
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attention_mask=None, attention_bias=None):
        """
        Args:
            x: (B, N, D) input features
            c: (B, D) timestep embedding
            attention_mask: optional attention mask
            attention_bias: optional attention bias
        """
        # c is the timestep embedding: (B, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # 1. Self-Attention Block with adaLN and DropPath
        attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attention_mask, attention_bias)
        x = x + self.drop_path(gate_msa.unsqueeze(1) * attn_out)
        
        # 2. Feed-Forward Block with adaLN and DropPath
        mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + self.drop_path(gate_mlp.unsqueeze(1) * mlp_out)
        return x


# -----------------------------------------------------------------------------
# Diffusion Transformer for Region-based Models
# -----------------------------------------------------------------------------

class GETRegionDiTEncoder(nn.Module):
    """
    DiT-style Transformer encoder for region-based inputs.
    Similar to GETTransformer but with timestep conditioning via adaLN.
    Enhanced with DropPath and proper attention mask support.
    """
    def __init__(self, 
                 embed_dim: int = 768, 
                 depth: int = 12, 
                 num_heads: int = 12, 
                 mlp_ratio: float = 4.0, 
                 dropout: float = 0.1,
                 drop_path_rate: float = 0.1,
                 attn_drop_rate: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Time Embedding
        self.t_embedder = TimestepEmbedder(embed_dim)

        # DropPath schedule (matching GETTransformer)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # DiT Blocks with DropPath
        self.blocks = nn.ModuleList([
            DiTBlock(
                embed_dim, 
                num_heads, 
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                attn_drop=attn_drop_rate,
                proj_drop=dropout
            )
            for i in range(depth)
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

    def forward(self, x, t, mask=None, bias=None):
        """
        x: (B, N, embed_dim) - embedded region features
        t: (B,) - timestep indices
        mask: optional attention mask (B, N) or (B, 1, N)
        bias: optional attention bias
        Returns: (B, N, embed_dim)
        """
        # Embed timestep
        c = self.t_embedder(t)  # (B, embed_dim)

        # Apply DiT blocks with attention mask support
        for blk in self.blocks:
            x = blk(x, c, attention_mask=mask, attention_bias=bias)

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
        # Use encoder config parameters if available, otherwise use defaults
        encoder_cfg = cfg.encoder if hasattr(cfg, 'encoder') and isinstance(cfg.encoder, dict) else {}
        self.encoder = GETRegionDiTEncoder(
            embed_dim=cfg.embed_dim,
            depth=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_ratio=4.0,
            dropout=cfg.dropout,
            drop_path_rate=encoder_cfg.get('drop_path_rate', 0.1),
            attn_drop_rate=encoder_cfg.get('attn_drop_rate', 0.0)
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
            predicted_noise: predicted noise for masked positions
            true_noise: true noise that was added
            region_motif: original input
            mask: the mask used
        """
        B, N, M = region_motif.shape
        device = region_motif.device
        
        # 1. Extract masked regions (targets for diffusion)
        mask_squeezed = mask.squeeze(-1)  # (B, N)
        num_masked_per_batch = mask_squeezed.sum(dim=1)  # (B,)
        total_masked = mask_squeezed.sum().item()
        
        if total_masked == 0:
            # No masked regions, return zeros
            return torch.zeros(B, N, M, device=device), torch.zeros(B, N, M, device=device), region_motif, mask, torch.zeros(0, dtype=torch.long, device=device)
        
        # Extract masked regions: (total_masked, M)
        masked_regions = region_motif[mask_squeezed]  # (total_masked, M)
        
        # 2. Sample timestep for each masked region
        t = torch.randint(0, self.num_timesteps, (total_masked,), device=device).long()
        
        # 3. Sample noise
        noise = torch.randn_like(masked_regions)  # (total_masked, M)
        
        # 4. Add noise to masked regions: q(x_t | x_0)
        # For each masked region, use its corresponding timestep
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)  # (total_masked, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)  # (total_masked, 1)
        
        noisy_masked_regions = sqrt_alpha_t * masked_regions + sqrt_one_minus_alpha_t * noise
        
        # 5. Create noisy input: replace masked regions with noisy versions
        noisy_region_motif = region_motif.clone()
        noisy_region_motif[mask_squeezed] = noisy_masked_regions
        
        # 6. Embed regions (including noisy masked ones)
        # CRITICAL: We keep the noisy embeddings for masked regions - don't replace with mask tokens!
        # The model needs to see the noisy data to learn to denoise it.
        x = self.region_embed(noisy_region_motif)
        C = x.shape[-1]
        
        # 7. Add CLS token (no mask tokens - we want the encoder to see noisy masked regions)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 8. Expand timesteps to batch level for encoder
        # Each sample in batch may have different number of masked regions
        # Use mean timestep per batch sample for better representation
        # (Each masked region has its own timestep, but encoder needs one per batch sample)
        t_batch = torch.zeros(B, dtype=torch.long, device=device)
        mask_idx = 0
        for b in range(B):
            if num_masked_per_batch[b] > 0:
                # Use mean timestep for this batch sample's masked regions
                t_start = mask_idx
                t_end = mask_idx + num_masked_per_batch[b]
                t_batch[b] = t[t_start:t_end].float().mean().long()
                mask_idx = t_end
        
        # 9. Run encoder with timestep conditioning
        # The encoder processes all regions (both clean unmasked and noisy masked)
        # Note: We use one timestep per batch sample, but each masked region has its own timestep
        # This is a limitation, but using mean timestep provides better conditioning than first timestep
        x = self.encoder(x, t_batch)
        
        # 10. Get predictions for all positions
        x = x[:, 1:]  # Remove CLS token
        x_all = self.head_mask(x)  # (B, N, M) - predictions for all regions
        
        # 11. Extract predictions only for masked positions (where we added noise)
        predicted_noise_masked = x_all[mask_squeezed]  # (total_masked, M)
        
        # 13. Reshape to match original structure for loss computation
        # We'll return flattened predictions and noise for masked regions only
        # Also return timesteps for metric computation (reconstructing x_0)
        return predicted_noise_masked, noise, region_motif, mask, t

    def before_loss(self, output, batch):
        """
        Prepare predictions and targets for loss computation.
        
        For diffusion models, we reconstruct x_0 from predicted noise to compute
        meaningful metrics (r2, pearson) comparing predictions vs actual values.
        The loss will be MSE(reconstructed_x0, actual), which is equivalent to
        predicting x_0 directly.
        
        Args:
            output: tuple of (predicted_noise_masked, true_noise, region_motif, mask, t)
                - predicted_noise_masked: (total_masked, M) - predicted noise for masked regions
                - true_noise: (total_masked, M) - true noise that was added
                - region_motif: (B, N, M) - original input
                - mask: (B, N, 1) - mask tensor
                - t: (total_masked,) - timesteps for each masked region
        """
        predicted_noise_masked, true_noise, region_motif, mask, t = output
        mask_squeezed = mask.squeeze(-1)  # (B, N)
        
        if predicted_noise_masked.numel() == 0:
            # No masked regions
            pred = {'masked': predicted_noise_masked}
            obs = {'masked': torch.zeros(0, region_motif.shape[-1], device=region_motif.device)}
            return pred, obs
        
        # Extract actual masked region values
        masked_regions = region_motif[mask_squeezed]  # (total_masked, M)
        
        # Reconstruct x_0 from x_t and predicted noise
        # x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * noise
        # So: x_0 = (x_t - sqrt(1 - alpha_hat_t) * noise) / sqrt(alpha_hat_t)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)  # (total_masked, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)  # (total_masked, 1)
        
        # Reconstruct noisy x_t (we had this in forward, but recompute for clarity)
        noisy_masked_regions = sqrt_alpha_t * masked_regions + sqrt_one_minus_alpha_t * true_noise
        
        # Reconstruct x_0 using predicted noise
        # Add small epsilon to avoid division by zero for numerical stability
        eps = 1e-8
        reconstructed_x0 = (noisy_masked_regions - sqrt_one_minus_alpha_t * predicted_noise_masked) / (sqrt_alpha_t + eps)
        
        # Return reconstructed predictions vs actual values for both loss and metrics
        # This allows metrics (r2, pearson) to compare predictions vs observations correctly
        pred = {'masked': reconstructed_x0}
        obs = {'masked': masked_regions}
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
        Inference: iterative denoising for generation using DDPM reverse process.
        Uses fewer steps than training for efficiency.
        
        Args:
            batch: dict with 'region_motif' and 'mask'
            num_steps: number of denoising steps (default 50)
        
        Returns:
            output: (B, N, M) - denoised predictions for masked regions merged with original
        """
        self.eval()
        region_motif = batch['region_motif']
        mask = batch['mask'].unsqueeze(-1).bool()
        mask_squeezed = mask.squeeze(-1)
        B, N, M = region_motif.shape
        device = region_motif.device
        
        # Extract masked regions
        total_masked = mask_squeezed.sum().item()
        if total_masked == 0:
            return region_motif
        
        masked_regions = region_motif[mask_squeezed]  # (total_masked, M)
        
        # Start from pure noise for masked regions
        x_t = torch.randn_like(masked_regions)  # (total_masked, M)
        
        # Use a subset of timesteps for faster inference
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]
        
        # Iterative denoising (reverse diffusion process)
        for i, t_val in enumerate(timesteps):
            t_tensor = torch.full((total_masked,), t_val, device=device, dtype=torch.long)
            
            # Create full input with current noisy masked regions
            current_region_motif = region_motif.clone()
            current_region_motif[mask_squeezed] = x_t
            
            # Embed and process
            x = self.region_embed(current_region_motif)
            C = x.shape[-1]
            
            cls_tokens = self.cls_token.expand(B, -1, -1)
            mask_token = self.mask_token.expand(B, N, -1)
            w = mask.type_as(mask_token)
            x = x * (1 - w) + mask_token * w
            x = torch.cat((cls_tokens, x), dim=1)
            
            # Get timestep per batch sample (use first masked region's timestep)
            num_masked_per_batch = mask_squeezed.sum(dim=1)
            t_batch = torch.zeros(B, dtype=torch.long, device=device)
            mask_idx = 0
            for b in range(B):
                if num_masked_per_batch[b] > 0:
                    t_batch[b] = t_tensor[mask_idx]
                    mask_idx += num_masked_per_batch[b]
            
            # Predict noise
            x_encoded = self.encoder(x, t_batch)
            x_encoded = x_encoded[:, 1:]  # Remove CLS token
            predicted_noise_full = self.head_mask(x_encoded)  # (B, N, M)
            predicted_noise = predicted_noise_full[mask_squeezed]  # (total_masked, M)
            
            # Denoising step using DDPM formula
            alpha_t = self.alphas[t_val]
            alpha_hat_t = self.alphas_cumprod[t_val]
            beta_t = self.betas[t_val]
            
            # Standard DDPM reverse process: x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_hat_t) * predicted_noise) + sqrt(beta_t) * z
            if i < len(timesteps) - 1:  # Not the last step
                # Compute mean: 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_hat_t) * predicted_noise)
                mean = (1.0 / torch.sqrt(alpha_t)).unsqueeze(-1) * (x_t - (beta_t / torch.sqrt(1 - alpha_hat_t)).unsqueeze(-1) * predicted_noise)
                
                # Sample noise
                noise = torch.randn_like(x_t)
                
                # Add noise: sqrt(beta_t) * z
                x_t = mean + torch.sqrt(beta_t).unsqueeze(-1) * noise
            else:
                # Last step: predict x_0 directly (no noise)
                # x_0 = (x_t - sqrt(1 - alpha_hat_t) * predicted_noise) / sqrt(alpha_hat_t)
                x_t = (x_t - torch.sqrt(1 - alpha_hat_t).unsqueeze(-1) * predicted_noise) / torch.sqrt(alpha_hat_t).unsqueeze(-1)
        
        # Merge predictions with original (keep unmasked, use denoised for masked)
        output = region_motif.clone()
        output[mask_squeezed] = x_t
        
        return output


# Alias for backward compatibility with config
GETDiffusion = GETRegionDiffusion
