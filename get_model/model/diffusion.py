import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from get_model.model.model import BaseGETModel, BaseGETModelConfig
from get_model.model.modules import MotifScanner, ATACSplitPool, RegionEmbed, MotifScannerConfig, ATACSplitPoolConfig, RegionEmbedConfig
from get_model.model.transformer import GETTransformer, EncoderConfig
from get_model.model.modules import BaseConfig

# --- 1. Diffusion Configuration ---
@dataclass
class DiffusionConfig(BaseConfig):
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    hidden_dim: int = 256

# --- 2. Diffusion Head (Denoising Network) ---
class DiffusionHead(nn.Module):
    def __init__(self, cfg: DiffusionConfig, input_dim: int, condition_dim: int):
        super().__init__()
        self.cfg = cfg
        
        # Time embedding (Sinusoidal)
        self.time_embed_dim = cfg.hidden_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        
        # Project condition (Transformer output) to hidden dim
        self.cond_proj = nn.Linear(condition_dim, cfg.hidden_dim)

        # Denoising MLP: Inputs [Noisy_Data + Time_Emb + Condition]
        in_dim = input_dim + self.time_embed_dim + cfg.hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, input_dim) # Output: Predicted Noise
        )

    def forward(self, x, t, condition):
        # x: (B, input_dim)
        # t: (B, 1)
        # condition: (B, embed_dim)
        
        t_emb = self.time_mlp(t.float())
        c_emb = self.cond_proj(condition)
        
        # Concatenate: [x, t_emb, c_emb]
        inp = torch.cat([x, t_emb, c_emb], dim=-1)
        return self.net(inp)

# --- 3. The Main Diffusion Model ---
@dataclass
class GETDiffusionModelConfig(BaseGETModelConfig):
    # Architecture Components
    motif_scanner: MotifScannerConfig = field(default_factory=MotifScannerConfig)
    atac_attention: ATACSplitPoolConfig = field(default_factory=ATACSplitPoolConfig)
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    
    # Diffusion & Output
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    output_dim: int = 1  # 1 for gene expression, >1 for profiles
    
class GETDiffusionModel(BaseGETModel):
    def __init__(self, cfg: GETDiffusionModelConfig):
        super().__init__(cfg)
        
        # 1. Initialize Foundation Components
        self.motif_scanner = MotifScanner(cfg.motif_scanner)
        self.atac_attention = ATACSplitPool(cfg.atac_attention)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.encoder = GETTransformer(**cfg.encoder)
        
        # 2. Initialize Diffusion Head
        self.diffusion_head = DiffusionHead(
            cfg.diffusion, 
            input_dim=cfg.output_dim, 
            condition_dim=cfg.encoder.embed_dim
        )
        
        # 3. Setup Noise Schedule (Buffers for DDPM)
        self.setup_diffusion_schedule(cfg.diffusion)
        
        # Note: LoRA is applied via the 'finetune' config in the training script,
        # so we don't need to hardcode PEFT logic here unless strictly necessary.

    def setup_diffusion_schedule(self, cfg):
        """Pre-compute diffusion parameters."""
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register as buffers so they move to GPU automatically
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def forward(self, batch):
        """
        Training Step:
        1. Encode Input -> Context
        2. Sample t, Noise
        3. Predict Noise
        """
        # A. Encode inputs to get Context
        # (Adapting to the input keys seen in your model.py)
        if 'sample_peak_sequence' in batch:
            x = self.motif_scanner(batch['sample_peak_sequence'], batch['motif_mean_std'])
            x = self.atac_attention(x, batch['sample_track'], batch['chunk_size'], 
                                    batch['n_peaks'], batch['max_n_peaks'])
            x = self.region_embed(x)
        else:
            # Fallback for pre-computed region inputs (like GETRegionPretrain)
            x = self.region_embed(batch['region_motif'])

        # Transformer Encoder
        encodings, _ = self.encoder(x, mask=batch.get('padding_mask', None))
        
        # Use CLS token (index 0) or Mean Pooling as the condition
        condition = encodings[:, 0, :] 
        
        # B. Prepare Diffusion Targets
        target = batch['exp_label'] # (B, output_dim) - Gene Expression
        B = target.size(0)
        device = target.device
        
        # Sample random timestep t
        t = torch.randint(0, self.cfg.diffusion.num_timesteps, (B, 1), device=device)
        
        # Sample noise epsilon
        noise = torch.randn_like(target)
        
        # Add noise: q(x_t | x_0)
        # Extract coeff for the specific batch indices
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        noisy_target = sqrt_alpha * target + sqrt_one_minus_alpha * noise
        
        # C. Predict Noise
        predicted_noise = self.diffusion_head(noisy_target, t, condition)
        
        # Return dictionaries for BaseGETModel loss calculation
        # The keys here must match the keys in your Loss Config
        return {'noise': predicted_noise}, {'noise': noise}

    @torch.no_grad()
    def predict(self, batch):
        """
        Zero-Shot Inference Step:
        Reverse diffusion process to generate prediction from pure noise.
        """
        self.eval()
        
        # A. Encode
        if 'sample_peak_sequence' in batch:
            x = self.motif_scanner(batch['sample_peak_sequence'], batch['motif_mean_std'])
            x = self.atac_attention(x, batch['sample_track'], batch['chunk_size'], 
                                    batch['n_peaks'], batch['max_n_peaks'])
            x = self.region_embed(x)
        else:
            x = self.region_embed(batch['region_motif'])

        encodings, _ = self.encoder(x, mask=batch.get('padding_mask', None))
        condition = encodings[:, 0, :]
        
        # B. Denoise Loop
        B = condition.size(0)
        shape = (B, self.cfg.output_dim)
        
        # Start from pure noise x_T
        img = torch.randn(shape, device=condition.device)
        
        for i in reversed(range(0, self.cfg.diffusion.num_timesteps)):
            t = torch.full((B, 1), i, device=condition.device, dtype=torch.long)
            
            # Predict noise
            pred_noise = self.diffusion_head(img, t, condition)
            
            # Update step (Simple DDPM)
            alpha = self.alphas[i]
            alpha_hat = self.alphas_cumprod[i]
            beta = 1 - alpha
            
            if i > 0:
                z = torch.randn_like(img)
            else:
                z = torch.zeros_like(img)
                
            # x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-alpha_hat) * pred_noise) + sigma * z
            term1 = 1 / torch.sqrt(alpha)
            term2 = (1 - alpha) / (torch.sqrt(1 - alpha_hat))
            img = term1 * (img - term2 * pred_noise) + torch.sqrt(beta) * z
            
        return img
