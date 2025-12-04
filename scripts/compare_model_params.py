#!/usr/bin/env python3
"""
Compare parameter counts between GETRegionPretrain and GETRegionDiffusion models.
"""

import sys
import os

codebase_dir = "/home/yoyomanzoor/Documents/get_multimodel"

# Add project root to Python path
PROJECT_ROOT = codebase_dir
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)

import torch
from get_model.config.config import load_config, load_config_from_yaml
from hydra.utils import instantiate

def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_number(num):
    """Format number with commas and millions/billions."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B ({num:,})"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M ({num:,})"
    else:
        return f"{num:,}"

def print_model_structure(model, name, indent=0):
    """Print model structure with parameter counts."""
    prefix = "  " * indent
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{prefix}{name}: {format_number(total_params)} params")
    
    if hasattr(model, 'children'):
        for child_name, child in model.named_children():
            child_total = sum(p.numel() for p in child.parameters())
            if child_total > 0:
                print_model_structure(child, child_name, indent + 1)

# Model 1: GETRegionPretrain
print("=" * 80)
print("MODEL 1: GETRegionPretrain")
print("=" * 80)
cfg1 = load_config('pretrain_tutorial')
cfg1.model = load_config('model/GETRegionPretrain').model.model

# Set config values to match the script
cfg1.model.cfg.num_regions = 900
cfg1.model.cfg.num_motif = 283
cfg1.model.cfg.embed_dim = 768
cfg1.model.cfg.num_layers = 12
cfg1.model.cfg.num_heads = 12
cfg1.model.cfg.dropout = 0.1
cfg1.model.cfg.output_dim = 283
cfg1.model.cfg.flash_attn = False
cfg1.model.cfg.pool_method = "mean"
cfg1.model.cfg.region_embed.num_features = 283
cfg1.model.cfg.region_embed.embed_dim = 768
cfg1.model.cfg.encoder.num_heads = 12
cfg1.model.cfg.encoder.embed_dim = 768
cfg1.model.cfg.encoder.num_layers = 12
cfg1.model.cfg.encoder.drop_path_rate = 0.1
cfg1.model.cfg.head_mask.in_features = 768
cfg1.model.cfg.head_mask.out_features = 283
cfg1.model.cfg.mask_token.embed_dim = 768

model1 = instantiate(cfg1.model)
total1, trainable1 = count_parameters(model1)

print("\nModel Structure:")
print_model_structure(model1, "GETRegionPretrain")
print(f"\nTotal Parameters: {format_number(total1)}")
print(f"Trainable Parameters: {format_number(trainable1)}")

# Model 2: GETRegionDiffusion
print("\n" + "=" * 80)
print("MODEL 2: GETRegionDiffusion")
print("=" * 80)
cfg2 = load_config('pretrain_tutorial')
cfg2.model = load_config_from_yaml('get_model/config/model/GETDiffusion.yaml').model

# Set config values to match the script
cfg2.model.cfg.num_regions = 900
cfg2.model.cfg.num_motif = 283
cfg2.model.cfg.embed_dim = 768
cfg2.model.cfg.num_layers = 12
cfg2.model.cfg.num_heads = 12
cfg2.model.cfg.dropout = 0.1
cfg2.model.cfg.output_dim = 283
cfg2.model.cfg.flash_attn = False
cfg2.model.cfg.pool_method = "mean"
cfg2.model.cfg.region_embed.num_features = 283
cfg2.model.cfg.region_embed.embed_dim = 768
cfg2.model.cfg.head_mask.in_features = 768
cfg2.model.cfg.head_mask.out_features = 283
cfg2.model.cfg.mask_token.embed_dim = 768

model2 = instantiate(cfg2.model)
total2, trainable2 = count_parameters(model2)

print("\nModel Structure:")
print_model_structure(model2, "GETRegionDiffusion")
print(f"\nTotal Parameters: {format_number(total2)}")
print(f"Trainable Parameters: {format_number(trainable2)}")

# Comparison
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
diff = total2 - total1
diff_pct = (diff / total1) * 100 if total1 > 0 else 0

print(f"\nParameter Difference:")
print(f"  GETRegionDiffusion: {format_number(total2)}")
print(f"  GETRegionPretrain: {format_number(total1)}")
print(f"  Difference:        {format_number(diff)} ({diff_pct:+.2f}%)")

# Detailed breakdown
print("\n" + "-" * 80)
print("DETAILED BREAKDOWN")
print("-" * 80)

def get_component_params(model, component_name):
    """Get parameters for a specific component."""
    if hasattr(model, component_name):
        component = getattr(model, component_name)
        if isinstance(component, torch.nn.Parameter):
            return component.numel()
        elif isinstance(component, torch.nn.Module):
            return sum(p.numel() for p in component.parameters())
    return 0

print("\nGETRegionPretrain Components:")
print(f"  region_embed:  {format_number(get_component_params(model1, 'region_embed'))}")
print(f"  encoder:       {format_number(get_component_params(model1, 'encoder'))}")
print(f"  head_mask:     {format_number(get_component_params(model1, 'head_mask'))}")
print(f"  mask_token:     {format_number(get_component_params(model1, 'mask_token'))}")
print(f"  cls_token:      {format_number(get_component_params(model1, 'cls_token'))}")

print("\nGETRegionDiffusion Components:")
print(f"  region_embed:  {format_number(get_component_params(model2, 'region_embed'))}")
print(f"  encoder:       {format_number(get_component_params(model2, 'encoder'))}")
print(f"  head_mask:     {format_number(get_component_params(model2, 'head_mask'))}")
print(f"  mask_token:    {format_number(get_component_params(model2, 'mask_token'))}")
print(f"  cls_token:     {format_number(get_component_params(model2, 'cls_token'))}")

# Encoder breakdown
print("\n" + "-" * 80)
print("ENCODER BREAKDOWN")
print("-" * 80)

encoder1_params = get_component_params(model1, 'encoder')
encoder2_params = get_component_params(model2, 'encoder')
encoder_diff = encoder2_params - encoder1_params

print(f"\nGETTransformer (GETRegionPretrain):")
print(f"  Total: {format_number(encoder1_params)}")

print(f"\nGETRegionDiTEncoder (GETRegionDiffusion):")
print(f"  Total: {format_number(encoder2_params)}")
if hasattr(model2.encoder, 't_embedder'):
    t_embedder_params = sum(p.numel() for p in model2.encoder.t_embedder.parameters())
    print(f"    t_embedder: {format_number(t_embedder_params)}")
if hasattr(model2.encoder, 'blocks'):
    dit_block_params = sum(p.numel() for p in model2.encoder.blocks[0].parameters()) if len(model2.encoder.blocks) > 0 else 0
    print(f"    per DiT block: {format_number(dit_block_params)}")
    print(f"    total DiT blocks (x12): {format_number(dit_block_params * 12)}")

print(f"\nEncoder Difference: {format_number(encoder_diff)} ({encoder_diff/encoder1_params*100:+.2f}%)")

