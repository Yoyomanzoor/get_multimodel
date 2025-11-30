#!/usr/bin/env python3

import os
from get_model.config.config import export_config, load_config, load_config_from_yaml
from get_model.run_region import run_zarr as run
os.chdir('/home/yoyomanzoor/Documents/get_multimodel')

codebase_dir = "/home/yoyomanzoor/Documents/get_multimodel"
data_path = "/home/yoyomanzoor/Crucial/get_data/annotation_dir/pbmc10k_multiome.zarr"
output_dir = "/home/yoyomanzoor/Crucial/get_data/output"
checkpoint_path = "/home/yoyomanzoor/Documents/get_multimodel/tutorials/checkpoint-799.pth"
run_name = "diffusion_pbmc_from_pretrain_lora"
config_path = f"tutorials/yamls/{run_name}.yaml"

# Predefined configuration
celltype_for_modeling = [
    'memory_b',
    'cd14_mono',
    'gdt',
    'cd8_tem_1',
    'naive_b',
    'mait',
    'intermediate_b',
    'cd4_naive',
    'cd8_tem_2',
    'cd8_naive',
    'cd4_tem',
    'cd4_tcm',
    'cd16_mono',
    'nk',
    'cdc',
    'treg']
cfg = load_config('finetune_tutorial') # load the predefined finetune tutorial config
cfg.run.project_name = 'pretrain_pbmc' # this is a unique name for this project
cfg.training.warmup_epochs = 10
cfg.dataset.leave_out_celltypes = 'cd8_tem_1'
cfg.dataset.zarr_path = data_path # the tutorial data which contains astrocyte atac & rna
cfg.dataset.celltypes = ','.join(celltype_for_modeling) # the celltypes you want to pretrain
cfg.dataset.leave_out_chromosomes = None # pretrain on all chromosomes
cfg.run.use_wandb=True # this is a logging system, you can turn it off by setting it to False
cfg.training.epochs = 20 # this is the number of epochs you want to train for
cfg.training.val_check_interval = 1.0 # validation check every epochs; this is for mac because the evaluation step is slow on it somehow...

# Model selection
cfg.model = load_config_from_yaml('get_model/config/model/GetRegionDiffusion.yaml').model
cfg.dataset.mask_ratio = 0.5 # mask 50% of the motifs. This has to be set for pretrain dataloader to generate proper mask

# Train the model without the pretrain checkpoint
cfg.run.run_name = run_name
cfg.machine.codebase = codebase_dir
cfg.machine.output_dir = output_dir
cfg.dataset.zarr_path = data_path
cfg.finetune.checkpoint = checkpoint_path
cfg.finetune.model_key = "model"
cfg.finetune.rename_config = {
  "encoder.head.": "head_mask.",
  "encoder.region_embed": "region_embed",
  "region_embed.proj.": "region_embed.embed.",
  "encoder.cls_token": "cls_token",
}
# strict=False allows loading base model weights while leaving diffusion components randomly initialized
cfg.finetune.strict = False
cfg.finetune.use_lora = True
cfg.finetune.layers_with_lora = ['region_embed', 'encoder']

# Export the final configuration to a yaml file
export_config(cfg, config_path)

cfg = load_config_from_yaml(config_path)

# Train
trainer = run(cfg)
trainer.callback_metrics

