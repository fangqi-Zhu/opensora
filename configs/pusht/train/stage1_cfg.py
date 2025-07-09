dataset = dict(
    type="LiberoDataset",
    num_frames=8,
)

grad_checkpoint = False

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
plugin = "zero2"

# Model settings
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="/mnt/hdfs/zhufangqi/pretrained_models/hpcai-tech/OpenSora-STDiT-v3/model.safetensors",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    freeze_y_embedder=True,
    skip_y_embedder=True,
    cfg=True,
    training=True,
)

vae = dict(
    type="SDXL",
    from_pretrained="/mnt/hdfs/zhufangqi/pretrained_models/stabilityai/stable-diffusion-xl-base-1.0/vae",
    micro_frame_size=8,
    micro_batch_size=64,
)

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
)

# Mask settings
mask_ratios = {
    "video_half_tail": 1.0,
}

batch_size = 8

# Log settings
seed = 42
outputs = "/mnt/hdfs/zhufangqi/checkpoints/opensora/libero/01/27/cfg"
wandb = True
epochs = 1000
log_every = 10
ckpt_every = 10000

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-4
ema_decay = 0.99
adam_eps = 1e-15
warmup_steps = 1000

cache_pin_memory = True
pin_memory_cache_pre_alloc_numels = [(290 + 20) * 1024**2] * (2 * 8 + 4)
