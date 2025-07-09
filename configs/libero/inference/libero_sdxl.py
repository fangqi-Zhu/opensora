dataset = dict(
    type="LiberoDataset",
    data_paths=[
        "/mnt/hdfs/zhufangqi/datasets/libero",
    ],
    action_dim = 7,
)

image_size = (224,224)
num_frames = 12
fps = 3
data_path = "/mnt/hdfs/zhufangqi/datasets/libero"
save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 4
# align = 5

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="/mnt/hdfs/zhufangqi/checkpoints/opensora/libero/06/12/debug/001-STDiT3-XL-2/epoch0-global_step2500/ema.pt",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    action_dim=7,
    pad_action_num = 4,
)
vae = dict(
    type="SDXL",
    from_pretrained="/mnt/hdfs/zhufangqi/pretrained_models/stabilityai/stable-diffusion-xl-base-1.0/vae",
    micro_frame_size=8,
    micro_batch_size=64,
)
# text_encoder = dict(
#     type="t5",
#     from_pretrained="/mnt/hdfs/zhufangqi/pretrained_models/DeepFloyd/t5-v1_1-xxl",
#     model_max_length=300,
# )
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

# aes = 6.5
# flow = None


# export HF_ENDPOINT=https://hf-mirror.com

# huggingface-cli download --resume-download PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers --local-dir PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers --local-dir-use-symlinks False

