# resolution = "240p"
# aspect_ratio = "9:16"
image_size = (256,256)
num_frames = 8
fps = 3
frame_interval = 1
save_fps = 3

data_path = '/mnt/bn/zhufangqi-lq-c2ec0f30/zhufangqi/world-model/data_preprocessing/data/libero'

save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 7
align = 5

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="outputs/libero/001-STDiT3-XL-2/epoch75-global_step80000/ema.safetensors",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="SDXL",
    from_pretrained="/mnt/bn/zhufangqi-lq-c2ec0f30/zhufangqi/world-model/IRASim/pretrained_models/stabilityai/stable-diffusion-xl-base-1.0/vae",
    micro_frame_size=8,
    micro_batch_size=64,
)
text_encoder = dict(
    type="t5",
    from_pretrained="./DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5
flow = None


# export HF_ENDPOINT=https://hf-mirror.com

# huggingface-cli download --resume-download PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers --local-dir PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers --local-dir-use-symlinks False

