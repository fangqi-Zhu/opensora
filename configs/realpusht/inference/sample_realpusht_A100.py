# resolution = "240p"
# aspect_ratio = "9:16"
image_size = (72,96)
num_frames = 18
fps = 3
save_fps = 3

dataset = dict(
    type="RealPushTDataset",
    data_path=["/opt/tiger/opensora/data/0412.h5", "/opt/tiger/opensora/data/0416_inference.h5"],
    num_frames=18,
    image_size = [96, 72]
)


save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 2
align = 5

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="/mnt/bn/zhufangqi-lq-c2ec0f30/zhufangqi/world-model/Open-Sora/pretrained_models/hpcai-tech/OpenSora-STDiT-v3/model.safetensors",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    action_dim=2
)
vae = dict(
    type="SDXL",
    from_pretrained="/mnt/bn/zhufangqi-lq-c2ec0f30/zhufangqi/world-model/opensora/pretrained_models/vae/0412_all/epoch_5",
    micro_frame_size=8,
    micro_batch_size=64,
)
text_encoder = dict(
    type="t5",
    from_pretrained="/mnt/bn/zhufangqi-lq-c2ec0f30/zhufangqi/world-model/Open-Sora/pretrained_models/DeepFloyd/t5-v1_1-xxl",
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

