dataset = dict(
    type="LiberoContinueDataset",
    data_path = "/opt/tiger/simplevla-rl/debug/valid_batch",
    stats_path = "/mnt/hdfs/zhufangqi/datasets/libero/dataset_statistics.json"
)

image_size = (224,224)
num_frames = 12
fps = 3
data_path = "/opt/tiger/simplevla-rl/debug/valid_batch"
save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 4
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="/mnt/hdfs/zhufangqi/checkpoints/opensora/libero/06/15/sdxl_continue_fix/001-STDiT3-XL-2/epoch59-global_step19260/ema.pt",
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

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

