import torch
import os
import imageio

from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.misc import to_torch_dtype
from opensora.utils.config_utils import parse_configs

from torchvision import transforms
from PIL import Image
from einops import rearrange
import numpy as np

cfg = parse_configs(training=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = to_torch_dtype("bf16")
vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

images_dir = '/mnt/hdfs/zhufangqi/datasets/libero/libero_10_no_noops/images/000000'

# 加载并转换图片，移动到 GPU
images = torch.stack([
    transforms.ToTensor()(Image.open(os.path.join(images_dir, f))).to(device)
    for f in sorted(os.listdir(images_dir))
])

images = (images-0.5)*2

images = images[32:64]

images = rearrange(images, 'f c h w -> 1 c f h w')

x = images.to(device, dtype)
z = vae.encode(x)
x_rec = vae.decode(z, num_frames=x.size(2))

videos = x_rec

videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()
videos = rearrange(videos, '1 c f h w -> f h w c')

ori_videos = ((images / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()
ori_videos = rearrange(ori_videos, '1 c f h w -> f h w c')

videos = np.concatenate([ori_videos, videos], axis=2)

with imageio.get_writer('test_vae.mp4', fps=3, codec='libx264') as writer:
    for frame in videos:
        writer.append_data(frame)

print()