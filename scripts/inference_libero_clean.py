import os
import time
import json
import imageio
import numpy as np
import tensorflow as tf
import torch
import torch.distributed as dist
from collections import deque
from PIL import Image
from tqdm import tqdm
from pprint import pformat

import colossalai
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from transformers import AutoConfig, AutoTokenizer
import torchvision.transforms as transforms

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.datasets.action_tokenizer import ActionTokenizer
from opensora.datasets.datasets import resize_image
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, DATASETS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype

def load_dataset_statistics(stat_path):
    with open(stat_path, 'r') as f:
        return json.load(f)

def process_image(image_path, image_size, device, dtype):
    image = Image.open(image_path).convert("RGB")
    img = np.array(image)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, size=image_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()
    img = torch.tensor(img).permute(2, 0, 1).float() / 255.0 * 2 - 1
    return img.unsqueeze(1).unsqueeze(0).to(device).to(dtype)

def build_video(original_frames, generated_frames):
    return np.concatenate((original_frames[:generated_frames.shape[0]], generated_frames), axis=2)

def prepare_actions(actions, q01, q99):
    norm_actions = (2 * ((actions - q01) / (q99 - q01))) - 1
    return norm_actions

def encode_condition_frames(vae, image_paths, image_size, device, dtype, queue_len):
    image_queue = deque(maxlen=queue_len)
    images = [process_image(path, image_size, device, dtype) for path in image_paths]
    images_tensor = torch.concat(images, dim=2)
    latents = vae.encode(images_tensor)
    for i in range(latents.shape[2]):
        image_queue.append(latents[:, :, i:i+1])
    return image_queue

def main():
    torch.set_grad_enabled(False)

    cfg = parse_configs(training=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    set_random_seed(seed=cfg.get("seed", 1024))

    is_vae_v1_2 = cfg.vae.get('type') == 'OpenSoraVAE_V1_2'

    logger = create_logger()
    logger.info("Inference configuration:\n%s", pformat(cfg.to_dict()))
    progress = tqdm if cfg.get("verbose", 1) == 1 else lambda x: x

    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    dataset = build_module(cfg.dataset, DATASETS)

    image_size = cfg.get("image_size") or get_image_size(cfg.resolution, cfg.aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)
    latent_size = vae.get_latent_size((num_frames, *image_size))

    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
    ).to(device, dtype).eval()

    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    model_args = prepare_multi_resolution_info(cfg.get("multi_resolution"), 1, image_size, num_frames, cfg.fps, device, dtype)

    dataset_statistics = load_dataset_statistics('opensora/datasets/libero_dataset_statistics.json')
    task_names = [f"{name}_no_noops" for name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]]

    for task_name in task_names:
        task_path = os.path.join(cfg.data_path, task_name)
        q01 = np.array(dataset_statistics[task_name]['action']['min'])
        q99 = np.array(dataset_statistics[task_name]['action']['max'])
        output_dir = f"./video_results/{'/'.join(cfg.model.from_pretrained.split('/')[9:12])}/{task_name}"
        os.makedirs(output_dir, exist_ok=True)

        for episode_id in range(3):
            action_data = np.load(os.path.join(task_path, "actions", f"{episode_id:06d}.npy"))[:-1]
            actions = prepare_actions(action_data, q01, q99)
            # actions = dataset.convert_tokens_to_actions(dataset.convert_actions_to_tokens(prepare_actions(action_data, q01, q99)))
            discrete_actions = torch.from_numpy(actions)[cfg.condition_frame_length - 1:]

            image_paths = [
                os.path.join(task_path, "images", f"{episode_id:06d}", f"{i:03d}.png")
                for i in range(cfg.condition_frame_length)
            ]
            image_queue = encode_condition_frames(
                vae, image_paths, dataset.image_size, device, dtype,
                queue_len=cfg.condition_frame_length // 4 if is_vae_v1_2 else cfg.condition_frame_length
            )

            all_image_paths = sorted(os.listdir(os.path.join(task_path, f"images/{episode_id:06d}")))
            full_video = np.stack([
                np.array(Image.open(os.path.join(task_path, f"images/{episode_id:06d}", name)).resize((224, 224)))
                for name in sorted(all_image_paths, key=lambda x: int(x.split('.')[0]))
            ])

            predicted_video = [full_video[:cfg.condition_frame_length]]
            chunk_len = num_frames - cfg.condition_frame_length
            latent_chunk = chunk_len // 4 if is_vae_v1_2 else chunk_len

            for i in progress(range(0, len(discrete_actions), chunk_len)):
                act_chunk = discrete_actions[i:i+chunk_len]
                if len(act_chunk) < chunk_len:
                    continue

                y = act_chunk.reshape(-1, dataset.action_dim).unsqueeze(0).to(device).to(dtype)
                mask_images = torch.concat(list(image_queue), dim=2)
                z = torch.randn(1, vae.out_channels, latent_chunk, *latent_size[1:], device=device, dtype=dtype)
                z = torch.concat([mask_images, z], dim=2)
                masks = torch.tensor([[0]*mask_images.shape[2] + [1]*latent_chunk], device=device, dtype=dtype)

                samples = scheduler.sample(model, z=z, y=y, device=device, additional_args=model_args, progress=False, mask=masks)
                pred_latents = samples[:, :, -latent_chunk:].to(dtype)

                image_queue.extend(pred_latents.clone().chunk(latent_chunk, dim=2))
                decoded_images = vae.decode(pred_latents, num_frames=12 if is_vae_v1_2 else None)
                pred_imgs = (decoded_images.to(torch.float32).squeeze().cpu().permute(1, 2, 3, 0).numpy() * 0.5 + 0.5) * 255
                predicted_video.append(pred_imgs.clip(0, 255).astype(np.uint8))

            final_video = build_video(full_video[cfg.get("start_index", 0):], np.concatenate(predicted_video))
            imageio.mimwrite(os.path.join(output_dir, f"{episode_id:06d}.mp4"), final_video, fps=30)

        logger.info("Inference finished for task %s.", task_name)

if __name__ == "__main__":
    main()
