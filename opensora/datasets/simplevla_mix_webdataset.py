import io
import json
import random
import glob
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from torch.utils.data import IterableDataset, DataLoader
from opensora.registry import DATASETS

@DATASETS.register_module()
class SimpleVLAWebDataset(IterableDataset):
    def __init__(
        self,
        shards_pattern: str,
        stats_path: str,
        *,
        Ta: int = 8,
        To: int = 4,
        stride: int = 1,
        action_dim: int = 7,
        image_size: Tuple[int, int] = (256, 256),
        episode_per_shard: int = 10,
        episode_buf_size: int = 30,
        sample_buf_size: int = 10000,
    ) -> None:
        super().__init__()
        self.Ta, self.To, self.stride = Ta, To, stride
        self.context_length = self.Ta + self.To
        self.image_size = image_size

        # ---------- 1. 发现所有 shard ----------
        shards: List[str] = sorted(glob.glob(shards_pattern, recursive=True))
        if not shards:
            raise FileNotFoundError(
                f"[SimpleVLA] pattern '{shards_pattern}' did not match any .tar files"
            )
        print(f"[SimpleVLA] Found {len(shards)} shard(s) for pattern: {shards_pattern}")

        # ---------- 2. 读取归一化常量 ----------
        with open(stats_path, "r") as f:
            stats = json.load(f)
        task = "libero_10_no_noops"  # NOTE: adapt to your own task key
        self.q01 = np.asarray(stats[task]["action"]["q01"], np.float32)
        self.q99 = np.asarray(stats[task]["action"]["q99"], np.float32)

        # ---------- 3. 构建 WebDataset Pipeline ----------
        self.ds = wds.DataPipeline(
            wds.SimpleShardList(shards, seed=42),  # non‑replacement sampling
            wds.split_by_node,                      # multi‑node split
            wds.split_by_worker,                   # multi‑worker split
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(episode_buf_size, initial=episode_buf_size),
            wds.to_tuple("video.npy", "action.npy", "meta.json"),
            self._split_to_windows,                # ⇐ custom window extractor
            wds.shuffle(sample_buf_size, initial=sample_buf_size),
        )

    # ---------------------------------------------------------------------
    # IterableDataset interface
    # ---------------------------------------------------------------------
    def __iter__(self):
        return iter(self.ds)

    # ------------------------------------------------------------------
    # Helper that converts one episode → multiple (video, action) windows
    # ------------------------------------------------------------------
    def _split_to_windows(
        self, src: Iterable[Tuple[bytes, bytes, bytes]]
    ) -> Iterable[Dict]:
        for v_bytes, a_bytes, m_bytes in src:
            # (T, C, H, W)
            video_np = np.load(io.BytesIO(v_bytes), allow_pickle=False)
            # (T, action_dim)
            action_np = np.load(io.BytesIO(a_bytes), allow_pickle=False)
            meta = json.loads(m_bytes.decode())
            finish_step: int = int(meta.get("finish_step"))

            for start in range(0, finish_step - self.Ta + 1, self.stride):
                vs, ve = start - self.To + 1, start + self.Ta + 1
                if vs < 0:  # pad if we look back before t=0
                    pad = np.repeat(video_np[0:1], -vs, axis=0)
                    vid_win = np.concatenate([pad, video_np[:ve]], axis=0)
                else:
                    vid_win = video_np[vs:ve]

                # (num_frames, C, H, W) → (C, num_frames, H, W)
                video = torch.from_numpy(vid_win).float() / 127.5 - 1
                video = video.permute(3, 0, 1, 2)

                act_np = action_np[start : start + self.Ta]
                action = 2 * ((act_np - self.q01) / (self.q99 - self.q01)) - 1
                action = torch.from_numpy(action).float()

                debug_meta = {
                    "fpath": meta.get("fpath", ""),
                    "episode_name": meta.get("episode_name", ""),
                    "video_start": vs,
                    "video_end": ve,
                    "action_start": start,
                    "action_end": start + self.Ta,
                }

                yield {
                    "video": video,
                    "action": action,
                    "fps": 30,
                    "num_frames": video.shape[1],
                    "height": self.image_size[0],
                    "width": self.image_size[1],
                    "meta": debug_meta,
                }


# -----------------------------------------------------------------------------
# NEW: Multi‑source dataset that mixes several SimpleVLAWebDataset objects
#       according to user‑specified sampling weights.
# -----------------------------------------------------------------------------

class MixedSimpleVLAWebDataset(IterableDataset):
    """Sample from multiple *already‑constructed* datasets with given ratios.

    This wrapper keeps separate iterators for each underlying dataset and on
    every `__next__` randomly chooses which iterator to draw from, according to
    the supplied *weights*.
    """

    def __init__(
        self,
        shards_pattern1: str,
        shards_pattern2: str,
        stats_path: str,
        *,
        weights: Tuple[float, float] = (0.7, 0.3),
        Ta: int = 8,
        To: int = 4,
        stride: int = 1,
        action_dim: int = 7,
        image_size: Tuple[int, int] = (256, 256),
        episode_per_shard: int = 10,
        episode_buf_size: int = 30,
        sample_buf_size: int = 10000,
    ):
        super().__init__()
        ds1 = SimpleVLAWebDataset(shards_pattern1, stats_path=stats_path, Ta=Ta, To=To, stride=stride, action_dim=action_dim, image_size=image_size, episode_per_shard=episode_per_shard, episode_buf_size=episode_buf_size, sample_buf_size=sample_buf_size)
        ds2 = SimpleVLAWebDataset(shards_pattern2, stats_path=stats_path, Ta=Ta, To=To, stride=stride, action_dim=action_dim, image_size=image_size, episode_per_shard=episode_per_shard, episode_buf_size=episode_buf_size, sample_buf_size=sample_buf_size)
        datasets = [ds1, ds2]
        self.context_length = Ta + To
        self.image_size = image_size

        if len(datasets) != len(weights):
            raise ValueError("`datasets` and `weights` must have the same length")
        if any(w < 0 for w in weights):
            raise ValueError("`weights` must be non‑negative")

        total = float(sum(weights))
        if total == 0:
            raise ValueError("`weights` must sum to a positive value")

        self.datasets = datasets
        # normalize so that sum(weights) == 1.0
        self.weights = [w / total for w in weights]

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _choose_dataset_idx(self) -> int:
        """Return an index in [0, len(datasets)) sampled wrt `self.weights`."""
        r = random.random()
        cum = 0.0
        for idx, w in enumerate(self.weights):
            cum += w
            if r <= cum:
                return idx
        return len(self.weights) - 1  # fallback due to FP precision

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------
    def __iter__(self):
        # create one independent iterator per underlying dataset
        iterators = [iter(ds) for ds in self.datasets]

        while True:
            idx = self._choose_dataset_idx()
            try:
                yield next(iterators[idx])
            except StopIteration:
                # Underlying iterator exhausted → recreate it & continue
                if idx == 1:
                    iterators[idx] = iter(self.datasets[idx])
                    yield next(iterators[idx])
                else:
                    break


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from tqdm import tqdm

    # ------------------------------------------------------------------
    # 1. Define your shard patterns
    # ------------------------------------------------------------------
    pattern1 = "/mnt/hdfs/zhufangqi/datasets/simplevla_rl/webdataset_shards/06/16_mp/**/*.tar"
    pattern2 = "/mnt/hdfs/zhufangqi/datasets/simplevla_rl/webdataset_shards/libero/libero_10_no_noops/**/*.tar"

    stats_json = "/mnt/hdfs/zhufangqi/datasets/libero/dataset_statistics.json"

    # ------------------------------------------------------------------
    # 2. Build individual datasets
    # ------------------------------------------------------------------
    ds1 = SimpleVLAWebDataset(pattern1, stats_json, Ta=8, To=4, stride=1)
    ds2 = SimpleVLAWebDataset(pattern2, stats_json, Ta=8, To=4, stride=1)

    # ------------------------------------------------------------------
    # 3. Wrap them in a mixed dataset with desired sampling ratios
    # ------------------------------------------------------------------
    mixed_ds = MixedSimpleVLAWebDataset([ds1, ds2], weights=[0.7, 0.3])

    # ------------------------------------------------------------------
    # 4. Build a WebLoader (or a regular DataLoader with a suitable collate_fn)
    # ------------------------------------------------------------------
    batch_size = 4
    dataloader = wds.WebLoader(
        mixed_ds,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 5. Consume some batches for sanity checking
    # ------------------------------------------------------------------

    sampled_rollout_batches_meta: List[Dict] = []
    sampled_libero_batches_meta: List[Dict] = []
    for batch in tqdm(dataloader):  # first 10 batches is enough for a smoke test
        # `batch` is a dictionary where each key maps to a tensor/list of length `batch_size`
        for i in range(batch["video"].shape[0]):
            if batch["meta"]["fpath"][i] != "":
                sampled_rollout_batches_meta.append(
                    {
                        "fpath": batch["meta"]["fpath"][i],
                        "video_start": int(batch["meta"]["video_start"][i]),
                        "video_end": int(batch["meta"]["video_end"][i]),
                        "action_start": int(batch["meta"]["action_start"][i]),
                        "action_end": int(batch["meta"]["action_end"][i]),
                    }
                )
            else:
                sampled_libero_batches_meta.append(
                    {
                        "episode_name": batch["meta"]["episode_name"][i],
                        "video_start": int(batch["meta"]["video_start"][i]),
                        "video_end": int(batch["meta"]["video_end"][i]),
                        "action_start": int(batch["meta"]["action_start"][i]),
                        "action_end": int(batch["meta"]["action_end"][i]),
                    }
                )

    # Optional: save meta information for inspection
    with open("sampled_libero_batches_meta.json", "w") as f:
        json.dump(sampled_libero_batches_meta, f, indent=4)
    with open("sampled_rollout_batches_meta.json", "w") as f:
        json.dump(sampled_rollout_batches_meta, f, indent=4)

    print("✓ Completed sanity run with mixed dataset (0.7 / 0.3).")
