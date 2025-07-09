#!/usr/bin/env python3
# coding: utf-8


"""
Locate indices of webdataset_data_info items inside hdf5_data_info
and save the indices to a text file.
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict


def load_json(path: str | Path) -> list:
    """Load a JSON file and return its root object."""
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def build_index_map(source: list) -> Dict[str, List[int]]:
    """
    Build a mapping from the JSON-serialized form of each element to
    all indices where that element appears in *source*.

    We JSON-dump with sort_keys=True so that key order doesn’t affect equality.
    """
    index_map: Dict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(source):
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        index_map[key].append(idx)
    return index_map


def lookup_indices(source_map: Dict[str, List[int]], targets: list) -> Tuple[List[int], List[int]]:
    """
    For every *targets* element, fetch the first index recorded in *source_map*.
    Returns (found_indices, missing_count).
    """
    found: List[int] = []
    missing: List[int] = []

    for item in targets:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in source_map:
            found.append(source_map[key][0])  # 取第一次出现的位置
        else:
            missing.append(item)

    return found, missing


def main() -> None:
    # -------- 路径设置 --------
    hdf5_path = Path("/opt/tiger/opensora/hdf5_data_info.json")
    web_path = Path("/opt/tiger/opensora/sampled_rollout_batches_meta.json")
    out_path = Path("web_in_hdf5_indices.txt")

    # -------- 数据载入 --------
    hdf5_data = load_json(hdf5_path)
    web_data = load_json(web_path) # 如需全部数据，删除切片

    # -------- 生成索引 --------
    index_map = build_index_map(hdf5_data)
    indices, missing = lookup_indices(index_map, web_data)
    print(len(set(indices)))
    print(len(indices)) # , "有重复的索引"

    # -------- 写入文件 --------
    out_path.write_text("\n".join(map(str, indices)), encoding="utf-8")
    print(f"✅ 已写入 {len(indices)} 个索引到 {out_path.resolve()}")

    # -------- 结果检查 --------
    if missing:
        print(f"⚠️  有 {len(missing)} 个元素在 hdf5_data_info.json 中未找到。")
    else:
        print("🎉 webdataset_first_buffer 中的所有元素均出现在 hdf5_data_info.json 中。")


if __name__ == "__main__":
    main()
