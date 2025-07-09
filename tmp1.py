import os
import ast
from collections import defaultdict

# 用于记录每个 sample（以字符串形式）的出现次数
sample_count = defaultdict(list)

for rank in range(8):
    log_file = f"/mnt/hdfs/zhufangqi/checkpoints/opensora/libero/06/20/sdxl_simplevla_debug/025-STDiT3-XL-2/debug_rank_{rank}.log"
    if not os.path.exists(log_file):
        print(f"File not found: {log_file}")
        continue

    with open(log_file, "r") as f:
        for line in f:
            try:
                # 提取 {...} 部分
                info_str = line[11:]
                info = ast.literal_eval(info_str)

                # 只检查 fpath 非空的样本
                if info.get("fpath", ""):
                    # 使用 frozenset 保证 sample 是 hashable 的，也可以用 str(info)
                    key = str(info)  # 字典序列化后作为唯一标识
                    sample_count[key].append(info)
            except Exception as e:
                print(f"Parse error in {log_file}: {line.strip()}\n{e}")

# 找出重复的样本
duplicates = {k: v for k, v in sample_count.items() if len(v) > 1}

# 打印结果
if duplicates:
    print("Found duplicate samples (with non-empty fpath):")
    for sample_str, instances in duplicates.items():
        print(f"\nDuplicate sample ({len(instances)} times):")
        print(instances[0])  # 内容都是一样的，只打印一次
else:
    print("No duplicate samples found.")
