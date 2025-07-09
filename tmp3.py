import glob
import json
from tqdm import tqdm

pattern = "/mnt/hdfs/zhufangqi/datasets/simplevla_rl/webdataset_shards/06/17_mp/**/*.tar"
shards = sorted(glob.glob(pattern, recursive=True))
# shards = sorted(glob.glob(args.pattern, recursive=True))

val_shards = shards[: 200]

metas = []

min_finish_steps = 512

import tarfile
for val_shard in tqdm(val_shards, total = len(val_shards)):
    with tarfile.open(val_shard, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith('.meta.json'):  # 假设meta文件是以 .meta 结尾
                # 读取meta文件的内容
                f = tar.extractfile(member)  # 提取该文件对象
                content = f.read().decode('utf-8')  # 解码为字符串（假设是UTF-8编码）
                try:
                    json_data = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"错误: 无法解析文件 {member.name} 为JSON: {e}")
                min_finish_steps = min(min_finish_steps, json_data['finish_step'])
                metas.append(json_data)

print(f"min_finish_steps: {min_finish_steps}")

results = {}

for meta in metas:
    task_name = meta['task_id']
    complete = meta['complete']
    if task_name not in results:
        results[task_name] = {'succ': 0, 'total': 0}
    if complete:
        results[task_name]['succ'] += 1
    results[task_name]['total'] += 1

succ_rates = []

for task_name, result in sorted(results.items()):
    succ_rate = result['succ'] / result['total']
    succ_rates.append(succ_rate)
    print(f"{task_name}: {succ_rate:.4f}")

print(f"平均成功率: {sum(succ_rates) / len(succ_rates):.4f}")
