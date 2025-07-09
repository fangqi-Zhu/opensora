import json
import numpy as np
import shutil
import os

data_path = 'results/final_results.json'

with open(data_path, 'r') as f:
    data = json.load(f)

seq_1 = [0.3699, 0.4474, 0.8674, 0.7289, 0.0377, 0.4454, 0.2935, 0.3912, 0.1810, 0.4525]

thres = np.linspace(0.3, 1.0, 20)

for thre in thres:
    results = {}
    for sample in data:
        video_path = sample['video_path']
        task_id = int(os.path.basename(video_path).split('_')[1])
        meta = sample['results']
        # label = sample['label']
        label = 1 if any(r["prob"][1] >= thre for r in meta) else 0
        
        if task_id not in results:
            results[task_id] = {'succ': 0, 'total': 0}
        if label:
            results[task_id]['succ'] += 1
        results[task_id]['total'] += 1
    succ_rates = []

    for task_name, result in sorted(results.items()):
        succ_rate = result['succ'] / result['total']
        succ_rates.append(succ_rate)
        print(f"Task {task_name}: {succ_rate:.4f}")

    pearson_correlation = np.corrcoef(seq_1, succ_rates)[0, 1]
    print(f"Pearson correlation: {pearson_correlation:.4f}")
    print(f"平均成功率: {sum(succ_rates) / len(succ_rates):.4f}")
    print(f"thre={thre:.2f}")


    # video_name = os.path.basename(video_path).split('.')[0]
    # new_video_name = f"{video_name}_{post_suffix}.mp4"
    # new_video_path = os.path.join(video_dir, new_video_name)
    # shutil.copy(video_path, new_video_path)

# thres = np.linspace(0.3, 1.0, 20)
# for thre in thres:
#     succ_num = 0
#     for sample in result:
#         video_path = sample['video_path']
#         meta = sample['results']
#         label = 0
#         for i, m in enumerate(meta):
#             if m['prob'][1] >= thre:
#                 label = 1
#         succ_num += label
#     print(f"thre={thre:.2f}, succ_num={succ_num}, succ_rate={succ_num/len(result):.4f}")

