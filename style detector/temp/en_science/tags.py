from collections import Counter
from datasets import load_dataset

# 加载 arXiv 数据集（metadata + abstracts等）
ds = load_dataset("nick007x/arxiv-papers", split="train")

# 统计 subjects（假设字段名叫 "subjects" 或类似）
# 根据数据集页面，这里用 ds["subjects"]
all_subjects = []
for item in ds:
    subs = item.get("subjects", None)
    if subs:
        # 有可能 subjects 是一个 list 或字符串
        if isinstance(subs, list):
            all_subjects.extend(subs)
        elif isinstance(subs, str):
            # 有时是字符串多个 subject 用空格或逗号分隔
            parts = [s.strip() for s in subs.split() if s.strip()]
            all_subjects.extend(parts)

# 统计频率
cnt = Counter(all_subjects)
print("最常见的 subject 类别 (前 20)：")
for subj, c in cnt.most_common(20):
    print(subj, c)
