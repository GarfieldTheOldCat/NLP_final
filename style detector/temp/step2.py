import os
import glob
import math
import random
import pandas as pd
import csv

# 定义路径
language = "en"
style = "science"
source_dir = r"G:\ECE\ECE684_NLP\final\style detector\temp\\" + language + "_" + style
train_dir = r"../corpus/train"
test_dir = r"../corpus/test"

# 创建目标目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 1. 获取所有txt文件
txt_files = glob.glob(os.path.join(source_dir, "*.txt"))
n = len(txt_files)
print(f"找到 {n} 个txt文件")

# 2. 计算每个文件需要抽取的数量
per_file_samples = math.ceil(1000 / n) + 1
print(f"每个文件抽取: {per_file_samples} 条")

# 3. 从每个文件抽取候选句子，同时收集剩余句子作为测试集
all_candidates = []  # 候选训练样本
test_samples = []  # 全部剩余样本

for idx, file_path in enumerate(txt_files):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # 文件不足则全取
    sample_count = min(per_file_samples, len(lines))

    # 随机抽样
    candidate_indices = set(random.sample(range(len(lines)), sample_count))

    # 划分训练候选 & 剩余数据
    for i, line in enumerate(lines):
        if i in candidate_indices:
            all_candidates.append(line)
        else:
            test_samples.append(line)

    print(f"文件 {idx}: {len(lines)} 行 → 抽取 {sample_count} 条，剩余 {len(lines) - sample_count} 条")

print(f"\n总共抽取候选句子: {len(all_candidates)} 条")
print(f"测试集总句子数: {len(test_samples)} 条")

# 4. 从候选池抽1000条训练集
if len(all_candidates) < 1000:
    raise ValueError(f"候选句子数量({len(all_candidates)})不足1000条！")

train_samples = random.sample(all_candidates, 1000)

print(f"训练集: {len(train_samples)} 条")

# 5. 如果测试集超过 2000 → 随机抽 2000
if len(test_samples) > 2000:
    test_samples = random.sample(test_samples, 2000)

print(f"测试集最终数量: {len(test_samples)} 条")


# 6. 构建 DataFrame
def create_dataframe(samples):
    return pd.DataFrame({
        'text': samples,
        'language': [language] * len(samples),
        'style': [style] * len(samples)
    })


train_df = create_dataframe(train_samples)
test_df = create_dataframe(test_samples)


# 7. 保存 CSV（‼ 每个字段用引号包裹，没有转义符）
def save_csv_clean(df, save_path):
    df.to_csv(
        save_path,
        index=False,
        encoding='utf-8',
        quoting=csv.QUOTE_ALL  # 每个字段用引号 → 避免转义符
    )


# 保存
train_path = os.path.join(train_dir, "train_" + language + "_" + style + ".csv")
test_path = os.path.join(test_dir, "test_" + language + "_" + style + ".csv")

save_csv_clean(train_df, train_path)
save_csv_clean(test_df, test_path)

print("\n✅ 数据集划分完成！")
print(f"训练集路径: {train_path}")
print(f"测试集路径: {test_path}")
