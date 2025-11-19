import os
import random
import csv

def split_txt_to_csv(txt_path, language, style):
    # 输出目录
    train_dir = r"G:\ECE\ECE684_NLP\final\style_detector\corpus\train"
    test_dir  = r"G:\ECE\ECE684_NLP\final\style_detector\corpus\test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 读取 txt 文件
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 如果 txt 行数少于 3000，则全部使用
    sample_size = min(3000, len(lines))
    sampled = random.sample(lines, sample_size)

    # 划分训练集 1000 行 / 测试集 2000 行
    train_size = min(1000, len(sampled))
    test_size = sample_size - train_size

    train_data = sampled[:train_size]
    test_data = sampled[train_size:]

    # 保存训练集 CSV
    train_path = os.path.join(train_dir, f"train_{language}_{style}.csv")
    with open(train_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["text","language","style"])
        for line in train_data:
            writer.writerow([line, language, style])

    # 保存测试集 CSV
    test_path = os.path.join(test_dir, f"test_{language}_{style}.csv")
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["text","language","style"])
        for line in test_data:
            writer.writerow([line, language, style])

    print(f"完成！训练集: {len(train_data)} 行, 测试集: {len(test_data)} 行")
    print(f"训练集保存: {train_path}")
    print(f"测试集保存: {test_path}")


# ===== 示例用法 =====
txt_file = r"G:\ECE\ECE684_NLP\final\style_detector\corpus\temp\news.txt"
language = "ch"      # en 或 ch
style = "news" # law / literature / science / news

split_txt_to_csv(txt_file, language, style)
