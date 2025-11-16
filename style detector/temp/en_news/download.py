import re
from datasets import load_dataset

def is_english(text):
    """检查文本是否只包含英文字符和常见标点"""
    try:
        text.encode('ascii')
    except UnicodeEncodeError:
        return False
    return True

def split_text_keep_delimiters(text):
    """
    将文本按标点分割，同时保留分隔符。
    返回 [(句子, 分隔符), ...]
    """
    pattern = r'([.,:!])'
    parts = re.split(pattern, text)
    sentences = []
    for i in range(0, len(parts)-1, 2):
        sent = parts[i].strip()
        delim = parts[i+1]
        if sent:
            sentences.append((sent, delim))
    # 如果最后没有分隔符
    if len(parts) % 2 != 0 and parts[-1].strip():
        sentences.append((parts[-1].strip(), ""))
    return sentences

def process_dataset(ds, max_samples=30000, target_count=5000, min_words=50, max_words=100):
    """
    对单个数据集进行处理，返回 label -> sentences 字典
    """
    collected = {0: [], 1: [], 2: [], 3: []}
    label_counts = {0: 0, 1:0, 2:0, 3:0}
    buffer = {0:"", 1:"", 2:"", 3:""}  # 用于跨 text 拼接
    for i, item in enumerate(ds):
        if i >= max_samples:
            break

        text = item["text"].strip()
        label = item["label"]
        if "=" in text:
            continue
        if not is_english(text):
            continue
        # 舍弃标题
        if "- " in text:
            text = text.split("- ", 1)[1].strip()

        # 分割文本
        sentences = split_text_keep_delimiters(text)
        for sent, delim in sentences:
            # 舍弃以数字开头
            if re.match(r'^\d', sent):
                continue
            # 拼接 buffer
            cur_text = (buffer[label] + " " + sent + delim).strip() if buffer[label] else (sent + delim)
            words = cur_text.split()
            if len(words) > max_words:
                buffer[label] = ""  # 舍弃过长句
                continue
            elif len(words) >= min_words:
                collected[label].append(cur_text)
                buffer[label] = ""
            else:
                buffer[label] = cur_text
        # 检查总量是否已够
        total_collected = sum(len(v) for v in collected.values())
        if total_collected >= target_count:
            break

    # 最后处理 buffer
    for label, buf_text in buffer.items():
        if buf_text:
            words = buf_text.split()
            if min_words <= len(words) <= max_words:
                collected[label].append(buf_text)

    # 平衡采样
    final_sentences = []
    per_label = target_count // 4
    for label in range(4):
        final_sentences.extend(collected[label][:per_label])

    return final_sentences

def main():
    dataset_path = r"1.txt"
    ds = load_dataset("fancyzhx/ag_news", split="train")
    max_samples = 30000
    target_count = 5000

    final_sentences = process_dataset(ds, max_samples=max_samples, target_count=target_count)

    with open(dataset_path, "w", encoding="utf-8") as f:
        for sent in final_sentences:
            f.write(sent + "\n")
    print(f"Saved {len(final_sentences)} sentences to {dataset_path}")

if __name__ == "__main__":
    main()
