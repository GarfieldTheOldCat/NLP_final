from datasets import load_dataset
import os
import re


def split_into_sentences(text):
    """按标点符号分割句子"""
    # 按.!?分割，保留标点
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # 清理空字符串和多余空格
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def count_words(text):
    """统计词数（按空格分割）"""
    return len(text.split())


def process_article(sentences):
    """处理单篇文章的句子列表"""
    processed = []
    buffer = ""

    for sentence in sentences:
        word_count = count_words(sentence)

        # 情况3.1: 50-100词，直接保留
        if 50 <= word_count <= 100:
            # 如果buffer有内容，先处理buffer
            if buffer:
                buffer_word_count = count_words(buffer)
                if 50 <= buffer_word_count <= 100:
                    processed.append(buffer.strip())
                buffer = ""
            # 保留当前句子
            processed.append(sentence)

        # 情况3.2: 小于50词，加入buffer
        elif word_count < 50:
            # 拼接句子（注意添加空格）
            buffer = buffer + " " + sentence if buffer else sentence
            buffer_word_count = count_words(buffer)

            # 如果buffer达到50-100词，保留并清空
            if 50 <= buffer_word_count <= 100:
                processed.append(buffer.strip())
                buffer = ""
            # 如果buffer超过100词，舍弃并清空
            elif buffer_word_count > 100:
                buffer = ""

        # 情况3.3: 大于100词，舍弃
        else:
            # 如果buffer有内容，先检查是否保留
            if buffer:
                buffer_word_count = count_words(buffer)
                if 50 <= buffer_word_count <= 100:
                    processed.append(buffer.strip())
                buffer = ""
            # 当前句子大于100词，直接舍弃
            continue

    # 循环结束后处理剩余buffer
    if buffer:
        buffer_word_count = count_words(buffer)
        # 只有达到50-100词才保留，否则舍弃
        if 50 <= buffer_word_count <= 100:
            processed.append(buffer.strip())

    return processed


# 加载数据集
ds = load_dataset("ACOSharma/literature")

# 定义保存路径
save_dir = r"G:\ECE\ECE684_NLP\final\style detector\temp\en_literature"
os.makedirs(save_dir, exist_ok=True)

# 处理前35篇文章
for idx in range(35):
    # 获取文章内容
    article_text = ds['train'][idx]['text']

    # 分割句子
    sentences = split_into_sentences(article_text)

    # 处理句子
    processed_sentences = process_article(sentences)

    # 保存到文件
    file_path = os.path.join(save_dir, f"{idx}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in processed_sentences:
            f.write(sentence + '\n')

    print(f"文章 {idx}: {len(sentences)} 句 → {len(processed_sentences)} 句")

print(f"\n✅ 处理完成！文件已保存到: {save_dir}")