import re
from datasets import load_dataset

def is_english(text):
    return re.fullmatch(r'[A-Za-z0-9\s.,:;!?\'"-]*', text) is not None

def split_and_filter_sentences(text):
    sentences = re.split(r'[.,:!]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    filtered = []

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        if sent and not sent[0].isdigit():
            words = sent.split()
            if 50 <= len(words) <= 100:
                filtered.append(' '.join(words))
            elif len(words) < 50:
                j = i + 1
                while j < len(sentences) and len(words) < 50:
                    next_words = sentences[j].split()
                    words += next_words
                    j += 1
                if 50 <= len(words) <= 100:
                    filtered.append(' '.join(words))
                i = j - 1
        i += 1
    return filtered


def process_arxiv(save_path, limit=100000, target_count=5000):
    ds = load_dataset("nick007x/arxiv-papers", split="train")
    output = []

    for i, item in enumerate(ds):
        if i >= limit:
            break
        abstract = item.get('abstract', '').strip()
        if not abstract:
            continue
        filtered_sents = split_and_filter_sentences(abstract)
        output.extend(filtered_sents)
        if len(output) >= target_count:
            break

    if len(output) < target_count:
        idx = 0
        while len(output) < target_count:
            output.append(output[idx % len(output)])
            idx += 1

    with open(save_path, 'w', encoding='utf-8') as f:
        for line in output[:target_count]:
            f.write(line + '\n')
    print(f"Saved {len(output[:target_count])} sentences to {save_path}")

if __name__ == "__main__":

    process_arxiv(r"G:\ECE\ECE684_NLP\final\style detector\temp\en_wiki\1.txt")
