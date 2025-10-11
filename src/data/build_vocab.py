import os, json
from collections import Counter
from typing import Iterable
from src.data.icdar24 import split_into_chars

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>"]  # pad, bos, eos

def build_char_vocab(text_files: Iterable[str], min_freq: int = 1):
    counter = Counter()
    for fp in text_files:
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read().strip()
        for ch in split_into_chars(text):
            counter[ch] += 1
    chars = [c for c, n in counter.items() if n >= min_freq]
    chars = sorted(chars)
    vocab = SPECIAL_TOKENS + chars
    stoi = {s:i for i, s in enumerate(vocab)}
    itos = {i:s for s, i in stoi.items()}
    return {"vocab": vocab, "stoi": stoi, "itos": itos,
            "freq": {k: int(v) for k, v in counter.items()}}

def save_vocab(vocab_dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_dict["vocab"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "stoi.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_dict["stoi"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "freq.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_dict["freq"], f, ensure_ascii=False, indent=2)
    print(f"Saved vocab with {len(vocab_dict['vocab'])} tokens to {out_dir}")
