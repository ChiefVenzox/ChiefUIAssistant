"""
backend/app/training/dataset.py
===============================
JSONL talimat verisini ( {instruction, input, output} ) token id dizisine çevirip
eğitim için ikili (binary) dosyalara yazar. Örnekler <|user|>..<|assistant|>..
formatında paketlenir, aralarına <|endoftext|> konur.

Çıktı: <out>/train.bin, <out>/val.bin, <out>/meta.json

    python scripts/prepare_dataset.py --input backend/datasets --out backend/datasets/bin
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from typing import Dict, Iterator

import numpy as np
from tqdm import tqdm

from app.tokenizer import load_tokenizer
from app.inference.prompt_format import build_training_ids


def iter_examples(inputs) -> Iterator[Dict]:
    files = []
    for item in inputs:
        if os.path.isdir(item):
            files += glob.glob(os.path.join(item, "**", "*.jsonl"), recursive=True)
        elif os.path.isfile(item) and item.endswith(".jsonl"):
            files.append(item)
    files = sorted(set(f for f in files if "/bin/" not in f.replace("\\", "/")))
    if not files:
        raise SystemExit(f"JSONL bulunamadı: {inputs}")
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def build_bins(inputs, tokenizer_path, out_dir, val_ratio=0.05):
    tok = load_tokenizer(tokenizer_path)
    dtype = np.uint16 if tok.vocab_size <= 65535 else np.uint32

    all_ids, n_ex = [], 0
    for ex in tqdm(iter_examples(inputs), desc="kodlanıyor"):
        ids = build_training_ids(tok, ex)
        all_ids.extend(ids)
        all_ids.append(tok.eot_id)
        n_ex += 1

    if not all_ids:
        raise SystemExit("Hiç örnek kodlanamadı.")

    arr = np.array(all_ids, dtype=dtype)
    n = len(arr)
    n_val = max(1, int(n * val_ratio)) if n > 200 else 0
    train, val = arr[: n - n_val], arr[n - n_val:]

    os.makedirs(out_dir, exist_ok=True)
    train.tofile(os.path.join(out_dir, "train.bin"))
    (val if n_val else train).tofile(os.path.join(out_dir, "val.bin"))

    meta = {
        "dtype": np.dtype(dtype).name, "vocab_size": tok.vocab_size,
        "n_examples": n_ex, "n_tokens": int(n),
        "n_train": int(len(train)), "n_val": int(n_val),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[data] {n_ex} örnek, {n:,} token (train={len(train):,}, val={n_val:,})")
    print(f"[data] yazıldı -> {out_dir}/")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", default=["backend/datasets"])
    ap.add_argument("--tokenizer", default="backend/checkpoints/tokenizer.json")
    ap.add_argument("--out", default="backend/datasets/bin")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    args = ap.parse_args()
    build_bins(args.input, args.tokenizer, args.out, args.val_ratio)


if __name__ == "__main__":
    main()
