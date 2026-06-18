"""
backend/app/tokenizer/train_tokenizer.py
=========================================
Kendi byte-level BPE tokenizer'ımızı UI/kod + doğal dil üzerinde sıfırdan eğitir.
Byte-level olduğu için HTML/CSS/JS sembolleri, Bootstrap sınıf adları, Türkçe
karakterler ve doğal dil sorunsuz işlenir; <unk> yoktur.

Girdi: datasets/ içindeki *.jsonl (instruction/input/output alanları) ve
opsiyonel *.txt/*.html/*.css/*.js dosyaları.

    python scripts/train_tokenizer.py --input backend/datasets --vocab-size 16000
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import tempfile

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from app.model.config import SPECIAL_TOKENS


def gather_texts(inputs):
    """JSONL alanlarını ve düz dosyaları geçici bir metin dosyasına toplar."""
    jsonl, raw = [], []
    for item in inputs:
        if os.path.isdir(item):
            jsonl += glob.glob(os.path.join(item, "**", "*.jsonl"), recursive=True)
            for ext in ("*.txt", "*.html", "*.css", "*.js"):
                raw += glob.glob(os.path.join(item, "**", ext), recursive=True)
        elif os.path.isfile(item):
            (jsonl if item.endswith(".jsonl") else raw).append(item)

    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    n = 0
    for fp in sorted(set(jsonl)):
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in ("instruction", "input", "output"):
                    val = obj.get(key)
                    if val:
                        tmp.write(val + "\n")
                        n += 1
    for fp in sorted(set(raw)):
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            tmp.write(f.read() + "\n")
            n += 1
    tmp.close()
    return tmp.name, n, len(jsonl), len(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", default=["backend/datasets"])
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--output", default="backend/checkpoints/tokenizer.json")
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    corpus, n, n_jsonl, n_raw = gather_texts(args.input)
    if n == 0:
        raise SystemExit(f"Eğitim metni bulunamadı: {args.input}")
    print(f"[tokenizer] {n_jsonl} jsonl + {n_raw} ham dosya, {n} metin parçası, "
          f"vocab={args.vocab_size}")

    tk = Tokenizer(BPE(unk_token=None))
    tk.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tk.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tk.train([corpus], trainer)
    os.unlink(corpus)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tk.save(args.output)
    print(f"[tokenizer] kaydedildi -> {args.output} (gerçek vocab={tk.get_vocab_size()})")
    sample = '<div class="container"><h1>Merhaba</h1></div> .btn{color:#c45a26}'
    enc = tk.encode(sample)
    print(f"[test] {len(enc.ids)} token -> geri: {tk.decode(enc.ids)!r}")


if __name__ == "__main__":
    main()
