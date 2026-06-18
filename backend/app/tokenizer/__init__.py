"""
backend/app/tokenizer
======================
Kendi eğittiğimiz byte-level BPE tokenizer'ı yükleyip kullanmak için wrapper.
`tokenizers` kütüphanesi bir TOKENIZER aracıdır (LLM değil); kelime dağarcığını
kendi UI/kod verimiz üzerinde eğitiriz.
"""
from __future__ import annotations
import os
from typing import List

from tokenizers import Tokenizer as _HFTokenizer

from app.model.config import SPECIAL_TOKENS

DEFAULT_PATH = "checkpoints/tokenizer.json"


class Tokenizer:
    def __init__(self, path: str = DEFAULT_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Tokenizer yok: {path}\n"
                f"Önce: python scripts/train_tokenizer.py"
            )
        self.tk = _HFTokenizer.from_file(path)
        self.path = path
        self.vocab_size = self.tk.get_vocab_size()
        self.eot_id = self._id("<|endoftext|>")
        self.user_id = self._id("<|user|>")
        self.assistant_id = self._id("<|assistant|>")
        self.system_id = self._id("<|system|>")
        self.end_id = self._id("<|end|>")
        self.stop_ids = {self.eot_id, self.end_id}

    def _id(self, tok: str) -> int:
        i = self.tk.token_to_id(tok)
        if i is None:
            raise ValueError(f"Özel token tokenizer'da yok: {tok}")
        return i

    def encode(self, text: str) -> List[int]:
        return self.tk.encode(text).ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        return self.tk.decode(ids, skip_special_tokens=skip_special)


def load_tokenizer(path: str = DEFAULT_PATH) -> Tokenizer:
    return Tokenizer(path)
