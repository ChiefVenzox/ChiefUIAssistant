"""
backend/app/model/config.py
============================
ChiefUI modelinin tüm mimari/eğitim ayarları. Tek yerden ölçeklenir.
Hiçbir hazır model/ağırlık kullanılmaz — bu sayılarla model sıfırdan kurulur.

    from app.model.config import get_config
    cfg = get_config("chiefui-30m")
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json

# Sohbet/talimat şablonu için özel tokenlar (tokenizer ile paylaşılır).
# NOT: <response>/<html>/<css>/<notes> ÖZEL token DEĞİL; modelin öğrendiği düz metin.
SPECIAL_TOKENS = [
    "<|endoftext|>",   # dizi sonu (id 0)
    "<|user|>",        # kullanıcı turu
    "<|assistant|>",   # asistan turu
    "<|system|>",      # sistem mesajı
    "<|end|>",         # tur sonu
]


@dataclass
class GPTConfig:
    # ---- Mimari ----
    vocab_size: int = 16000
    block_size: int = 1024          # context_length
    n_layer: int = 6
    n_head: int = 8
    n_kv_head: int = 8              # = n_head -> tam MHA; < n_head -> GQA
    n_embd: int = 512              # hidden_size
    ffn_mult: float = 8 / 3        # SwiGLU ara katman çarpanı (~2.667)
    rope_theta: float = 10000.0
    dropout: float = 0.0
    rms_eps: float = 1e-5
    tie_embeddings: bool = True

    # ---- Eğitim (6 GB dostu) ----
    batch_size: int = 8
    grad_accum_steps: int = 16
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200
    max_steps: int = 30000
    lr_decay_steps: int = 30000
    gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = False

    # ---- Değerlendirme / kayıt ----
    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 10
    save_interval: int = 1000
    out_dir: str = "checkpoints"

    def _ffn_hidden(self) -> int:
        h = int(self.ffn_mult * self.n_embd)
        return ((h + 63) // 64) * 64

    def n_params(self) -> int:
        v, d, L = self.vocab_size, self.n_embd, self.n_layer
        head_dim = d // self.n_head
        kv_dim = self.n_kv_head * head_dim
        attn = d * d + 2 * d * kv_dim + d * d
        mlp = 3 * d * self._ffn_hidden()
        total = v * d + L * (attn + mlp)
        if not self.tie_embeddings:
            total += v * d
        return total

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


# ---- Presetler (isimler yaklaşık; gerçek param sayısı n_params() ile) -------
_PRESETS = {
    # Hızlı duman testi için (~7M)
    "chiefui-tiny": dict(
        n_layer=4, n_head=6, n_kv_head=6, n_embd=384, block_size=1024,
        vocab_size=8000, batch_size=16, grad_accum_steps=8,
        max_steps=8000, lr_decay_steps=8000,
    ),
    # ~30M hedef (1660 Ti için önerilen başlangıç)
    "chiefui-30m": dict(
        n_layer=6, n_head=8, n_kv_head=8, n_embd=512, block_size=1024,
        vocab_size=16000, batch_size=8, grad_accum_steps=16,
        max_steps=40000, lr_decay_steps=40000,
    ),
    # ~60M hedef
    "chiefui-60m": dict(
        n_layer=8, n_head=10, n_kv_head=10, n_embd=640, block_size=1024,
        vocab_size=16000, batch_size=6, grad_accum_steps=24,
        max_steps=60000, lr_decay_steps=60000,
    ),
}

DEFAULT_PRESET = "chiefui-30m"


def get_config(preset: str = DEFAULT_PRESET, **overrides) -> GPTConfig:
    if preset not in _PRESETS:
        raise ValueError(f"Bilinmeyen preset: {preset!r}. Seçenekler: {list(_PRESETS)}")
    params = dict(_PRESETS[preset])
    params.update(overrides)
    return GPTConfig(**params)


def list_presets() -> None:
    print(f"{'preset':<15}{'~params':>10}{'layer':>7}{'hidden':>8}{'head':>6}{'ffn':>7}{'ctx':>7}")
    print("-" * 60)
    for name in _PRESETS:
        c = get_config(name)
        p = c.n_params()
        human = f"{p/1e6:.0f}M" if p < 1e9 else f"{p/1e9:.2f}B"
        print(f"{name:<15}{human:>10}{c.n_layer:>7}{c.n_embd:>8}"
              f"{c.n_head:>6}{c._ffn_hidden():>7}{c.block_size:>7}")


if __name__ == "__main__":
    list_presets()
