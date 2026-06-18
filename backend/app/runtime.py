"""
backend/app/runtime.py
======================
Yüklü modeli/tokenizer'ı tutan paylaşılan durum. main.py açılışta load() çağırır;
api/routes.py STATE'i okur.
"""
from __future__ import annotations
import os

STATE: dict = {}


def load(ckpt_path: str, tokenizer_path: str, device: str):
    import torch
    from app.inference.generate import load_model
    from app.tokenizer import load_tokenizer

    if not os.path.exists(ckpt_path):
        print(f"[runtime] checkpoint yok ({ckpt_path}) — önce eğit. Model yüklenmedi.")
        STATE["ready"] = False
        return
    if not os.path.exists(tokenizer_path):
        print(f"[runtime] tokenizer yok ({tokenizer_path}). Model yüklenmedi.")
        STATE["ready"] = False
        return

    if device == "cuda" and not torch.cuda.is_available():
        print("[runtime] CUDA yok, CPU'ya düşülüyor.")
        device = "cpu"

    tok = load_tokenizer(tokenizer_path)
    model, cfg = load_model(ckpt_path, device)
    STATE.update(model=model, tok=tok, cfg=cfg, device=device, ready=True)
    print(f"[runtime] model hazır: {model.num_params()/1e6:.0f}M, device={device}")
