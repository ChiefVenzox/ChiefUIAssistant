"""
backend/app/inference/generate.py
=================================
Eğitilmiş checkpoint'i yükleyip UI kodu üretir. Hem CLI hem FastAPI buradaki
fonksiyonları kullanır. CUDA + CPU fallback.
"""
from __future__ import annotations
import sys
from dataclasses import fields
from typing import Dict, Iterator, Optional

import torch

from app.model.config import GPTConfig
from app.model import GPT
from app.tokenizer import load_tokenizer, Tokenizer
from app.inference.prompt_format import build_input_ids, parse_response

STOP_TEXT = "@@END"


def _cfg_from_dict(d: dict) -> GPTConfig:
    valid = {f.name for f in fields(GPTConfig)}
    return GPTConfig(**{k: v for k, v in d.items() if k in valid})


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = _cfg_from_dict(ckpt["config"])
    cfg.gradient_checkpointing = False
    cfg.dropout = 0.0
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def stream_ui(model, tok: Tokenizer, instruction: str, input_text: Optional[str],
              device: str, max_new_tokens: int = 512, temperature: float = 0.8,
              top_k: int = 40, top_p: float = 0.95) -> Iterator[str]:
    """UI çıktısını parça parça (UTF-8 güvenli) yield eder. </response> görünce durur."""
    ids = build_input_ids(tok, instruction, input_text)
    max_ctx = max(1, model.block_size - max_new_tokens)
    if len(ids) > max_ctx:
        ids = ids[len(ids) - max_ctx:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    generated, prev_text = [], ""
    for tok_tensor in model.generate_stream(
        idx, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, eos_token_id=None, use_cache=True,
    ):
        tid = int(tok_tensor.item())
        if tid in tok.stop_ids:
            break
        generated.append(tid)
        text = tok.decode(generated, skip_special=True)
        if text.endswith("�"):     # yarım UTF-8; sonraki token'i bekle
            continue
        new = text[len(prev_text):]
        prev_text = text
        if new:
            yield new
        if STOP_TEXT in text:            # yapısal çıktı tamamlandı
            break


def generate_ui(model, tok, instruction, input_text, device, **kw) -> Dict[str, str]:
    full = "".join(stream_ui(model, tok, instruction, input_text, device, **kw))
    return parse_response(full)


def _main():
    import argparse
    for s in (sys.stdout, sys.stderr):       # Windows konsol UTF-8
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="backend/checkpoints/ckpt.pt")
    ap.add_argument("--tokenizer", default="backend/checkpoints/tokenizer.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--instruction", default="Create a responsive hero section")
    ap.add_argument("--input", default="Style: dark, orange accent #c45a26")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    model, cfg = load_model(args.ckpt, args.device)
    print(f"[generate] {model.num_params()/1e6:.0f}M, ctx={cfg.block_size}, {args.device}\n")
    for piece in stream_ui(model, tok, args.instruction, args.input, args.device,
                           max_new_tokens=args.max_new_tokens, temperature=args.temperature):
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    _main()
