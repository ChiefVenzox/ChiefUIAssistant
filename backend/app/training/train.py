"""
backend/app/training/train.py
=============================
Modeli sıfırdan eğitir. CUDA (Turing'de fp16) + CPU fallback. 6 GB dostu:
mixed precision, gradient checkpointing, grad accumulation, cosine LR, checkpoint.

    python scripts/train_model.py --preset chiefui-30m --data backend/datasets/bin
    python scripts/train_model.py --resume backend/checkpoints/ckpt.pt --data backend/datasets/bin
"""
from __future__ import annotations
import argparse
import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, fields

import numpy as np
import torch

from app.model.config import GPTConfig, get_config
from app.model import GPT


def build_cfg_from_dict(d: dict) -> GPTConfig:
    valid = {f.name for f in fields(GPTConfig)}
    return GPTConfig(**{k: v for k, v in d.items() if k in valid})


def load_meta(data_dir):
    p = os.path.join(data_dir, "meta.json")
    if not os.path.exists(p):
        raise SystemExit(f"meta.json yok: {p}\nÖnce: python scripts/prepare_dataset.py")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.lr_decay_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, cfg.lr_decay_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + np.cos(np.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def save_ckpt(model, optimizer, cfg, step, best_val, out_dir, name):
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model": raw.state_dict(), "optimizer": optimizer.state_dict(),
        "config": asdict(cfg), "step": step, "best_val": best_val,
    }, os.path.join(out_dir, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="chiefui-30m")
    ap.add_argument("--data", default="backend/datasets/bin")
    ap.add_argument("--out", default="backend/checkpoints")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr-decay-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    meta = load_meta(args.data)
    np_dtype = np.dtype(meta["dtype"])

    start_step, best_val, ckpt = 0, float("inf"), None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = build_cfg_from_dict(ckpt["config"])
        start_step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val", float("inf"))
        print(f"[train] devam: {args.resume} (step={start_step})")
    else:
        cfg = get_config(args.preset)

    cfg.vocab_size = meta["vocab_size"]
    cfg.out_dir = args.out
    if args.max_steps:
        cfg.max_steps = args.max_steps
        cfg.lr_decay_steps = args.max_steps
    if args.lr_decay_steps:
        cfg.lr_decay_steps = args.lr_decay_steps
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.grad_accum:
        cfg.grad_accum_steps = args.grad_accum

    os.makedirs(cfg.out_dir, exist_ok=True)
    device = args.device
    device_type = "cuda" if "cuda" in device else "cpu"

    if device_type == "cuda":
        # Turing (cc 7.5) bf16'yı sadece emüle eder (yavaş); gerçek bf16 cc>=8.
        major = torch.cuda.get_device_capability()[0]
        bf16_ok = torch.cuda.is_bf16_supported() and major >= 8
        ptdtype = torch.bfloat16 if bf16_ok else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=ptdtype)
        use_scaler = (ptdtype == torch.float16)
        print(f"[train] GPU: {torch.cuda.get_device_name(0)} | autocast={ptdtype} | scaler={use_scaler}")
    else:
        ptdtype, ctx, use_scaler = torch.float32, nullcontext(), False
        print("[train] UYARI: CUDA yok, CPU (sadece test).")

    _GradScaler = getattr(torch.amp, "GradScaler", None) or torch.cuda.amp.GradScaler
    scaler = _GradScaler(enabled=use_scaler)

    def get_batch(split):
        data = np.memmap(os.path.join(args.data, f"{split}.bin"), dtype=np_dtype, mode="r")
        max_start = len(data) - cfg.block_size - 1
        if max_start < 1:
            raise SystemExit(f"{split}.bin çok küçük (block_size={cfg.block_size}).")
        ix = torch.randint(max_start, (cfg.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
        if device_type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    model = GPT(cfg).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    print(f"[train] {model.num_params()/1e6:.1f}M param (preset={args.preset}, "
          f"vocab={cfg.vocab_size}, ctx={cfg.block_size})")

    optimizer = model.configure_optimizers(cfg, device_type)
    if ckpt is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[train] optimizer durumu yüklenemedi ({e})")

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                x, y = get_batch(split)
                with ctx:
                    _, loss, _ = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    model.train()
    best_saved = False  # bu koşuda eval-best ckpt.pt yazıldı mı?
    tokens_per_step = cfg.batch_size * cfg.block_size * cfg.grad_accum_steps
    t0, running = time.time(), None
    print(f"[train] efektif batch = {cfg.batch_size}x{cfg.grad_accum_steps} | "
          f"{tokens_per_step:,} token/adım")

    for step in range(start_step, cfg.max_steps):
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch("train")
            with ctx:
                _, loss, _ = model(x, y)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        lossf = loss.item() * cfg.grad_accum_steps
        running = lossf if running is None else 0.9 * running + 0.1 * lossf
        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tps = tokens_per_step * cfg.log_interval / dt if step > start_step else 0
            mem = (torch.cuda.max_memory_allocated() / 1e9) if device_type == "cuda" else 0
            print(f"adım {step:>6} | loss {lossf:6.3f} (ort {running:6.3f}) | "
                  f"lr {lr:.2e} | {tps:7.0f} tok/s | VRAM {mem:4.2f} GB")

        if step > start_step and step % cfg.eval_interval == 0:
            ev = estimate_loss()
            print(f"  >> eval | train {ev['train']:.3f} | val {ev['val']:.3f}")
            if ev["val"] < best_val:
                best_val = ev["val"]
                save_ckpt(model, optimizer, cfg, step, best_val, cfg.out_dir, "ckpt.pt")
                best_saved = True
                print(f"  >> en iyi val ({best_val:.3f}) -> {cfg.out_dir}/ckpt.pt")
        if step > start_step and step % cfg.save_interval == 0:
            save_ckpt(model, optimizer, cfg, step, best_val, cfg.out_dir, "ckpt_last.pt")

    save_ckpt(model, optimizer, cfg, cfg.max_steps, best_val, cfg.out_dir, "ckpt_last.pt")
    # Bu koşuda eval-best ckpt.pt yazılmadıysa (kısa koşu / eval hiç çalışmadı),
    # son modeli ckpt.pt olarak YAZ (eski/bayat bir ckpt.pt'yi de üzerine yaz).
    if not best_saved:
        save_ckpt(model, optimizer, cfg, cfg.max_steps, best_val, cfg.out_dir, "ckpt.pt")
        print("[train] eval-best yok; son model ckpt.pt olarak kaydedildi.")
    print(f"[train] bitti -> {cfg.out_dir}/")


if __name__ == "__main__":
    main()
