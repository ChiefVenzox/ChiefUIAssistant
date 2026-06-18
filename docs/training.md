# Eğitim Rehberi

## Boru hattı

```
1) veri (seed üret veya kendi JSONL'lerini koy)   backend/datasets/*.jsonl
2) tokenizer eğit  ->  backend/checkpoints/tokenizer.json
3) veri hazırla    ->  backend/datasets/bin/{train,val}.bin + meta.json
4) model eğit      ->  backend/checkpoints/ckpt.pt
5) çalıştır        ->  backend (uvicorn) + frontend (vite)
```

## Komutlar (proje kökünden)

```bash
# 0) örnek tohum veri (8 kategori) — kendi verin varsa atla
python backend/datasets/build_seed.py

# 1) kendi BPE tokenizer'ımız
python scripts/train_tokenizer.py --input backend/datasets --vocab-size 16000

# 2) JSONL -> token bin
python scripts/prepare_dataset.py --input backend/datasets --out backend/datasets/bin

# 3) eğitim (GTX 1660 Ti)
python scripts/train_model.py --preset chiefui-30m --data backend/datasets/bin
#   hızlı deneme:
python scripts/train_model.py --preset chiefui-tiny --data backend/datasets/bin --max-steps 1000
#   devam:
python scripts/train_model.py --resume backend/checkpoints/ckpt.pt --data backend/datasets/bin
```

## Presetler

```bash
python -c "import sys; sys.path.insert(0,'backend'); from app.model.config import list_presets; list_presets()"
```

| preset | hedef | not |
|---|---|---|
| `chiefui-tiny` | ~7M | hızlı duman testi |
| `chiefui-30m` | ~30M | **1660 Ti için önerilen** |
| `chiefui-60m` | ~60M | daha güçlü, daha yavaş |

## 6 GB VRAM ipuçları

- Turing (1660 Ti) **fp16** kullanır (kod otomatik seçer; bf16 emülasyon → kapalı).
- OOM olursa: `--batch-size` düşür, `--grad-accum` artır; `gradient_checkpointing`
  zaten açık.
- `block_size` (context) 1024 → 512 yapmak da VRAM düşürür (config/preset).

## CPU fallback

GPU yoksa `--device cpu` ile çalışır (çok yavaş, sadece test).

## PC 2 worker (gelecek)

MVP tek PC. İleride PC 2:
- veri temizleme / tokenizasyon (CPU işleri),
- eval / önizleme ekran görüntüsü testleri,
- `torchrun` ile dağıtık (DDP) eğitim.
VRAM birleştirme **yok**; kararlılık önce. Kod `--device`, `--data`, `--out`
parametreleriyle worker olarak ayrı görev alacak şekilde yazıldı.
