# ChiefUI Assistant — Mimari

Yerel, sıfırdan eğitilen, UI/CSS üretimine odaklı bir AI asistanı. **Hiçbir hazır
LLM / bulut API kullanılmaz** (LLaMA, Qwen, Ollama, GGUF, OpenAI, Claude yok).
Model PyTorch ile sıfırdan eğitilir.

## Genel akış

```
                    ┌──────────────────────────── Frontend (React + Vite) ───────────────────────────┐
                    │  Prompt input → WebSocket/REST → token stream → HTML/CSS/notes sekmeleri        │
                    │  Three.js arka plan · canlı iframe önizleme · ZIP dışa aktar                     │
                    └───────────────┬──────────────────────────────────────────────────────────────┘
                                    │  POST /api/generate-ui  veya  WS /ws/generate
                    ┌───────────────▼──────────────────────────── Backend (FastAPI) ─────────────────┐
                    │  1. prompt al                                                                    │
                    │  2. prompt_format → model girişi (<|user|> ... <|assistant|>)                    │
                    │  3. model.generate_stream → token üretimi (CUDA, CPU fallback)                   │
                    │  4. parse_response → <html>/<css>/<notes> ayrıştır                               │
                    │  5. validate → temel HTML/CSS kontrolleri                                        │
                    │  6. SQLite'a kaydet (geçmiş)                                                      │
                    │  7. sonucu döndür / stream et                                                     │
                    └───────────────┬──────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────▼───────────────────────────┐
        │  AI çekirdeği (PyTorch, sıfırdan)                       │
        │  · decoder-only Transformer (RoPE+RMSNorm+SwiGLU)       │
        │  · kendi BPE tokenizer (HTML/CSS/JS/Bootstrap + NL)     │
        │  · checkpoint kaydet/yükle · generate                   │
        └─────────────────────────────────────────────────────────┘
```

## Model

Spec'inle birebir uyumlu, sıfırdan PyTorch decoder-only Transformer. Modern ama
küçük bileşenler (sıfırdan eğitimde standart vanilla'dan daha stabil/kaliteli):

| Bileşen | Seçim |
|---|---|
| Konum kodlama | RoPE (öğrenilen konum yok) |
| Norm | RMSNorm |
| MLP | SwiGLU (ffn_size ile boyutlanır) |
| Dikkat | MHA (küçük modelde `n_kv_head = n_head`), SDPA çekirdeği |
| Üretim | KV-cache (hızlı stream) |
| Bellek | fp16 (Turing) + gradient checkpointing |

Presetler (`backend/configs`):

| isim | ~params | layer | hidden | head | ffn | ctx |
|---|---|---|---|---|---|---|
| `chiefui-30m` | ~30M | 6 | 384 | 6 | 1536 | 1024 |
| `chiefui-60m` | ~60M | 8 | 512 | 8 | 2048 | 1024 |

İkisi de 6 GB VRAM'e rahat sığar (bkz. `scripts/vram_probe` mantığı).

## Veri sözleşmesi (API)

**REST** `POST /api/generate-ui`
```jsonc
// istek
{ "instruction": "Create a modern SaaS hero section",
  "input": "Style: dark, orange #c45a26, responsive",
  "max_new_tokens": 512, "temperature": 0.8, "top_k": 40, "top_p": 0.95 }
// yanıt
{ "id": 12, "html": "...", "css": "...", "notes": "...",
  "raw": "<response>...", "validation": { "ok": true, "issues": [] } }
```

**WebSocket** `WS /ws/generate` — istemci yukarıdaki istek JSON'unu yollar, sunucu:
```jsonc
{ "type": "token", "text": "<!DOCTYPE" }      // çok kez
{ "type": "done", "id": 12, "html": "...", "css": "...",
  "notes": "...", "validation": {...} }
```

`GET /api/history?limit=20` · `GET /api/health`

## Model giriş/çıkış formatı

Giriş (prompt_format):
```
<|user|>
Instruction: {instruction}
Style: {input}
<|assistant|>
```
Çıkış (modelin ürettiği, dataset'te de bu format):
```
@@HTML@@
...tam HTML belgesi (stil hariç)...
@@CSS@@
...CSS...
@@NOTES@@
Kısa tasarım açıklaması.
@@END@@
```
Üretim `@@END@@`, `<|end|>` veya `<|endoftext|>` görülünce durur. Backend
işaretçilere göre `html/css/notes` bölümlerini ayrıştırır (kesilmeye dayanıklı).
İşaretçiler `@@...@@` biçiminde, HTML/CSS içeriğiyle çakışmaz.

## Klasör yapısı

```
chiefui-assistant/
├─ backend/
│  ├─ app/
│  │  ├─ main.py                FastAPI uygulaması
│  │  ├─ api/routes.py          /api/generate-ui, /ws/generate, /api/history
│  │  ├─ model/
│  │  │  ├─ config.py           GPTConfig + presetler (chiefui-30m/60m)
│  │  │  └─ gpt.py              Transformer (doğrulanmış çekirdek)
│  │  ├─ tokenizer/
│  │  │  ├─ __init__.py         Tokenizer wrapper (yükle/encode/decode)
│  │  │  └─ train_tokenizer.py  BPE eğitimi
│  │  ├─ training/
│  │  │  ├─ dataset.py          JSONL loader + prompt format + tokenize→bin
│  │  │  └─ train.py            eğitim döngüsü + checkpoint
│  │  ├─ inference/
│  │  │  ├─ prompt_format.py    giriş kur + <response> ayrıştır
│  │  │  └─ generate.py         model yükle + üret (stream)
│  │  ├─ validation/validate.py HTML/CSS doğrulama
│  │  └─ database/db.py         SQLite geçmiş
│  ├─ configs/                  (ileride YAML; şimdilik model/config.py)
│  ├─ datasets/                 *.jsonl (eğitim verisi)
│  ├─ checkpoints/              (gitignore) eğitilmiş ağırlıklar
│  └─ requirements.txt
├─ frontend/                    React + Vite + Three.js
│  ├─ index.html · package.json · vite.config.js
│  └─ src/ (App, components/, three/, api/)
├─ scripts/                     train_tokenizer.py, train_model.py,
│                               generate.py, prepare_dataset.py (ince CLI sarmalları)
└─ docs/                        architecture.md, training.md, dataset-format.md
```

## PC 2 (gelecek worker) — sadece yapı, MVP'de implement edilmez

Kod, ileride PC 2'nin şu işleri almasına izin verecek şekilde modüler:
- `training/dataset.py` → veri temizleme / tokenizasyon işleri (CPU-bound, GPU gerekmez)
- `inference/` → eval / önizleme ekran görüntüsü testleri
- `train.py` → ileride `torchrun --nproc_per_node` ile dağıtık (DDP) eğitim;
  şimdilik tek-GPU. **VRAM birleştirme YOK** (kararlılık önce).

Config'te `device`, `data_dir`, `out_dir` dışarıdan verilebilir; böylece PC 2
ayrı bir worker olarak aynı kod tabanını farklı görevle çalıştırır.

## Dürüst beklenti

Sıfırdan ~30–60M model, **küçük** veriyle çalışır ama çıktı kalitesi başta
**basittir**. Kaliteli HTML/CSS üretimi için binlerce iyi örnek + uzun eğitim
gerekir. MVP'nin amacı: uçtan uca **çalışan** bir boru hattı ve mimari.
```
```
