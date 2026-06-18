# ChiefUI Assistant

Yerel, **sıfırdan** eğitilen, UI/CSS üretimine odaklı bir AI asistanı.
Hiçbir hazır LLM / bulut API kullanılmaz (LLaMA, Qwen, Ollama, GGUF, OpenAI,
Claude yok). Model PyTorch ile sıfırdan eğitilir.

> Donanım hedefi: NVIDIA CUDA GPU (~6 GB, GTX 1660 Ti). CPU fallback testte çalışır.
> MVP tek PC; mimari ileride PC 2'yi worker olarak eklemeye uygun (bkz. docs).

Prompt → `<html>/<css>/<notes>` yapısal çıktı → canlı önizleme + kod sekmeleri + ZIP.

## Mimari

Detay: [docs/architecture.md](docs/architecture.md) ·
veri: [docs/dataset-format.md](docs/dataset-format.md) ·
eğitim: [docs/training.md](docs/training.md)

```
chiefui-assistant/
├─ backend/    FastAPI + PyTorch model (sıfırdan GPT) + tokenizer + SQLite
├─ frontend/   React + Vite + Three.js (prompt, canlı önizleme, kod sekmeleri, ZIP)
├─ scripts/    train_tokenizer · prepare_dataset · train_model · generate
└─ docs/
```

## Kurulum

### 1) Backend (Python)

```bash
cd chiefui-assistant

# torch'u CUDA'na göre kur (GTX 1660 Ti / Turing -> cu126 doğrulandı):
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r backend/requirements.txt
```
> Aynı wheel hem GPU hem CPU çalışır. GPU yoksa `--device cpu` kullan.

### 2) Frontend (Node)

```bash
cd frontend
npm install
```

## Çalıştırma (MVP, uçtan uca)

Proje kökünden (`chiefui-assistant/`):

```bash
# 1) tohum veri (8 kategori) — kendi verin varsa atla
python backend/datasets/build_seed.py

# 2) kendi tokenizer'ımız
python scripts/train_tokenizer.py --input backend/datasets --vocab-size 16000

# 3) JSONL -> token bin
python scripts/prepare_dataset.py --input backend/datasets --out backend/datasets/bin

# 4) eğit (hızlı deneme: tiny + az adım)
python scripts/train_model.py --preset chiefui-tiny --data backend/datasets/bin --max-steps 1000
#   gerçek: --preset chiefui-30m  (uzun sürer)

# 5a) backend'i başlat
cd backend
uvicorn app.main:app --port 8000
#   (model checkpoints/ckpt.pt + checkpoints/tokenizer.json'dan yüklenir)

# 5b) ayrı terminalde frontend
cd frontend
npm run dev
#   -> tarayıcı: http://localhost:5173
```

Terminalden test (backend gerekmez):
```bash
python scripts/generate.py --instruction "Create a responsive hero section" \
  --input "Style: dark, orange #c45a26"
```

## API

- `POST /api/generate-ui` → `{ id, html, css, notes, raw, validation }`
- `WS /ws/generate` → `{type:"token",text}` … `{type:"done", html, css, notes, validation, id}`
- `GET /api/history?limit=20` · `GET /api/health`

## Dürüst beklenti

Sıfırdan ~30M model, küçük veriyle **çalışır ama çıktı başta basittir**. Kaliteli
HTML/CSS için **binlerce** iyi örnek + uzun eğitim gerekir. MVP'nin amacı: uçtan
uca çalışan boru hattı ve temiz, geliştirilebilir mimari. Veriyi büyütmek en
yüksek etkili adımdır.
```
```
