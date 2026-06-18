# ChiefUI Assistant

**English** | [Türkçe](README.md)

A local, **from-scratch** AI assistant focused on UI/CSS generation.
**No pretrained LLM / cloud API is used** (no LLaMA, Qwen, Ollama, GGUF, OpenAI,
Claude). The model is trained from scratch in PyTorch.

> Hardware target: NVIDIA CUDA GPU (~6 GB, e.g. **GTX 1660 Ti** / Turing).
> CPU fallback works for testing. MVP is single-PC; the architecture is ready to
> add a 2nd PC as a worker later (see docs / wiki).

Prompt → structured `@@HTML / @@CSS / @@NOTES` output → live iframe preview + code
tabs + ZIP export.

📖 **[Wiki](https://github.com/ChiefVenzox/ChiefUIAssistant/wiki)** —
[Architecture](https://github.com/ChiefVenzox/ChiefUIAssistant/wiki/Architecture) ·
[Training](https://github.com/ChiefVenzox/ChiefUIAssistant/wiki/Training) ·
[Dataset Format](https://github.com/ChiefVenzox/ChiefUIAssistant/wiki/Dataset-Format) ·
[API](https://github.com/ChiefVenzox/ChiefUIAssistant/wiki/API)

## Architecture

Docs: [docs/architecture.md](docs/architecture.md) ·
data: [docs/dataset-format.md](docs/dataset-format.md) ·
training: [docs/training.md](docs/training.md)

```
chiefui-assistant/
├─ backend/    FastAPI + PyTorch model (from-scratch GPT) + tokenizer + SQLite
├─ frontend/   React + Vite + Three.js (prompt, live preview, code tabs, ZIP)
├─ scripts/    train_tokenizer · prepare_dataset · train_model · generate
└─ docs/
```

The model is a small decoder-only Transformer built from scratch:
**RoPE** positions, **RMSNorm**, **SwiGLU** MLP, **(G)MHA + SDPA**, **KV-cache**,
trained with **fp16 + gradient checkpointing** to fit 6 GB.

## Setup

### 1) Backend (Python)

```bash
cd chiefui-assistant

# Install torch for your CUDA (GTX 1660 Ti / Turing -> cu126 verified):
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r backend/requirements.txt
```
> The same wheel runs on both GPU and CPU. No GPU? Use `--device cpu`.

### 2) Frontend (Node)

```bash
cd frontend
npm install
```

## Run (MVP, end-to-end)

From the project root (`chiefui-assistant/`):

```bash
# 1) seed data (8 balanced categories) — skip if you have your own data
python backend/datasets/build_seed.py

# 2) train our own tokenizer
python scripts/train_tokenizer.py --input backend/datasets --vocab-size 16000

# 3) JSONL -> token bins
python scripts/prepare_dataset.py --input backend/datasets --out backend/datasets/bin

# 4) train (quick try: tiny + few steps)
python scripts/train_model.py --preset chiefui-tiny --data backend/datasets/bin --max-steps 1000
#   real run: --preset chiefui-30m  (takes longer)

# 5a) start the backend
cd backend
uvicorn app.main:app --port 8000
#   (model loads from checkpoints/ckpt.pt + checkpoints/tokenizer.json)

# 5b) frontend in a separate terminal
cd frontend
npm run dev
#   -> browser: http://localhost:5173
```

Test from the terminal (no backend needed):
```bash
python scripts/generate.py --instruction "Create a responsive hero section" \
  --input "Style: dark, orange #c45a26"
```

## API

- `POST /api/generate-ui` → `{ id, html, css, notes, raw, validation }`
- `WS /ws/generate` → `{type:"token",text}` … `{type:"done", html, css, notes, validation, id}`
- `GET /api/history?limit=20` · `GET /api/health`

## Presets

| preset | ~params | note |
|---|---|---|
| `chiefui-tiny` | ~7–10M | quick smoke test |
| `chiefui-30m` | ~27M | **recommended for 1660 Ti** |
| `chiefui-60m` | ~50M | larger, slower |

> Turing (1660 Ti) uses **fp16** (auto-selected; bf16 is emulated → disabled).

## Honest expectations

A from-scratch ~7–60M model **works** with small data but the output is **basic at
first** — it picks the right component category and produces valid HTML/CSS, but
design variety is limited to the seed templates. For high-quality, original
generation you need **thousands** of good examples + longer training. The goal of
this MVP is a working end-to-end pipeline and a clean, extensible architecture.
**Growing the dataset is the highest-impact step** (extend `build_seed.py` or add
your own `*.jsonl`).
