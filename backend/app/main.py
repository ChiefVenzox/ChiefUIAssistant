"""
backend/app/main.py
===================
ChiefUI Assistant FastAPI uygulaması.

    cd backend
    uvicorn app.main:app --reload --port 8000
veya:
    python -m app.main            (basit çalıştırma)

Ortam değişkenleri (opsiyonel):
    CHIEFUI_CKPT      (vars: checkpoints/ckpt.pt)
    CHIEFUI_TOKENIZER (vars: checkpoints/tokenizer.json)
    CHIEFUI_DEVICE    (vars: cuda varsa cuda, yoksa cpu)
    CHIEFUI_DB        (vars: database/history.db)
"""
from __future__ import annotations
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.database import db
from app import runtime

app = FastAPI(title="ChiefUI Assistant")

# Vite dev sunucusu farklı portta -> CORS izinleri
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def _startup():
    db.init_db(os.environ.get("CHIEFUI_DB", "database/history.db"))
    runtime.load(
        ckpt_path=os.environ.get("CHIEFUI_CKPT", "checkpoints/ckpt.pt"),
        tokenizer_path=os.environ.get("CHIEFUI_TOKENIZER", "checkpoints/tokenizer.json"),
        device=os.environ.get("CHIEFUI_DEVICE",
                              "cuda" if _cuda() else "cpu"),
    )


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
