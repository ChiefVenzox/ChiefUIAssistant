"""
backend/app/api/routes.py
=========================
HTTP + WebSocket uçları:
  POST /api/generate-ui   tek seferde UI üret (JSON sonuç)
  WS   /ws/generate       token token (streaming) UI üret
  GET  /api/history       son üretimler
  GET  /api/health        durum
"""
from __future__ import annotations
import asyncio
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.runtime import STATE
from app.inference.generate import stream_ui
from app.inference.prompt_format import parse_response
from app.validation.validate import validate
from app.database import db

router = APIRouter()


class GenerateRequest(BaseModel):
    instruction: str
    input: Optional[str] = ""
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95


def _gen_kwargs(req: GenerateRequest) -> dict:
    return dict(max_new_tokens=req.max_new_tokens, temperature=req.temperature,
                top_k=req.top_k, top_p=req.top_p)


def _settings(req: GenerateRequest) -> dict:
    return {"max_new_tokens": req.max_new_tokens, "temperature": req.temperature,
            "top_k": req.top_k, "top_p": req.top_p}


@router.get("/api/health")
def health():
    ready = STATE.get("ready", False)
    return {
        "ready": ready,
        "device": STATE.get("device"),
        "params_m": (STATE["model"].num_params() / 1e6) if ready else None,
    }


@router.get("/api/history")
def history(limit: int = 20):
    return {"items": db.get_history(limit)}


@router.post("/api/generate-ui")
def generate_ui_endpoint(req: GenerateRequest):
    if not STATE.get("ready"):
        raise HTTPException(503, "Model yüklenmedi. Önce eğitip checkpoint oluştur.")
    full = "".join(stream_ui(STATE["model"], STATE["tok"], req.instruction,
                             req.input, STATE["device"], **_gen_kwargs(req)))
    result = parse_response(full)
    val = validate(result["html"], result["css"])
    rid = db.save_generation(req.instruction, req.input, result, val,
                             _settings(req), db.now())
    return {"id": rid, **result, "validation": val}


@router.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    await ws.accept()
    try:
        req_raw = await ws.receive_json()
    except Exception:
        await ws.close()
        return

    if not STATE.get("ready"):
        await ws.send_json({"type": "error", "message": "Model yüklenmedi."})
        await ws.close()
        return

    req = GenerateRequest(**req_raw)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def worker():
        try:
            for piece in stream_ui(STATE["model"], STATE["tok"], req.instruction,
                                   req.input, STATE["device"], **_gen_kwargs(req)):
                loop.call_soon_threadsafe(queue.put_nowait, ("token", piece))
        except Exception as e:  # üretim hatasını istemciye ilet
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, (SENTINEL, None))

    threading.Thread(target=worker, daemon=True).start()

    pieces = []
    errored = False
    try:
        while True:
            kind, val = await queue.get()
            if kind is SENTINEL:
                break
            if kind == "token":
                pieces.append(val)
                await ws.send_json({"type": "token", "text": val})
            elif kind == "error":
                errored = True
                await ws.send_json({"type": "error", "message": val})
    except WebSocketDisconnect:
        return

    # Hata olduysa: tek terminal mesaj (error) gönderildi; done/parse/DB atla.
    if errored:
        try:
            await ws.close()
        except Exception:
            pass
        return

    full = "".join(pieces)
    result = parse_response(full)
    validation = validate(result["html"], result["css"])
    rid = db.save_generation(req.instruction, req.input, result, validation,
                             _settings(req), db.now())
    try:
        await ws.send_json({"type": "done", "id": rid, "validation": validation, **result})
        await ws.close()
    except Exception:
        pass
