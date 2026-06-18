"""
backend/app/database/db.py
==========================
MVP için SQLite geçmişi. Her üretim (prompt + sonuç) kaydedilir.
Basitlik için her işlemde yeni bağlantı açılır (FastAPI threadpool ile güvenli).
"""
from __future__ import annotations
import json
import os
import sqlite3
import time
from typing import Dict, List, Optional

DEFAULT_DB = "backend/database/history.db"
_DB_PATH = DEFAULT_DB


def init_db(path: str = DEFAULT_DB):
    global _DB_PATH
    _DB_PATH = path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  REAL NOT NULL,
            instruction TEXT,
            input       TEXT,
            html        TEXT,
            css         TEXT,
            notes       TEXT,
            raw         TEXT,
            validation  TEXT,
            settings    TEXT
        )
    """)
    con.commit()
    con.close()


def save_generation(instruction: str, input_text: Optional[str], result: Dict,
                    validation: Dict, settings: Dict, created_at: float) -> int:
    con = sqlite3.connect(_DB_PATH)
    cur = con.execute(
        """INSERT INTO generations
           (created_at, instruction, input, html, css, notes, raw, validation, settings)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (created_at, instruction, input_text or "", result.get("html", ""),
         result.get("css", ""), result.get("notes", ""), result.get("raw", ""),
         json.dumps(validation, ensure_ascii=False), json.dumps(settings, ensure_ascii=False)),
    )
    con.commit()
    rid = cur.lastrowid
    con.close()
    return rid


def get_history(limit: int = 20) -> List[Dict]:
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, created_at, instruction, input, html, css, notes "
        "FROM generations ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def now() -> float:
    return time.time()
