"""
backend/app/inference/prompt_format.py
=======================================
Model giriş/çıkış formatı. Hem eğitim (dataset) hem üretim (API) aynı formatı
kullanır.

Giriş tokenları:   [<|user|>] "Instruction: ...\nStyle: ..." [<|assistant|>]
Çıkış (metin):     <response><html>..</html><css>..</css><notes>..</notes></response>
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional


def build_user_text(instruction: str, input_text: Optional[str]) -> str:
    parts = [f"Instruction: {(instruction or '').strip()}"]
    if input_text and input_text.strip():
        parts.append(f"Style: {input_text.strip()}")
    return "\n".join(parts)


def build_input_ids(tok, instruction: str, input_text: Optional[str]) -> List[int]:
    """Üretim için giriş token id'leri ( <|assistant|> ile biter )."""
    ids = [tok.user_id]
    ids += tok.encode(build_user_text(instruction, input_text))
    ids.append(tok.assistant_id)
    return ids


# Bölüm işaretçileri: HTML/CSS ile çakışmaz VE kapanış "@@" yok.
# (Kapanışlı "@@HTML@@" tokenizasyonda "@@|HTML|@@" oluyordu; paylaşılan "@@"
#  hem açılış hem kapanış olunca model "@@" sonrası en sık geleni (\n) seçip
#  "HTML@@"yi atlıyordu. Kapanış "@@" kaldırılınca "@@" daima bir bölüm adıyla
#  devam eder -> belirsizlik biter.)
SECTIONS = {"html": "@@HTML", "css": "@@CSS", "notes": "@@NOTES"}
END_MARK = "@@END"
_ALL_MARKS = list(SECTIONS.values()) + [END_MARK]


def build_response_block(html: str, css: str, notes: str) -> str:
    return (
        f"@@HTML\n{(html or '').strip()}\n"
        f"@@CSS\n{(css or '').strip()}\n"
        f"@@NOTES\n{(notes or '').strip()}\n"
        f"@@END"
    )


def example_output_text(example: Dict) -> str:
    """Dataset örneğinden hedef metni üretir.
    - 'output' varsa olduğu gibi kullanılır (tercihen <response> bloğu).
    - yoksa html/css/notes alanlarından <response> bloğu kurulur."""
    out = example.get("output")
    if out:
        return out.strip()
    return build_response_block(
        example.get("html", ""), example.get("css", ""), example.get("notes", "")
    )


def build_training_ids(tok, example: Dict) -> List[int]:
    ids = build_input_ids(tok, example.get("instruction", ""), example.get("input", ""))
    ids += tok.encode(example_output_text(example))
    ids.append(tok.end_id)
    return ids


def _section(text: str, name: str) -> str:
    """@@NAME@@ ... (bir sonraki işaretçi veya metin sonu) arasını alır.
    Kesilmiş (truncated) çıktıya dayanıklı: kapanış işaretçisi olmasa da çalışır."""
    marker = SECTIONS[name]
    i = text.find(marker)
    if i < 0:
        return ""
    start = i + len(marker)
    ends = [text.find(m, start) for m in _ALL_MARKS]
    ends = [e for e in ends if e >= 0]
    end = min(ends) if ends else len(text)
    return text[start:end].strip()


def parse_response(text: str) -> Dict[str, str]:
    """Model çıktısından html/css/notes ayrıştırır (işaretçi-tabanlı, kesilmeye dayanıklı)."""
    html = _section(text, "html")
    css = _section(text, "css")
    notes = _section(text, "notes")

    # Hiç bölüm yoksa: işaretçileri temizleyip tüm metni HTML say (fallback)
    if not html and not css:
        cleaned = text
        for m in _ALL_MARKS:
            cleaned = cleaned.replace(m, "")
        html = cleaned.strip()

    return {"html": html, "css": css, "notes": notes, "raw": text}
