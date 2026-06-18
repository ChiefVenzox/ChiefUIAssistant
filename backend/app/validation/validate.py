"""
backend/app/validation/validate.py
===================================
Üretilen UI kodu için basit, bağımlılıksız doğrulama (MVP). İleride stylelint /
html5lib eklenebilir.

validate(html, css) -> {"ok": bool, "issues": [...], "warnings": [...]}
  issues   = ciddi sorunlar (ok=False yapar)
  warnings = öneriler (ok'u etkilemez)
"""
from __future__ import annotations
import re
from typing import Dict


def _balanced(text: str, open_ch: str, close_ch: str) -> bool:
    return text.count(open_ch) == text.count(close_ch)


def validate(html: str, css: str) -> Dict:
    html = html or ""
    css = css or ""
    issues, warnings = [], []

    # 1) boş mu
    if not html.strip() and not css.strip():
        issues.append("Üretilen kod boş.")
        return {"ok": False, "issues": issues, "warnings": warnings}

    low = html.lower()

    # 2) temel HTML etiketleri
    if "<" not in html or ">" not in html:
        issues.append("HTML etiketi bulunamadı.")
    if "<!doctype" not in low:
        warnings.append("<!DOCTYPE html> eksik.")
    if "<body" not in low and "<div" not in low and "<section" not in low:
        warnings.append("Gövde/yapısal etiket (body/div/section) görünmüyor.")

    # 3) viewport meta
    if "viewport" not in low:
        warnings.append("Responsive için viewport meta etiketi eksik "
                        "(<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">).")

    # 4) HTML açı parantez dengesi (kaba)
    if html.count("<") != html.count(">"):
        warnings.append("HTML'de < ve > sayısı eşleşmiyor (eksik/bozuk etiket olabilir).")

    # 5) CSS süslü parantez dengesi
    if css.strip():
        if not _balanced(css, "{", "}"):
            issues.append("CSS süslü parantezleri dengesiz ({ } eşleşmiyor).")
        if not _balanced(css, "(", ")"):
            warnings.append("CSS'te ( ) dengesiz olabilir.")
    # inline <style> içindeki CSS de kontrol edilsin
    for style in re.findall(r"<style[^>]*>(.*?)</style>", html, re.S | re.I):
        if not _balanced(style, "{", "}"):
            issues.append("Inline <style> içinde { } dengesiz.")

    return {"ok": len(issues) == 0, "issues": issues, "warnings": warnings}
