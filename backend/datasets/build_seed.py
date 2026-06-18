"""
backend/datasets/build_seed.py
==============================
DENGELI ve CESITLI tohum veri uretir. Her kategoride BOL ve FARKLI ifadelerle
ornek (orn. "Create a button", "Make a CTA button", "Generate a button
component"...) -> tiny model bile prompt'taki anahtar kelimeyi (button/card/
navbar...) cikti turuyle iliskilendirmeyi ogrenir.

Cikti formati: @@HTML@@ ... @@CSS@@ ... @@NOTES@@ ... @@END@@  (parse_response ile ayni)

    python backend/datasets/build_seed.py     # -> backend/datasets/seed.jsonl
"""
from __future__ import annotations
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed.jsonl")

# (tema, bg, fg, accent, renk-adi)
PALETTES = [
    ("dark", "#0f1117", "#e6e8ee", "#c45a26", "orange"),
    ("dark", "#0b132b", "#e0e6f0", "#3a86ff", "blue"),
    ("light", "#ffffff", "#1a1a1a", "#16a34a", "green"),
    ("dark", "#121212", "#f5f5f5", "#9b5de5", "purple"),
    ("light", "#f8fafc", "#0f172a", "#e11d48", "rose"),
    ("dark", "#1a1a2e", "#eaeaea", "#00b4d8", "cyan"),
]


def wrap(html, css, notes):
    # Kapanış "@@" YOK (tokenizasyon belirsizligini onlemek icin)
    return ("@@HTML\n" + html.strip() + "\n@@CSS\n" + css.strip()
            + "\n@@NOTES\n" + notes.strip() + "\n@@END")


def doc(title, body):
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"  <title>{title}</title>\n</head>\n<body>\n{body}\n</body>\n</html>"
    )


examples = []


def add(instructions, palettes, render):
    """her (instruction x palette) icin bir ornek uretir."""
    for instr in instructions:
        for theme, bg, fg, accent, cname in palettes:
            inp = f"Style: {theme}, {cname} accent {accent}, responsive."
            html, css, notes = render(theme, bg, fg, accent, cname)
            examples.append({"instruction": instr, "input": inp,
                             "output": wrap(html, css, notes)})


# ============================ BILESENLER (cogunluk) ==========================

# --- BUTON ------------------------------------------------------------------
BTN_INSTR = [
    "Create a button", "Create a Button", "Make a primary button",
    "Generate a button component", "I need a CTA button",
    "Design a call-to-action button", "Build a rounded button with hover effect",
]
def r_button(theme, bg, fg, accent, cname):
    html = "<button class=\"btn\">Get Started</button>"
    css = (f".btn {{\n  background:{accent};\n  color:#fff;\n  border:none;\n"
           f"  padding:12px 26px;\n  border-radius:10px;\n  font-size:1rem;\n"
           f"  font-weight:600;\n  cursor:pointer;\n  transition:transform .1s ease, filter .1s ease;\n}}\n"
           f".btn:hover {{ transform:translateY(-2px); filter:brightness(1.1); }}\n"
           f".btn:active {{ transform:translateY(0); }}")
    return html, css, f"Primary button using {accent}; lifts and brightens on hover."
add(BTN_INSTR, PALETTES, r_button)

# --- KART -------------------------------------------------------------------
CARD_INSTR = [
    "Create a card", "Make a pricing card", "Generate a card component",
    "Design a product card", "Build a feature card", "Create a premium pricing card",
]
def r_card(theme, bg, fg, accent, cname):
    card_bg = "#ffffff" if theme == "light" else "#1c1f2a"
    txt = "#1a1a1a" if theme == "light" else fg
    html = ("<div class=\"card\">\n  <h3>Premium</h3>\n"
            "  <p class=\"price\">$29<span>/mo</span></p>\n"
            "  <ul><li>Unlimited projects</li><li>Priority support</li><li>Custom domain</li></ul>\n"
            "  <a href=\"#\" class=\"card-cta\">Choose plan</a>\n</div>")
    css = (f".card {{\n  background:{card_bg};\n  color:{txt};\n  width:280px;\n"
           f"  border-radius:16px;\n  padding:28px;\n  font-family:system-ui,sans-serif;\n"
           f"  box-shadow:0 10px 30px rgba(0,0,0,.18);\n}}\n"
           f".card .price {{ font-size:2.4rem; font-weight:700; color:{accent}; }}\n"
           f".card .price span {{ font-size:1rem; opacity:.6; }}\n"
           f".card ul {{ list-style:none; padding:0; margin:18px 0; }}\n"
           f".card li {{ padding:6px 0; opacity:.85; }}\n"
           f".card-cta {{ display:block; text-align:center; background:{accent};"
           f" color:#fff; padding:12px; border-radius:10px; text-decoration:none; font-weight:600; }}")
    return html, css, f"Pricing card with {accent} price and CTA, soft elevation shadow."
add(CARD_INSTR, PALETTES, r_card)

# --- NAVBAR -----------------------------------------------------------------
NAV_INSTR = [
    "Create a navbar", "Make a navigation bar", "Generate a responsive navbar",
    "Design a top navigation", "Build a sticky header nav",
]
def r_navbar(theme, bg, fg, accent, cname):
    html = ("<header class=\"nav\">\n  <div class=\"logo\">Brand</div>\n"
            "  <nav>\n    <a href=\"#\">Home</a><a href=\"#\">Features</a>"
            "<a href=\"#\">Pricing</a><a href=\"#\" class=\"cta\">Sign up</a>\n  </nav>\n</header>")
    css = (f"* {{ box-sizing:border-box; }}\n"
           f".nav {{ display:flex; justify-content:space-between; align-items:center;\n"
           f"  padding:16px 6%; background:{bg}; font-family:system-ui,sans-serif; }}\n"
           f".logo {{ font-weight:700; font-size:1.3rem; color:{accent}; }}\n"
           f".nav nav a {{ color:{fg}; text-decoration:none; margin-left:22px; }}\n"
           f".nav .cta {{ background:{accent}; color:#fff; padding:8px 16px; border-radius:8px; }}\n"
           f"@media (max-width:600px) {{ .nav nav a {{ margin-left:12px; }} }}")
    return html, css, f"Responsive flex navbar with {accent} logo and CTA."
add(NAV_INSTR, PALETTES, r_navbar)

# --- FORM -------------------------------------------------------------------
FORM_INSTR = [
    "Create a login form", "Make a contact form", "Generate a signup form",
    "Design a newsletter form", "Build a simple login form",
]
def r_form(theme, bg, fg, accent, cname):
    fbg = "#ffffff" if theme == "light" else "#1c1f2a"
    txt = "#1a1a1a" if theme == "light" else fg
    html = ("<form class=\"form\">\n  <h2>Sign in</h2>\n"
            "  <input type=\"email\" placeholder=\"Email\">\n"
            "  <input type=\"password\" placeholder=\"Password\">\n"
            "  <button type=\"submit\">Log in</button>\n</form>")
    css = (f".form {{ display:flex; flex-direction:column; gap:12px; width:300px;\n"
           f"  background:{fbg}; color:{txt}; padding:28px; border-radius:14px;\n"
           f"  font-family:system-ui,sans-serif; box-shadow:0 10px 30px rgba(0,0,0,.18); }}\n"
           f".form input {{ padding:11px 14px; border:1px solid #8884; border-radius:8px;\n"
           f"  background:transparent; color:inherit; }}\n"
           f".form button {{ background:{accent}; color:#fff; border:none; padding:12px;\n"
           f"  border-radius:8px; font-weight:600; cursor:pointer; }}")
    return html, css, f"Login form card with {accent} submit button."
add(FORM_INSTR, PALETTES, r_form)

# --- ALERT ------------------------------------------------------------------
ALERT_INSTR = [
    "Create an alert box", "Make a success alert", "Generate a notification banner",
    "Design a warning alert",
]
def r_alert(theme, bg, fg, accent, cname):
    html = ("<div class=\"alert\">\n  <strong>Success!</strong> Your changes have been saved.\n"
            "  <button class=\"close\">&times;</button>\n</div>")
    css = (f".alert {{ display:flex; align-items:center; gap:10px; max-width:480px;\n"
           f"  background:{accent}22; border-left:4px solid {accent}; color:{fg};\n"
           f"  padding:14px 18px; border-radius:8px; font-family:system-ui,sans-serif; }}\n"
           f".alert .close {{ margin-left:auto; background:none; border:none; color:{fg};\n"
           f"  font-size:1.2rem; cursor:pointer; }}")
    return html, css, f"Inline alert with {accent} accent and dismiss button."
add(ALERT_INSTR, PALETTES, r_alert)

# --- BADGE ------------------------------------------------------------------
BADGE_INSTR = ["Create a badge", "Make a status badge", "Generate a label badge"]
def r_badge(theme, bg, fg, accent, cname):
    html = "<span class=\"badge\">New</span>"
    css = (f".badge {{ display:inline-block; background:{accent}; color:#fff;\n"
           f"  padding:4px 10px; border-radius:999px; font-size:.8rem; font-weight:600;\n"
           f"  font-family:system-ui,sans-serif; }}")
    return html, css, f"Pill badge in {accent}."
add(BADGE_INSTR, PALETTES, r_badge)

# --- HERO (landing, AMA artik baskin degil) ---------------------------------
HERO_INSTR = [
    "Create a hero section", "Make a landing page hero", "Design a SaaS hero section",
]
def r_hero(theme, bg, fg, accent, cname):
    body = ("  <section class=\"hero\">\n    <h1>Build faster, ship smarter.</h1>\n"
            "    <p>The all-in-one platform for modern teams.</p>\n"
            "    <a href=\"#\" class=\"btn\">Start free</a>\n  </section>")
    html = doc("Hero", body)
    css = (f"* {{ margin:0; box-sizing:border-box; font-family:system-ui,sans-serif; }}\n"
           f"body {{ background:{bg}; color:{fg}; }}\n"
           f".hero {{ max-width:720px; margin:10% auto; text-align:center; padding:0 6%; }}\n"
           f".hero h1 {{ font-size:3rem; line-height:1.1; margin-bottom:18px; }}\n"
           f".hero p {{ font-size:1.15rem; opacity:.8; margin-bottom:28px; }}\n"
           f".btn {{ display:inline-block; background:{accent}; color:#fff; padding:12px 26px;\n"
           f"  border-radius:10px; text-decoration:none; font-weight:600; }}\n"
           f"@media (max-width:600px) {{ .hero h1 {{ font-size:2rem; }} }}")
    return html, css, f"Centered hero, {theme} theme, {accent} CTA, responsive."
add(HERO_INSTR, PALETTES, r_hero)

# --- FULL SINGLE-FILE PAGE --------------------------------------------------
PAGE_INSTR = ["Create a single-file landing page", "Generate a full HTML landing page"]
def r_page(theme, bg, fg, accent, cname):
    body = ("  <header class=\"nav\"><div class=\"logo\">Nimbus</div>\n"
            "    <nav><a href=\"#\">Features</a><a href=\"#\" class=\"cta\">Get Started</a></nav></header>\n"
            "  <section class=\"hero\">\n    <h1>Design that scales with you.</h1>\n"
            "    <p>Premium tools for serious builders.</p>\n"
            "    <a href=\"#\" class=\"btn\">Try it free</a>\n  </section>")
    html = doc("Nimbus", body)
    css = (f"* {{ margin:0; box-sizing:border-box; font-family:system-ui,sans-serif; }}\n"
           f"body {{ background:{bg}; color:{fg}; }}\n"
           f".nav {{ display:flex; justify-content:space-between; align-items:center; padding:18px 6%; }}\n"
           f".logo {{ font-weight:700; color:{accent}; }}\n"
           f".nav a {{ color:{fg}; text-decoration:none; margin-left:20px; }}\n"
           f".nav .cta {{ background:{accent}; color:#fff; padding:8px 16px; border-radius:8px; }}\n"
           f".hero {{ max-width:720px; margin:8% auto; text-align:center; }}\n"
           f".hero h1 {{ font-size:2.8rem; margin-bottom:16px; }}\n"
           f".hero p {{ opacity:.8; margin-bottom:26px; }}\n"
           f".btn {{ background:{accent}; color:#fff; padding:12px 26px; border-radius:10px;\n"
           f"  text-decoration:none; font-weight:600; }}")
    return html, css, f"Single-file landing page with nav + hero, {accent} accent."
add(PAGE_INSTR, PALETTES[:4], r_page)

# ============================ DONUSUM KATEGORILERI ==========================

# --- Bozuk CSS -> duzeltilmis ----------------------------------------------
for bad, fixed, note in [
    (".box { width 100px; color: #zzz; margin: 10 }",
     ".box {\n  width: 100px;\n  color: #333;\n  margin: 10px;\n}",
     "width:, gecersiz renk #zzz->#333, ve px birimi eklendi."),
    ("a {color:red text-decoration:none}",
     "a {\n  color: red;\n  text-decoration: none;\n}",
     "Bildirimler arasina noktali virgul eklendi."),
    (".btn { padding: 10px; background #fff; border-radius: }",
     ".btn {\n  padding: 10px;\n  background: #fff;\n  border-radius: 6px;\n}",
     "background: eklendi, bos border-radius'a deger verildi."),
]:
    examples.append({"instruction": "Fix this broken CSS so it is valid.",
                     "input": f"Broken CSS:\n{bad}",
                     "output": wrap("<!-- CSS only -->", fixed, note)})

# --- Renk paleti -> CSS degiskenleri ---------------------------------------
for theme, bg, fg, accent, cname in PALETTES:
    css = (f":root {{\n  --bg: {bg};\n  --text: {fg};\n  --accent: {accent};\n"
           f"  --accent-soft: {accent}33;\n  --radius: 10px;\n}}\n"
           f"body {{ background:var(--bg); color:var(--text); }}\n"
           f".accent {{ color:var(--accent); }}")
    examples.append({"instruction": "Convert this color palette into reusable CSS variables.",
                     "input": f"Palette: background {bg}, text {fg}, accent {accent} ({cname}).",
                     "output": wrap("<!-- CSS variables only -->", css,
                                    ":root custom properties for the palette.")})

# --- Masaustu -> responsive -------------------------------------------------
for _ in range(4):
    css = (".grid {\n  display:grid;\n  grid-template-columns:repeat(3,1fr);\n  gap:24px;\n}\n"
           "@media (max-width:900px) {\n  .grid { grid-template-columns:repeat(2,1fr); }\n}\n"
           "@media (max-width:600px) {\n  .grid { grid-template-columns:1fr; }\n}")
    examples.append({"instruction": "Make this fixed 3-column desktop grid responsive.",
                     "input": "Desktop:\n.grid { display:grid; grid-template-columns:repeat(3,1fr); gap:24px; }",
                     "output": wrap("<!-- CSS only -->", css,
                                    "Breakpoints: 3 cols desktop, 2 tablet (<=900px), 1 mobile (<=600px).")})

# --- Bootstrap layout -------------------------------------------------------
for theme, bg, fg, accent, cname in PALETTES[:4]:
    html = ("<link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css\" rel=\"stylesheet\">\n"
            "<div class=\"container py-5\"><div class=\"row g-4\">\n"
            "  <div class=\"col-md-4\"><div class=\"card h-100\"><div class=\"card-body\">"
            "<h5 class=\"card-title\">Fast</h5><p class=\"card-text\">Lightning quick.</p></div></div></div>\n"
            "  <div class=\"col-md-4\"><div class=\"card h-100\"><div class=\"card-body\">"
            "<h5 class=\"card-title\">Secure</h5><p class=\"card-text\">Safe by default.</p></div></div></div>\n"
            "  <div class=\"col-md-4\"><div class=\"card h-100\"><div class=\"card-body\">"
            "<h5 class=\"card-title\">Simple</h5><p class=\"card-text\">Easy to use.</p></div></div></div>\n"
            "</div></div>")
    css = f".card-title {{ color:{accent}; }}"
    examples.append({"instruction": "Create a responsive 3-column Bootstrap feature card layout.",
                     "input": f"Style: Bootstrap 5, {cname} accent {accent}.",
                     "output": wrap(html, css, "Bootstrap 5 grid: 3 cols, stacks on mobile.")})


def main():
    # kategori dagilimini raporla
    from collections import Counter
    cats = Counter(e["instruction"].split()[0] for e in examples)
    with open(OUT, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[seed] {len(examples)} ornek yazildi -> {OUT}")
    print(f"[seed] ilk-kelime dagilimi: {dict(cats)}")
    print("[seed] NOT: gercek kalite icin binlerce ornekle buyut.")


if __name__ == "__main__":
    main()
