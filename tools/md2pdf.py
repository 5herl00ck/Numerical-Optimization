"""Переводит markdown-конспект в PDF: markdown -> HTML (формулы через KaTeX) -> Chrome headless.

    uv run python tools/md2pdf.py lectures/lecture01/lecture01.md            # -> lecture01.pdf рядом
    uv run python tools/md2pdf.py lectures/lecture01/*.md --out build/        # все в папку
    uv run python tools/md2pdf.py lecture01.md --html                         # заодно оставить .html

Что нужно: Google Chrome / Chromium в PATH (или переменная CHROME=/путь/к/chrome) и
доступ в интернет при сборке — KaTeX подгружается с CDN. Картинки берутся по
относительным путям из папки md-файла, свёрнутые блоки <details> печатаются раскрытыми.
"""

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
html { font-size: 11pt; }
body { font-family: "Noto Sans", "DejaVu Sans", "Liberation Sans", Arial, sans-serif;
       line-height: 1.45; color: #111; max-width: 100%; margin: 0; }
h1 { font-size: 1.7em; margin: 0 0 0.4em; line-height: 1.2; }
h2 { font-size: 1.35em; margin: 1.4em 0 0.5em; border-bottom: 1px solid #bbb; padding-bottom: 0.15em;
     page-break-after: avoid; }
h3 { font-size: 1.12em; margin: 1.2em 0 0.4em; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
img { max-width: 100%; height: auto; display: block; margin: 0.6em auto; page-break-inside: avoid; }
table { border-collapse: collapse; margin: 0.8em 0; font-size: 0.95em; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 0.3em 0.6em; vertical-align: top; text-align: left; }
th { background: #f0f0f0; }
thead tr:not(:has(th:not(:empty))) { display: none; }   /* таблицы без заголовка (| | |) */
td img { margin: 0.2em auto; }
code { font-family: "DejaVu Sans Mono", "Liberation Mono", monospace; font-size: 0.9em;
       background: #f4f4f4; padding: 0 0.2em; border-radius: 2px; }
pre { background: #f4f4f4; padding: 0.6em 0.8em; overflow-x: auto; font-size: 0.85em;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #bbb; margin: 0.8em 0; padding: 0.1em 1em; color: #333; }
details { border: 1px solid #ccc; border-radius: 4px; padding: 0.4em 0.9em; margin: 0.8em 0; }
summary { font-weight: 600; cursor: default; }
hr { border: 0; border-top: 1px solid #bbb; margin: 1.4em 0; }
.katex-display { margin: 0.7em 0; overflow-x: auto; overflow-y: hidden; }
.katex { font-size: 1.05em; }
a { color: #1a4d9c; text-decoration: none; }
"""

TEMPLATE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>{title}</title>
<base href="{base}">
<link rel="stylesheet" href="{katex}/katex.min.css">
<script defer src="{katex}/katex.min.js"></script>
<script defer src="{katex}/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [
    {{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}],
    throwOnError: false}});"></script>
<style>{css}</style>
</head><body>
{body}
</body></html>
"""

# Код (fenced и инлайн) не трогаем, формулы внутри него — не формулы.
CODE_RE = re.compile(r"(```.*?```|`[^`\n]*`)", re.S)
DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE_RE = re.compile(r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)")


def protect_math(text: str) -> tuple[str, list[str]]:
    """Заменяет формулы на плейсхолдеры, чтобы markdown не портил _ и * внутри них."""
    stash: list[str] = []

    def keep(m: re.Match, display: bool) -> str:
        stash.append(("$$%s$$" if display else "$%s$") % m.group(1))
        return f"MATHSTASH{len(stash) - 1}END"

    def process(segment: str) -> str:
        segment = DISPLAY_RE.sub(lambda m: keep(m, True), segment)
        return INLINE_RE.sub(lambda m: keep(m, False), segment)

    parts = CODE_RE.split(text)
    out = [p if i % 2 else process(p) for i, p in enumerate(parts)]
    return "".join(out), stash


def restore_math(body: str, stash: list[str]) -> str:
    # Экранируем <, >, & — браузер вернёт их в текст, и auto-render увидит исходный TeX.
    return re.sub(r"MATHSTASH(\d+)END", lambda m: html.escape(stash[int(m.group(1))], quote=False), body)


def md_to_html(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    text = text.replace("<details>", "<details open>")
    text, stash = protect_math(text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])
    body = restore_math(body, stash)
    m = re.search(r"^#\s+(.+)$", md_path.read_text(encoding="utf-8"), re.M)
    title = html.escape(m.group(1)) if m else md_path.stem
    return TEMPLATE.format(title=title, base=md_path.resolve().parent.as_uri() + "/",
                           katex=KATEX, css=CSS, body=body)


def find_chrome() -> str:
    for cand in (os.environ.get("CHROME"), "google-chrome", "google-chrome-stable", "chromium",
                 "chromium-browser", "chrome"):
        if cand and shutil.which(cand):
            return shutil.which(cand)
    sys.exit("Не найден Chrome/Chromium. Установите его или укажите путь: CHROME=/path/to/chrome")


def html_to_pdf(html_path: Path, pdf_path: Path, chrome: str) -> None:
    # Отдельный временный профиль: иначе headless Chrome может повиснуть на блокировке
    # профиля, если у пользователя уже открыт обычный Chrome.
    with tempfile.TemporaryDirectory(prefix="md2pdf-") as profile:
        cmd = [chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
               f"--user-data-dir={profile}", "--no-first-run", "--disable-extensions",
               "--virtual-time-budget=20000",            # даём KaTeX дорисовать формулы
               f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            sys.exit(f"Chrome не ответил за 180 с: {html_path}")
    if not pdf_path.exists():
        sys.exit(f"Chrome не создал PDF:\n{res.stderr[-2000:]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="markdown-файлы")
    ap.add_argument("--out", type=Path, help="папка для PDF (по умолчанию — рядом с md)")
    ap.add_argument("--html", action="store_true", help="сохранить и промежуточный .html")
    args = ap.parse_args()

    chrome = find_chrome()
    for md_path in args.files:
        out_dir = args.out or md_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / md_path.with_suffix(".pdf").name
        page = md_to_html(md_path)
        if args.html:
            html_path = out_dir / md_path.with_suffix(".html").name
            html_path.write_text(page, encoding="utf-8")
            html_to_pdf(html_path, pdf_path, chrome)
        else:
            with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp:
                tmp.write(page)
            try:
                html_to_pdf(Path(tmp.name), pdf_path, chrome)
            finally:
                os.unlink(tmp.name)
        print(f"{md_path} -> {pdf_path} ({pdf_path.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    main()
