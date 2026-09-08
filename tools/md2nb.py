"""Собрать ноутбук-конспект из markdown-файла: lectureNN.md -> lectureNN.ipynb.

Каждый заголовок второго или третьего уровня (## / ###) начинает новую
markdown-ячейку; код не выполняется, в ноутбуке только текст и картинки
(пути вида img/... остаются относительными и работают в Jupyter).

Запуск:  uv run python tools/md2nb.py lectures/lecture01/lecture01.md
         (ноутбук пишется рядом с md; -o задаёт другой путь)
"""

import argparse
import json
import re
from pathlib import Path

HEADING = re.compile(r"^#{2,3} ")


def split_cells(text: str) -> list[str]:
    cells, current = [], []
    for line in text.splitlines():
        if HEADING.match(line) and current:
            cells.append("\n".join(current).rstrip())
            current = []
        current.append(line)
    if current:
        cells.append("\n".join(current).rstrip())
    return [c for c in cells if c.strip()]


def to_notebook(cells: list[str]) -> dict:
    def md_cell(src: str) -> dict:
        lines = src.splitlines(keepends=True)
        return {"cell_type": "markdown", "metadata": {}, "source": lines}

    return {
        "cells": [md_cell(c) for c in cells],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("md", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.md.with_suffix(".ipynb")
    nb = to_notebook(split_cells(args.md.read_text(encoding="utf-8")))
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{out}: {len(nb['cells'])} ячеек")


if __name__ == "__main__":
    main()
