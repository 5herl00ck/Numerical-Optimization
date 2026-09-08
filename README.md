# Вычислительная оптимизация

Курс для магистрантов 1-го года. Осенний семестр 2026, занятия по средам с 9 сентября по 23 декабря.

За основу взята программа курса *Numerical Optimization* М. Диля (Systems Control and Optimization Laboratory, Университет Фрайбурга): <https://www.syscop.de/teaching/ss2025/numerical-optimization>.

## Материалы

- [Программа курса](syllabus.md) — цели, календарь, домашние задания, проект, оценивание, литература.
- Лекции:
  - Лекция 1. Введение: постановка задачи, классы задач, примеры —
    [конспект](lectures/lecture01/lecture01.md) · [конспект-ноутбук](lectures/lecture01/lecture01.ipynb) ·
    [демо-скрипт](lectures/lecture01/demo01.py) · [демо-ноутбук](lectures/lecture01/demo01.ipynb) ·
    [сценарий доски](lectures/lecture01/board01.md) · [разбор упражнений](lectures/lecture01/exercises01.md)
- Домашние задания:
  - [ДЗ 1](homeworks/hw01.md) — ноутбук [`hw01.ipynb`](homeworks/hw01.ipynb), срок сдачи 23 сентября.
  - Решения — в [`homeworks/solutions/`](homeworks/solutions/), публикуются после дедлайна.

Конспект каждой лекции есть в двух форматах: `.md` (читать на GitHub) и `.ipynb` (тот же
текст ячейками, удобно дополнять своими расчётами). Все картинки лекции лежат в одной папке `img/`;
иллюстрации конспекта строятся скриптом `make_figures.py`. Демо — тоже в двух форматах: ноутбук — пошаговый
урок с пояснениями по `scipy.optimize` и стандартной формой каждого примера (собирается
скриптом `build_demoNN.py`), скрипт — те же примеры компактно, графики сохраняет туда же в `img/` (файлы `demo_*.png`, не коммитятся). Сценарий
доски — раскадровка для преподавателя: что нарисовать живьём, что вставить заранее.

## Как запустить код

Основной способ — [uv](https://docs.astral.sh/uv/) (окружение и зависимости описаны в
`pyproject.toml` и зафиксированы в `uv.lock`):

```bash
uv sync                                        # создаёт .venv и ставит зависимости
uv run python lectures/lecture01/demo01.py     # демо-скрипт
uv run jupyter lab                             # ноутбуки лекций и демо
```

Отдельно активировать окружение не нужно: `uv run` делает это сам. Если всё же хочется —
`source .venv/bin/activate`.

С лекции 7 понадобится CasADi (алгоритмическое дифференцирование и интерфейс к IPOPT):

```bash
uv sync --group casadi
```

PDF из любого markdown-файла (конспект, разбор, ДЗ) собирает `tools/md2pdf.py` — нужен
установленный Chrome/Chromium и интернет при сборке (формулы рендерит KaTeX с CDN):

```bash
uv run python tools/md2pdf.py lectures/lecture01/lecture01.md              # lecture01.pdf рядом с md
uv run python tools/md2pdf.py lectures/lecture01/*.md --out build/pdf      # все md лекции в одну папку
```

Собранные PDF в git не попадают (`.gitignore`); чтобы выложить — `git add -f`.

Без uv, обычным `pip`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 lectures/lecture01/demo01.py
```

## Чтение конспекта в VSCode

Откройте **папку репозитория целиком** (`code .`, не отдельный файл), установите расширение
[Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one),
затем в любом `.md`-файле нажмите **`Ctrl+Shift+V`** (или иконку превью в верхнем правом углу вкладки) —
запустится встроенный КаTeX-рендер VSCode и будут видны формулы, таблицы, картинки из папки `img/`.
Рабочие настройки уже лежат в `.vscode/settings.json`.
