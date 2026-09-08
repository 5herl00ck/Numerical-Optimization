# Вычислительная оптимизация

Курс для магистрантов 1-го года. Осенний семестр 2026, занятия по средам с 9 сентября по 23 декабря.

За основу взята программа курса *Numerical Optimization* М. Диля (Systems Control and Optimization Laboratory, Университет Фрайбурга): <https://www.syscop.de/teaching/ss2025/numerical-optimization>.

## Материалы

- [Программа курса](syllabus.md) — цели, календарь, домашние задания, проект, оценивание, литература.
- Лекции:
  - [Лекция 1. Введение: постановка задачи, классы задач, примеры](lectures/lecture01/lecture01.md) · [демо-код](lectures/lecture01/demo01.py)
- Домашние задания:
  - [ДЗ 1](homeworks/hw01.md) — срок сдачи 23 сентября.

## Как запустить код

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 lectures/lecture01/demo01.py
```

Скрипты сохраняют графики в папку `figures/` рядом с собой.
