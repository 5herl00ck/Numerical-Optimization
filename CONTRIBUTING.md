# Как сдать домашнее задание

Домашние задания сдаются через pull request в этот репозиторий — так все работы видны в одном месте и проверяются единообразно.

## Один раз в начале семестра

1. Нажмите **Fork** в правом верхнем углу [репозитория курса](https://github.com/IlyaChichkanov/Numerical-Optimization) — у вас появится собственная копия под вашим аккаунтом.
2. Склонируйте свой форк себе:

   ```bash
   git clone https://github.com/<ваш-логин>/Numerical-Optimization.git
   cd Numerical-Optimization
   git remote add upstream https://github.com/IlyaChichkanov/Numerical-Optimization.git
   ```

   `upstream` — это репозиторий курса; из него вы будете забирать новые лекции и задания, а `origin` (ваш форк) — то, куда вы пушите решения.

## На каждое домашнее задание

1. Обновите свою копию перед началом работы:

   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   git push origin main
   ```

2. Создайте ветку под конкретное задание — имя ветки значения не имеет, но так проще ориентироваться:

   ```bash
   git checkout -b hw01-<фамилия>
   ```

3. Скачайте условия (`homeworks/hw01.ipynb`), выполните и заполните ноутбук, перед сохранением — `Kernel → Restart Kernel and Run All Cells`.
4. Положите заполненный ноутбук в `submissions/hw01/<фамилия>.ipynb` (см. [`submissions/README.md`](submissions/README.md)) и закоммитьте:

   ```bash
   git add submissions/hw01/<фамилия>.ipynb
   git commit -m "hw01: <фамилия>"
   git push origin hw01-<фамилия>
   ```

5. На странице своего форка на GitHub нажмите **Compare & pull request**. Base repository — `IlyaChichkanov/Numerical-Optimization`, base — `main`; head — ваш форк и ветка `hw01-<фамилия>`. Название PR — `hw01: <фамилия>`.
6. Дождитесь комментария с оценкой прямо в PR. Если нужно что-то исправить — коммитьте в ту же ветку, PR обновится автоматически, повторно открывать не нужно.

Дедлайн — момент открытия PR (по времени коммита в GitHub), а не мержа: мержить будет преподаватель после проверки.

## Если раньше не пользовались git/GitHub

Ничего сложного, но лучше один раз потренироваться до первого дедлайна:

- [GitHub: Fork a repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
- [GitHub: About pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
- Ноутбук можно редактировать локально (`uv run jupyter lab`, см. [README](README.md#быстрый-старт)) или прямо в вашем форке через [GitHub Codespaces](https://github.com/features/codespaces) — тогда `git`-команды не нужны вовсе, коммит и push делаются кнопками в интерфейсе.

## Если не хотите публиковать код на GitHub

Напишите преподавателю — решение обсудим индивидуально (например, приватный форк).
