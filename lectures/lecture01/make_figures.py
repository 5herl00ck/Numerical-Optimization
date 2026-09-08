"""Иллюстрации к конспекту лекции 1 -> папка img/ (коммитится в репозиторий).

Запуск:  uv run python lectures/lecture01/make_figures.py

Скрипт самодостаточен и не импортирует demo01.py. Картинки 07-10 повторяют
демонстрации (a)-(d) с теми же данными и seed, чтобы конспект и демо совпадали.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import PchipInterpolator
from scipy.optimize import LinearConstraint, linprog, minimize

IMG = Path(__file__).resolve().parent / "img"
IMG.mkdir(exist_ok=True)

# Палитра: категориальные цвета в фиксированном порядке; красный — только для решения.
BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7")
GRAY, INK = "#8a8985", "#0b0b0b"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.6,
    "lines.linewidth": 2, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 150,
})


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(IMG / name, bbox_inches="tight")
    plt.close()
    print("  ", name)


def note(ax, text, xy, xytext, **kw):
    """Подпись со стрелкой."""
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1), **kw)


# ---------------------------------------------------------------- 01 допустимое множество
def fig_feasible_set():
    # h1 = x1 >= 0,  h2 = 3 - x1 - x2 >= 0,  g = x2 - 0.5 x1^2 = 0,  f = |x - (2,2)|^2
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    xs = np.linspace(-0.8, 3.6, 400)
    # область неравенств
    ax.fill_between(xs, -0.8, 3 - xs, where=xs >= 0, color=BLUE, alpha=0.12, lw=0)
    ax.plot([0, 0], [-0.8, 3.6], color=BLUE, lw=1.5)
    ax.plot(xs, 3 - xs, color=BLUE, lw=1.5)
    # линии уровня f
    th = np.linspace(0, 2 * np.pi, 200)
    for r in (0.5, 1.0, 1.5, 2.0, 2.5):
        ax.plot(2 + r * np.cos(th), 2 + r * np.sin(th), color=GRAY, lw=0.8, ls="--")
    # равенство g = 0: парабола; допустимый кусок — там, где выполнены неравенства
    xp = np.linspace(-0.8, 3.0, 400)
    ax.plot(xp, 0.5 * xp**2, color=ORANGE, lw=1.5, ls=":")
    x_max = -1 + np.sqrt(7)          # x + x^2/2 = 3
    xo = np.linspace(0, x_max, 200)
    ax.plot(xo, 0.5 * xo**2, color=ORANGE, lw=3.5, solid_capstyle="round")
    xstar = np.array([x_max, 0.5 * x_max**2])
    ax.plot(*xstar, marker="*", ms=16, color=RED, mec="white", mew=0.8, ls="none", zorder=5)
    ax.plot(2, 2, "+", color=GRAY, ms=8, mew=1.2)

    ax.text(0.15, -0.55, r"$h_1(x)=x_1\geq 0$", color=BLUE, fontsize=9)
    ax.text(2.35, 0.25, r"$h_2(x)=3-x_1-x_2\geq 0$", color=BLUE, fontsize=9, rotation=-45)
    ax.text(-0.75, 1.5, r"$h(x)\geq 0$", color=BLUE, fontsize=10, alpha=0.9)
    ax.text(2.55, 3.2, r"$g(x)=0$", color=ORANGE, fontsize=10)
    note(ax, r"$\Omega$ — этот кусок кривой", xy=(0.9, 0.4), xytext=(-0.6, 2.9))
    note(ax, r"$x^\ast$: $h_2$ активно, $h_1$ нет", xy=xstar, xytext=(0.9, 3.35))
    ax.text(2.05, 3.05, "линии уровня $f$", color=GRAY, fontsize=8, ha="center")
    ax.set_xlim(-0.8, 3.6)
    ax.set_ylim(-0.8, 3.6)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Допустимое множество и активные ограничения")
    save("01_feasible_set.png")


# ---------------------------------------------------------------- 02 виды минимумов
def fig_minima_1d():
    kx = [0, 1, 1.5, 2, 3, 4, 4.5, 5, 5.5, 6, 7.2, 8, 9, 10]
    ky = [3.2, 1.7, 1.5, 1.7, 2.6, 2.2, 2.2, 2.2, 2.2, 2.6, 0.5, 0.8, 2.4, 3.4]
    f = PchipInterpolator(kx, ky)
    xs = np.linspace(0, 10, 600)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(xs, f(xs), color=BLUE)
    ax.plot(1.5, 1.5, "o", color=ORANGE, ms=8, zorder=5)
    ax.plot(xs[(xs >= 4) & (xs <= 5.5)], f(xs[(xs >= 4) & (xs <= 5.5)]), color=ORANGE, lw=4)
    ax.plot(7.2, 0.5, marker="*", ms=16, color=RED, mec="white", mew=0.8, ls="none", zorder=5)
    note(ax, "строгий локальный минимум", xy=(1.5, 1.5), xytext=(0.3, 0.35))
    note(ax, "нестрогий локальный минимум\n(плато: минимумы — все точки отрезка)",
         xy=(4.75, 2.2), xytext=(3.2, 3.3))
    note(ax, "глобальный минимум\n(он же строгий локальный)", xy=(7.2, 0.5), xytext=(7.6, 0.25))
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    save("02_minima_1d.png")


# ---------------------------------------------------------------- 03 патологии
def fig_pathologies():
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.0))
    (a, b), (c, d) = axs

    # (a) несовместность на числовой прямой
    a.axhline(0, color=INK, lw=1)
    a.annotate("", xy=(2.6, 0), xytext=(1, 0),
               arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=3))
    a.annotate("", xy=(-1.6, 0), xytext=(0, 0),
               arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=3))
    a.plot([1], [0], "o", color=BLUE, ms=7)
    a.plot([0], [0], "o", color=ORANGE, ms=7)
    a.text(1.7, 0.25, r"$x\geq 1$", color=BLUE, ha="center")
    a.text(-0.8, 0.25, r"$x\leq 0$", color=ORANGE, ha="center")
    a.text(0.5, -0.45, r"$\Omega=\varnothing$ — задача несовместна", ha="center")
    a.set_xlim(-1.8, 2.8)
    a.set_ylim(-0.8, 0.8)
    a.axis("off")
    a.set_title(r"$\min x$ при $x\geq 1,\ x\leq 0$")

    # (b) не ограничена снизу
    xs = np.linspace(-1.8, 1.8, 200)
    b.plot(xs, xs**3, color=BLUE)
    b.annotate("", xy=(-1.75, -5.6), xytext=(-1.2, -1.8),
               arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5))
    b.text(-1.1, -4.6, r"$\inf f=-\infty$", color=RED)
    b.set_title(r"$\min x^3$ на $\mathbb{R}$")
    b.set_xticks([])
    b.set_yticks([])

    # (c) инфимум не достигается
    xs = np.linspace(-4, 1.2, 200)
    c.plot(xs, np.exp(xs), color=BLUE)
    c.axhline(0, color=RED, lw=1, ls="--")
    c.text(-3.9, 0.25, r"$\inf f = 0$, но $e^x>0$ всюду", color=RED)
    c.set_ylim(-0.4, 3.3)
    c.set_title(r"$\min e^x$ на $\mathbb{R}$")
    c.set_xticks([])
    c.set_yticks([])

    # (d) много минимумов
    xs = np.linspace(-1.7, 1.7, 300)
    d.plot(xs, (xs**2 - 1) ** 2, color=BLUE)
    d.plot([-1, 1], [0, 0], marker="*", ms=14, color=RED, mec="white", mew=0.8, ls="none")
    d.text(0, 0.9, r"два глобальных минимума: $x=\pm 1$", ha="center", color=RED)
    d.set_ylim(-0.3, 2.2)
    d.set_title(r"$\min (x^2-1)^2$")
    d.set_xticks([-1, 0, 1])
    d.set_yticks([])
    save("03_pathologies.png")


# ---------------------------------------------------------------- 04 коэрцитивность
def fig_coercive():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    xs = np.linspace(-3.5, 3.5, 400)
    f = 0.5 * xs**2 + np.sin(2 * xs)
    x0 = 2.6
    lvl = 0.5 * x0**2 + np.sin(2 * x0)
    a.plot(xs, f, color=BLUE)
    a.axhline(lvl, color=GRAY, lw=1, ls="--")
    a.plot(x0, lvl, "o", color=ORANGE, ms=7)
    a.text(x0 - 0.1, lvl + 0.5, r"$f(x_0)$", color=ORANGE, ha="right")
    below = xs[f <= lvl]
    a.plot(below, np.full_like(below, -1.9), color=ORANGE, lw=5, solid_capstyle="butt")
    a.text(0, -2.9, r"$\{x: f(x)\leq f(x_0)\}$ — ограничено", ha="center", color=ORANGE, fontsize=9)
    a.set_ylim(-3.4, 7)
    a.set_title(r"$f\to+\infty$ при $|x|\to\infty$: коэрцитивна")
    a.set_xticks([])
    a.set_yticks([])

    xs = np.linspace(-4, 1.5, 400)
    g = np.exp(xs)
    x0 = 0.8
    lvl = np.exp(x0)
    b.plot(xs, g, color=BLUE)
    b.axhline(lvl, color=GRAY, lw=1, ls="--")
    b.plot(x0, lvl, "o", color=ORANGE, ms=7)
    b.text(x0 - 0.15, lvl + 0.35, r"$f(x_0)$", color=ORANGE, ha="right")
    below = xs[g <= lvl]
    b.plot(below, np.full_like(below, -0.6), color=ORANGE, lw=5, solid_capstyle="butt")
    b.annotate("", xy=(-4.2, -0.6), xytext=(-3.2, -0.6),
               arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2))
    b.text(-1.5, -1.25, "не ограничено — минимума нет", ha="center", color=ORANGE, fontsize=9)
    b.set_ylim(-1.6, 4.5)
    b.set_title(r"$e^x$: не коэрцитивна")
    b.set_xticks([])
    b.set_yticks([])
    save("04_coercive.png")


# ---------------------------------------------------------------- 05 классы задач
def fig_classes():
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    boxes = [  # (label, x, y, w, h, color)
        ("NLP", 0.0, 0.0, 10.0, 6.0, GRAY),
        ("QCQP", 0.3, 0.3, 7.2, 4.5, VIOLET),
        ("QP", 0.6, 0.6, 5.2, 3.2, AQUA),
        ("LP", 0.9, 0.9, 2.4, 1.6, BLUE),
    ]
    for label, x, y, w, h, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc=col, ec=col, alpha=0.10, lw=0))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc="none", ec=col, lw=1.6))
        ax.text(x + w - 0.15, y + h - 0.15, label, ha="right", va="top", color=col,
                fontsize=12, fontweight="bold")
    # водораздел: всё LP выпукло, у остальных классов есть обе половины
    ax.plot([3.8, 3.8], [0.0, 6.0], color=RED, lw=2, ls="--")
    ax.text(3.7, 6.15, "выпуклые", ha="right", color=RED, fontsize=10)
    ax.text(3.9, 6.15, "невыпуклые", ha="left", color=RED, fontsize=10)
    ax.text(3.8, -0.35, "«великий водораздел»", ha="center", color=RED, fontsize=9)

    def sticker(text, x, y):
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc=YELLOW, ec="none", alpha=0.55))

    sticker("диета, план\nпроизводства", 2.1, 1.7)
    sticker("МНК (полином)", 2.0, 3.2)
    sticker("цепь с полом", 3.0, 2.85)
    sticker("QP с индефинитной $Q$", 5.0, 2.0)
    sticker("точка окружности,\nближайшая к $(3,4)$", 5.9, 3.9)
    sticker("$\\|x-a\\|_2\\leq 1$", 2.2, 4.2)
    sticker("Химмельблау", 8.6, 1.0)
    sticker("$x_1 e^{-x_2 t}$ (нелин. МНК)", 8.4, 3.0)
    sticker("логистическая\nрегрессия", 1.6, 5.3)
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.6, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Классы задач: LP ⊂ QP ⊂ QCQP ⊂ NLP и водораздел выпуклости")
    save("05_classes.png")


# ---------------------------------------------------------------- 06 выпуклая vs невыпуклая
def _bfgs_path(f, x0):
    path = [np.asarray(x0, float)]
    minimize(f, x0, method="BFGS", callback=lambda xk: path.append(xk.copy()),
             options={"gtol": 1e-8})
    return np.array(path)


def fig_convex_vs_nonconvex():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.5))
    starts = [(-4, 4), (4, 4), (-4, -3.5), (4, -3.5)]
    colors = [BLUE, ORANGE, AQUA, YELLOW]

    fq = lambda v: v[0] ** 2 + 2 * v[1] ** 2 + 0.5 * v[0] * v[1]
    X, Y = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
    a.contour(X, Y, fq([X, Y]), levels=12, colors=GRAY, linewidths=0.6)
    for s, col in zip(starts, colors):
        p = _bfgs_path(fq, s)
        a.plot(p[:, 0], p[:, 1], "-o", color=col, ms=3, lw=1.5)
        a.plot(*p[0], "o", color=col, ms=6)
    a.plot(0, 0, marker="*", ms=14, color=RED, mec="white", mew=0.8, ls="none", zorder=5)
    a.set_title("Выпуклая: все старты → один минимум", fontsize=10)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])

    fh = lambda v: (v[0] ** 2 + v[1] - 11) ** 2 + (v[0] + v[1] ** 2 - 7) ** 2
    Z = fh([X, Y])
    b.contour(X, Y, Z, levels=np.logspace(0, 3, 14), colors=GRAY, linewidths=0.6)
    ends = []
    for s, col in zip(starts, colors):
        p = _bfgs_path(fh, s)
        b.plot(p[:, 0], p[:, 1], "-o", color=col, ms=3, lw=1.5)
        b.plot(*p[0], "o", color=col, ms=6)
        ends.append(p[-1])
    ends = np.array(ends)
    b.plot(ends[:, 0], ends[:, 1], marker="*", ms=14, color=RED, mec="white", mew=0.8,
           ls="none", zorder=5)
    b.set_title("Невыпуклая: 4 старта → 4 минимума", fontsize=10)
    b.set_aspect("equal")
    b.set_xticks([])
    b.set_yticks([])
    save("06_convex_vs_nonconvex.png")


# ---------------------------------------------------------------- 07 линейный МНК (demo a)
def fig_lstsq():
    rng = np.random.default_rng(1)
    N = 40
    t = np.linspace(0, 1, N)
    y = 1.0 + 2.0 * t - 3.0 * t**2 + 0.15 * rng.standard_normal(N)
    design = lambda t, deg: np.vander(t, deg + 1, increasing=True)
    x2, *_ = np.linalg.lstsq(design(t, 2), y, rcond=None)
    x15, *_ = np.linalg.lstsq(design(t, 15), y, rcond=None)
    tt = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(t, y, "o", ms=4, color=INK, alpha=0.7, label="данные $(t_i, y_i)$")
    ax.plot(tt, design(tt, 2) @ x2, color=BLUE, label="полином степени 2")
    ax.plot(tt, design(tt, 15) @ x15, color=ORANGE, ls="--", label="полином степени 15")
    ax.plot(tt, 1 + 2 * tt - 3 * tt**2, color=GRAY, ls=":", label="истинная зависимость")
    ax.set_ylim(-1, 3)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y$")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("Линейный МНК: подгонка полинома")
    save("07_lstsq.png")


# ---------------------------------------------------------------- 08 LP (demo b)
def fig_lp():
    c = np.array([-3.0, -5.0])
    A_ub = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]])
    b_ub = np.array([4.0, 12.0, 18.0])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)], method="highs")
    x1 = np.linspace(0, 7, 300)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.fill_between(x1, 0, np.minimum(6, (18 - 3 * x1) / 2), where=x1 <= 4,
                    color=BLUE, alpha=0.15, lw=0, label="допустимое множество")
    ax.axvline(4, color=BLUE, lw=1.2)
    ax.axhline(6, color=BLUE, lw=1.2)
    ax.plot(x1, (18 - 3 * x1) / 2, color=BLUE, lw=1.2)
    for val in (10, 20, 30, 36):
        ax.plot(x1, (val - 3 * x1) / 5, ls="--", lw=0.9, color=ORANGE)
        ax.text(0.15, val / 5 + 0.1, f"$3x_1+5x_2={val}$", color=ORANGE, fontsize=7.5)
    ax.plot(*res.x, marker="*", ms=16, color=RED, mec="white", mew=0.8, ls="none", zorder=5)
    ax.text(2.2, 6.3, "$x^\\ast=(2,6)$, прибыль 36", color=RED, fontsize=9)
    ax.text(4.1, 1.0, "$x_1\\leq 4$", color=BLUE, fontsize=8.5)
    ax.text(0.3, 5.6, "$2x_2\\leq 12$", color=BLUE, fontsize=8.5)
    ax.text(4.3, 3.2, "$3x_1+2x_2\\leq 18$", color=BLUE, fontsize=8.5, rotation=-56)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 9)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("LP: решение — в вершине многоугольника")
    save("08_lp.png")


# ---------------------------------------------------------------- 09 подвешенная цепь (demo c)
def fig_chain():
    N, D, g = 40, 70.0, 9.81
    m = 4.0 / N
    p_left, p_right = np.array([-2.0, 1.0]), np.array([2.0, 1.0])

    def unpack(v):
        y = np.concatenate(([p_left[0]], v[:N], [p_right[0]]))
        z = np.concatenate(([p_left[1]], v[N:], [p_right[1]]))
        return y, z

    def energy(v):
        y, z = unpack(v)
        return 0.5 * D * np.sum(np.diff(y) ** 2 + np.diff(z) ** 2) + m * g * np.sum(z[1:-1])

    def grad(v):
        y, z = unpack(v)
        gy = D * (2 * y[1:-1] - y[:-2] - y[2:])
        gz = D * (2 * z[1:-1] - z[:-2] - z[2:]) + m * g
        return np.concatenate((gy, gz))

    v0 = np.concatenate((np.linspace(p_left[0], p_right[0], N + 2)[1:-1],
                         np.linspace(p_left[1], p_right[1], N + 2)[1:-1]))
    r1 = minimize(energy, v0, jac=grad, method="BFGS", options={"gtol": 1e-6})
    A = np.hstack((-0.1 * np.eye(N), np.eye(N)))
    floor = LinearConstraint(A, lb=0.5 * np.ones(N), ub=np.inf)
    r2 = minimize(energy, v0, jac=grad, method="SLSQP", constraints=[floor],
                  options={"ftol": 1e-12, "maxiter": 500})
    y1, z1 = unpack(r1.x)
    y2, z2 = unpack(r2.x)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(y1, z1, "-o", ms=3, color=BLUE, label="без ограничений")
    ax.plot(y2, z2, "-s", ms=3, color=ORANGE, label="с полом $z\\geq 0.5+0.1y$")
    yy = np.array([-2.3, 2.3])
    ax.plot(yy, 0.5 + 0.1 * yy, color=INK, lw=1, ls="--")
    ax.plot([-2, 2], [1, 1], "o", color=INK, ms=7)
    ax.set_xlabel("$y$")
    ax.set_ylabel("$z$")
    ax.legend(loc="upper center", fontsize=8.5)
    ax.set_title(f"Подвешенная цепь, $N={N}$: минимум потенциальной энергии")
    save("09_chain.png")


# ---------------------------------------------------------------- 10 невыпуклая 1D (demo d)
def fig_nonconvex_starts():
    f = lambda x: (x**2 - 1) ** 2 + 0.3 * x
    df = lambda x: 4 * x * (x**2 - 1) + 0.3
    xs = np.linspace(-1.8, 1.8, 400)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(xs, f(xs), color=BLUE, label="$f(x)=(x^2-1)^2+0.3x$")
    for x0 in (-1.5, 0.3, 1.5):
        r = minimize(lambda v: f(v[0]), [x0], jac=lambda v: [df(v[0])], method="BFGS")
        ax.plot(x0, f(x0), "o", color=INK, ms=6)
        ax.plot(r.x[0], r.fun, marker="*", ms=15, color=RED, mec="white", mew=0.8, ls="none", zorder=5)
        ax.annotate("", xy=(r.x[0], r.fun), xytext=(x0, f(x0)),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))
        ax.text(x0, f(x0) + 0.25, f"$x_0={x0}$", ha="center", fontsize=8.5)
    ax.text(-0.55, -0.55, "глобальный\n$f=-0.305$", ha="center", color=RED, fontsize=8.5)
    ax.text(1.45, -0.35, "локальный\n$f=0.294$", ha="center", color=RED, fontsize=8.5)
    ax.set_ylim(-0.9, 3.2)
    ax.set_xlabel("$x$")
    ax.legend(loc="upper center", fontsize=9)
    ax.set_title("Старты (точки) и найденные минимумы (звёзды)")
    save("10_nonconvex_starts.png")


# ---------------------------------------------------------------- 11 итерации
def fig_iterates():
    Q = np.array([[2.0, 0.5], [0.5, 1.0]])
    grad = lambda v: Q @ v
    x = np.array([3.2, 2.6])
    path, gnorm, eps = [x.copy()], [np.linalg.norm(grad(x))], 1e-6
    while gnorm[-1] > eps and len(path) < 200:
        x = x - 0.35 * grad(x)             # градиентный шаг с постоянным шагом
        path.append(x.copy())
        gnorm.append(np.linalg.norm(grad(x)))
    path = np.array(path)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
    X, Y = np.meshgrid(np.linspace(-1.5, 4, 200), np.linspace(-2.5, 3.5, 200))
    Z = 0.5 * (Q[0, 0] * X**2 + 2 * Q[0, 1] * X * Y + Q[1, 1] * Y**2)
    a.contour(X, Y, Z, levels=np.linspace(0.2, 25, 14), colors=GRAY, linewidths=0.6)
    a.plot(path[:, 0], path[:, 1], "-o", color=BLUE, ms=3.5, lw=1.3)
    a.plot(0, 0, marker="*", ms=14, color=RED, mec="white", mew=0.8, ls="none", zorder=5)
    for k in (0, 1, 2):
        a.text(path[k, 0] + 0.12, path[k, 1] + 0.1, f"$x_{k}$", fontsize=9)
    a.text(0.15, -0.55, "$x^\\ast$", color=RED, fontsize=10)
    a.set_title("$x_{k+1}=x_k+p_k$: итерации сходятся к $x^\\ast$")
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])

    b.semilogy(gnorm, "-o", color=BLUE, ms=3)
    b.axhline(eps, color=RED, lw=1, ls="--")
    b.text(2, 2.2e-6, r"$\varepsilon = 10^{-6}$: стоп", color=RED, fontsize=9)
    b.set_xlabel("итерация $k$")
    b.set_ylabel(r"$\|\nabla f(x_k)\|$")
    b.set_title("Критерий остановки")
    save("11_iterates.png")


# ---------------------------------------------------------------- 12 slack-переменная
def fig_slack_l1():
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    t = np.linspace(-2.2, 2.2, 300)
    ax.fill_between(t, np.abs(t), 3, color=BLUE, alpha=0.12, lw=0)
    ax.plot(t, np.abs(t), color=BLUE, lw=2)
    ax.plot(1.3, 2.4, "o", color=ORANGE, ms=7)
    ax.annotate("", xy=(1.3, 1.34), xytext=(1.3, 2.32),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.8))
    ax.text(1.42, 1.9, "минимизация $s$\nприжимает к $|t|$", color=ORANGE, fontsize=9)
    ax.text(-1.1, 2.3, r"$s\geq |t|$", color=BLUE, fontsize=12, ha="center")
    ax.text(-1.1, 1.85, r"$\Leftrightarrow\ -s\leq t\leq s$", color=BLUE, fontsize=10, ha="center")
    ax.text(0, -0.55, "негладкий $|t|$ заменён двумя линейными неравенствами",
            ha="center", fontsize=9)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.8, 3)
    ax.set_xlabel("$t = a^\\top x - b$")
    ax.set_ylabel("$s$")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_title("Вспомогательная переменная $s$")
    save("12_slack_l1.png")


if __name__ == "__main__":
    print("Сохраняю в", IMG)
    fig_feasible_set()
    fig_minima_1d()
    fig_pathologies()
    fig_coercive()
    fig_classes()
    fig_convex_vs_nonconvex()
    fig_lstsq()
    fig_lp()
    fig_chain()
    fig_nonconvex_starts()
    fig_iterates()
    fig_slack_l1()
