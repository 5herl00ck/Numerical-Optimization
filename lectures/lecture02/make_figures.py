"""Иллюстрации к конспекту лекции 2 -> папка img/ (коммитится в репозиторий).

Запуск:  uv run python lectures/lecture02/make_figures.py

Скрипт самодостаточен. Палитра и оформление — те же, что в лекции 1.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle
from scipy.optimize import minimize

IMG = Path(__file__).resolve().parent / "img"
IMG.mkdir(exist_ok=True)

# Палитра: категориальные цвета в фиксированном порядке; красный — только для решения / нарушения.
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


def segment(ax, a, b, inside_fn, n=400):
    """Отрезок [a, b]: зелёным там, где он внутри множества, красным пунктиром — где вышел."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.linspace(0, 1, n)
    pts = a[None, :] + t[:, None] * (b - a)[None, :]
    inside = np.array([inside_fn(p) for p in pts])
    for i in range(n - 1):
        col, ls = (AQUA, "-") if inside[i] and inside[i + 1] else (RED, "--")
        ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1], color=col, ls=ls, lw=2.2, solid_capstyle="round")
    ax.plot(*a, "o", color=INK, ms=5, zorder=5)
    ax.plot(*b, "o", color=INK, ms=5, zorder=5)
    return bool(inside.all())


# ---------------------------------------------------------------- 01 галерея множеств
def fig_sets():
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.8))
    for ax in axes.flat:
        ax.set(xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), aspect="equal", xticks=[], yticks=[])
        ax.grid(False)

    # (1) полуплоскость a^T x <= b
    ax = axes[0, 0]
    ax.fill_between([-1.6, 1.6], [-1.6, -1.6], [0.3 + 0.5 * -1.6, 0.3 + 0.5 * 1.6], color=BLUE, alpha=0.18, lw=0)
    ax.plot([-1.6, 1.6], [0.3 - 0.8, 0.3 + 0.8], color=BLUE)
    segment(ax, (-1.1, -1.2), (1.0, 0.2), lambda p: p[1] <= 0.3 + 0.5 * p[0])
    ax.set_title("полуплоскость $a^\\top x \\leq b$")

    # (2) шар
    ax = axes[0, 1]
    ax.add_patch(Circle((0, 0), 1.1, color=BLUE, alpha=0.18, lw=0))
    ax.add_patch(Circle((0, 0), 1.1, fill=False, color=BLUE, lw=2))
    segment(ax, (-0.9, -0.4), (0.5, 0.9), lambda p: p @ p <= 1.1**2)
    ax.set_title("шар $\\Vert x - c\\Vert \\leq r$")

    # (3) многогранник
    ax = axes[0, 2]
    P = np.array([[-1.2, -0.8], [0.6, -1.2], [1.3, 0.2], [0.4, 1.2], [-1.0, 0.9]])
    ax.add_patch(Polygon(P, color=BLUE, alpha=0.18, lw=0))
    ax.add_patch(Polygon(P, fill=False, color=BLUE, lw=2))

    def in_poly(p):
        # выпуклый многоугольник против часовой стрелки: точка слева от каждого ребра
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < -1e-12:
                return False
        return True

    segment(ax, (-0.9, 0.5), (0.85, -0.55), in_poly)
    ax.set_title("многогранник $Ax \\leq b$")

    # (4) окружность (только граница)
    ax = axes[1, 0]
    ax.add_patch(Circle((0, 0), 1.1, fill=False, color=ORANGE, lw=2.5))
    segment(ax, (-1.1, 0), (0.55, 1.1 * np.sqrt(3) / 2), lambda p: abs(p @ p - 1.1**2) < 0.02)
    ax.set_title("окружность $\\Vert x\\Vert = r$")

    # (5) кольцо
    ax = axes[1, 1]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill(np.r_[1.2 * np.cos(th), 0.6 * np.cos(th[::-1])], np.r_[1.2 * np.sin(th), 0.6 * np.sin(th[::-1])],
            color=ORANGE, alpha=0.18, lw=0)
    ax.add_patch(Circle((0, 0), 1.2, fill=False, color=ORANGE, lw=2))
    ax.add_patch(Circle((0, 0), 0.6, fill=False, color=ORANGE, lw=2))
    segment(ax, (-0.9, 0.5), (0.9, -0.5), lambda p: 0.6**2 <= p @ p <= 1.2**2)
    ax.set_title("кольцо $r_1 \\leq \\Vert x\\Vert \\leq r_2$")

    # (6) объединение двух шаров
    ax = axes[1, 2]
    for c in ((-0.7, 0.2), (0.75, -0.3)):
        ax.add_patch(Circle(c, 0.65, color=ORANGE, alpha=0.18, lw=0))
        ax.add_patch(Circle(c, 0.65, fill=False, color=ORANGE, lw=2))
    segment(ax, (-1.0, 0.5), (1.05, -0.6),
            lambda p: (p - np.array([-0.7, 0.2])) @ (p - np.array([-0.7, 0.2])) <= 0.65**2
            or (p - np.array([0.75, -0.3])) @ (p - np.array([0.75, -0.3])) <= 0.65**2)
    ax.set_title("объединение двух шаров")

    fig.text(0.5, 0.995, "верхний ряд — выпуклые: отрезок остаётся внутри;   нижний — невыпуклые: отрезок выходит (красный пунктир)",
             ha="center", va="top", fontsize=9.5, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.965))
    plt.savefig(IMG / "01_sets.png", bbox_inches="tight")
    plt.close()
    print("   01_sets.png")


# ---------------------------------------------------------------- 02 пересечение полуплоскостей
def fig_intersection():
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    xs = np.linspace(-0.5, 5, 300)
    # полуплоскости: x1>=0, x2>=0, x1+x2<=4, x1-2x2<=1.5 ... в форме a^T x <= b
    H = [((-1, 0), 0, "$x_1 \\geq 0$"), ((0, -1), 0, "$x_2 \\geq 0$"),
         ((1, 1), 4, "$x_1 + x_2 \\leq 4$"), ((1, -2), 1.5, "$x_1 - 2x_2 \\leq 1.5$"),
         ((-1, 2.5), 6, "$-x_1 + 2.5x_2 \\leq 6$")]
    X1, X2 = np.meshgrid(np.linspace(-0.5, 5, 400), np.linspace(-0.5, 4, 400))
    inside = np.ones_like(X1, bool)
    for (a1, a2), b, lbl in H:
        inside &= a1 * X1 + a2 * X2 <= b + 1e-9
    ax.contourf(X1, X2, inside.astype(float), levels=[0.5, 1.5], colors=[BLUE], alpha=0.25)
    for (a1, a2), b, lbl in H:
        if a2 == 0:
            ax.axvline(b / a1 if a1 else 0, color=BLUE, lw=1.3)
        else:
            ax.plot(xs, (b - a1 * xs) / a2, color=BLUE, lw=1.3)
    ax.text(1.55, 1.15, "$\\Omega = \\bigcap_i \\{a_i^\\top x \\leq b_i\\}$", fontsize=11, color=INK)
    ax.text(0.08, 3.55, "$x_1 \\geq 0$", color=BLUE, fontsize=9, rotation=90)
    ax.text(3.6, 0.1, "$x_2 \\geq 0$", color=BLUE, fontsize=9)
    ax.text(2.55, 1.95, "$x_1 + x_2 \\leq 4$", color=BLUE, fontsize=9, rotation=-45)
    ax.text(2.6, 0.32, "$x_1 - 2x_2 \\leq 1.5$", color=BLUE, fontsize=9, rotation=26)
    ax.text(0.15, 2.72, "$-x_1 + 2.5x_2 \\leq 6$", color=BLUE, fontsize=9, rotation=22)
    ax.set(xlim=(-0.5, 5), ylim=(-0.5, 4), xlabel="$x_1$", ylabel="$x_2$", aspect="equal",
           title="многогранник — пересечение полуплоскостей, поэтому выпукл")
    save("02_intersection.png")


# ---------------------------------------------------------------- 03 секущая и надграфик
def fig_secant_epigraph():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    x = np.linspace(-2.2, 2.2, 300)
    f = lambda x: 0.5 * x**2 + 0.3 * np.exp(0.6 * x)
    g = lambda x: 0.35 * (x**2 - 1) ** 2 + 0.2 * x + 0.5

    ax = axes[0]
    ax.plot(x, f(x), color=BLUE)
    a, b = -1.6, 1.4
    ax.plot([a, b], [f(a), f(b)], color=ORANGE, lw=2)
    ax.plot([a, b], [f(a), f(b)], "o", color=INK, ms=5)
    t = 0.6
    z = a + t * (b - a)
    ax.plot([z, z], [f(z), f(a) + t * (f(b) - f(a))], color=GRAY, ls=":", lw=1.2)
    ax.plot(z, f(z), "o", color=BLUE, ms=5)
    ax.plot(z, f(a) + t * (f(b) - f(a)), "o", color=ORANGE, ms=5)
    ax.text(z + 0.08, 0.5 * (f(z) + f(a) + t * (f(b) - f(a))), "$f(z) \\leq$ хорда", fontsize=9)
    ax.set(title="выпуклая: хорда над графиком", xlabel="$x$", ylim=(-0.3, 4))

    ax = axes[1]
    ax.plot(x, g(x), color=ORANGE)
    a, b = -1.3, 0.9
    ax.plot([a, b], [g(a), g(b)], color=RED, lw=2)
    ax.plot([a, b], [g(a), g(b)], "o", color=INK, ms=5)
    ax.text(-0.55, 1.05, "хорда под графиком —\nнарушение", fontsize=9, color=RED, ha="center")
    ax.set(title="невыпуклая: хорда под графиком", xlabel="$x$", ylim=(-0.3, 4))

    ax = axes[2]
    ax.fill_between(x, f(x), 4.5, color=BLUE, alpha=0.18, lw=0)
    ax.plot(x, f(x), color=BLUE)
    ax.text(0.0, 2.9, "epi $f = \\{(x, s): s \\geq f(x)\\}$", fontsize=10, ha="center", color=INK)
    ax.set(title="надграфик выпуклой $f$ — выпуклое множество", xlabel="$x$", ylim=(-0.3, 4))
    for a in axes:
        a.set_xlim(-2.2, 2.2)
    save("03_secant_epigraph.png")


# ---------------------------------------------------------------- 04 касательная под графиком
def fig_tangent():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.linspace(-2.2, 2.2, 300)
    f = lambda x: 0.5 * x**2 + 0.3 * np.exp(0.6 * x)
    df = lambda x: x + 0.18 * np.exp(0.6 * x)
    g = lambda x: 0.35 * (x**2 - 1) ** 2 + 0.2 * x + 0.5
    dg = lambda x: 1.4 * x * (x**2 - 1) + 0.2

    ax = axes[0]
    ax.plot(x, f(x), color=BLUE)
    for x0 in (-1.3, 0.9):
        ax.plot(x, f(x0) + df(x0) * (x - x0), color=ORANGE, lw=1.4)
        ax.plot(x0, f(x0), "o", color=INK, ms=5)
    ax.text(-0.35, 3.0, "$f(y) \\geq f(x) + \\nabla f(x)^\\top (y - x)$", fontsize=10, ha="center")
    ax.set(title="выпуклая: касательные под графиком", xlabel="$x$", ylim=(-1.2, 4), xlim=(-2.2, 2.2))

    ax = axes[1]
    ax.plot(x, g(x), color=ORANGE)
    x0 = 0.55
    ax.plot(x, g(x0) + dg(x0) * (x - x0), color=RED, lw=1.4)
    ax.plot(x0, g(x0), "o", color=INK, ms=5)
    ax.text(-1.55, 2.2, "касательная\nпересекает график", fontsize=9, color=RED)
    ax.set(title="невыпуклая: касательная выше графика", xlabel="$x$", ylim=(-1.2, 4), xlim=(-2.2, 2.2))
    save("04_tangent.png")


# ---------------------------------------------------------------- 05 карта гессиана
def fig_hessian_map():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    xs = np.linspace(-2, 2, 200)
    X1, X2 = np.meshgrid(xs, xs)

    # f1 = x1^2 + x1 x2 + x2^2 + 0.3 exp(x1): гессиан [[2 + 0.3 e^{x1}, 1], [1, 2]] > 0 всюду
    l1 = np.empty_like(X1)
    f1 = X1**2 + X1 * X2 + X2**2 + 0.3 * np.exp(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            H = np.array([[2 + 0.3 * np.exp(X1[i, j]), 1.0], [1.0, 2.0]])
            l1[i, j] = np.linalg.eigvalsh(H)[0]
    # f2 = (x1^2 - 1)^2 + x2^2: гессиан diag(12 x1^2 - 4, 2)
    f2 = (X1**2 - 1) ** 2 + X2**2
    l2 = np.minimum(12 * X1**2 - 4, 2.0)

    for ax, F, L, title in ((axes[0], f1, l1, "$x_1^2 + x_1 x_2 + x_2^2 + 0.3e^{x_1}$: $\\lambda_{\\min} > 0$ всюду"),
                            (axes[1], f2, l2, "$(x_1^2 - 1)^2 + x_2^2$: полоса с $\\lambda_{\\min} < 0$")):
        vmax = np.abs(L).max()
        im = ax.contourf(X1, X2, L, levels=np.linspace(-vmax, vmax, 21), cmap="RdBu", alpha=0.9)
        ax.contour(X1, X2, F, levels=10, colors=[INK], linewidths=0.5, alpha=0.5)
        if (L < 0).any():
            ax.contour(X1, X2, L, levels=[0], colors=[RED], linewidths=2)
        ax.set(title=title, xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("$\\lambda_{\\min}\\,\\nabla^2 f(x)$")
    axes[1].text(0, -1.8, "$|x_1| < 1/\\sqrt{3}$: гессиан индефинитен", fontsize=9, ha="center", color=RED,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    save("05_hessian_map.png")


# ---------------------------------------------------------------- 06 максимум аффинных
def fig_max_affine():
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    x = np.linspace(-2.5, 2.5, 400)
    lines = [(-1.2, -1.0), (-0.3, -0.2), (0.5, -0.4), (1.4, -1.6)]   # a, b:  a x + b
    F = np.max([a * x + b for a, b in lines], axis=0)
    for a, b in lines:
        ax.plot(x, a * x + b, color=GRAY, lw=1, ls="--")
    ax.fill_between(x, F, 4, color=BLUE, alpha=0.15, lw=0)
    ax.plot(x, F, color=BLUE, lw=2.5)
    ax.text(0.05, 1.9, "$f(x) = \\max_i (a_i x + b_i)$", fontsize=10, color=INK)
    ax.text(-2.35, 2.55, "надграфик = пересечение полуплоскостей", fontsize=9, color=BLUE)
    ax.set(xlim=(-2.5, 2.5), ylim=(-1.6, 3.2), xlabel="$x$",
           title="максимум аффинных функций выпукл (и негладок в изломах)")
    save("06_max_affine.png")


# ---------------------------------------------------------------- 07 подуровневые множества
def fig_sublevel():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    xs = np.linspace(-2.2, 2.2, 300)
    X1, X2 = np.meshgrid(xs, xs)
    f1 = X1**2 + X1 * X2 + X2**2 + 0.3 * np.exp(X1)
    f2 = (X1**2 - 1) ** 2 + X2**2

    ax = axes[0]
    ax.contour(X1, X2, f1, levels=[0.5, 1, 2, 3, 5, 8], colors=[GRAY], linewidths=0.8)
    ax.contourf(X1, X2, f1, levels=[-1, 2], colors=[BLUE], alpha=0.25)
    ax.contour(X1, X2, f1, levels=[2], colors=[BLUE], linewidths=2)
    segment(ax, (-1.1, 0.4), (0.7, -0.9), lambda p: p[0]**2 + p[0]*p[1] + p[1]**2 + 0.3*np.exp(p[0]) <= 2)
    ax.text(0.2, 1.75, "$\\{x : f(x) \\leq 2\\}$ выпукло", fontsize=10, ha="center", color=BLUE)
    ax.set(title="выпуклая $f$: все подуровневые множества выпуклы", xlabel="$x_1$", ylabel="$x_2$", aspect="equal")

    ax = axes[1]
    ax.contour(X1, X2, f2, levels=[0.1, 0.3, 0.6, 1, 2, 4], colors=[GRAY], linewidths=0.8)
    ax.contourf(X1, X2, f2, levels=[-1, 0.6], colors=[ORANGE], alpha=0.25)
    ax.contour(X1, X2, f2, levels=[0.6], colors=[ORANGE], linewidths=2)
    segment(ax, (-1.0, 0.3), (1.0, -0.3), lambda p: (p[0]**2 - 1)**2 + p[1]**2 <= 0.6)
    ax.text(0, 1.75, "$\\{x : f(x) \\leq 0.6\\}$ — два куска", fontsize=10, ha="center", color=ORANGE)
    ax.set(title="невыпуклая $f$: подуровневое множество распалось", xlabel="$x_1$", aspect="equal")
    for a in axes:
        a.grid(False)
        a.set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2))
    save("07_sublevel.png")


# ---------------------------------------------------------------- 08 доказательство «локальный = глобальный»
def fig_local_global_proof():
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    th = np.linspace(0, 2 * np.pi, 300)
    # выпуклая «капля»
    r = 1.6 + 0.35 * np.cos(th) + 0.15 * np.cos(2 * th)
    px, py = 0.3 + r * np.cos(th), 0.1 + 0.8 * r * np.sin(th)
    ax.fill(px, py, color=BLUE, alpha=0.15, lw=0)
    ax.plot(px, py, color=BLUE, lw=1.5)
    xs, y = np.array([-0.6, -0.2]), np.array([1.6, 0.6])
    ax.add_patch(Circle(xs, 0.55, fill=False, color=ORANGE, lw=1.5, ls="--"))
    ax.plot([xs[0], y[0]], [xs[1], y[1]], color=GRAY, lw=1.5)
    t = 0.2
    xt = xs + t * (y - xs)
    ax.plot(*xs, "o", color=RED, ms=8, zorder=5)
    ax.plot(*y, "o", color=INK, ms=6, zorder=5)
    ax.plot(*xt, "o", color=ORANGE, ms=6, zorder=5)
    ax.text(xs[0] - 0.15, xs[1] - 0.3, "$x^\\ast$", fontsize=11)
    ax.text(y[0] + 0.1, y[1] + 0.05, "$y$", fontsize=11)
    ax.text(xt[0] + 0.02, xt[1] + 0.12, "$\\tilde x = x^\\ast + t(y - x^\\ast)$", fontsize=9)
    ax.text(xs[0] - 0.75, xs[1] + 0.35, "$N$", fontsize=11, color=ORANGE)
    ax.text(1.55, -0.75, "$\\Omega$", fontsize=12, color=BLUE)
    ax.text(-1.2, 1.55, "$f(x^\\ast) \\leq f(\\tilde x) \\leq f(x^\\ast) + t\\,(f(y) - f(x^\\ast))$\n$\\Rightarrow f(y) \\geq f(x^\\ast)$",
            fontsize=9.5, color=INK)
    ax.set(xlim=(-2.2, 2.6), ylim=(-1.6, 2.1), aspect="equal", xticks=[], yticks=[],
           title="локальный минимум выпуклой задачи — глобальный")
    ax.grid(False)
    save("08_local_global_proof.png")


# ---------------------------------------------------------------- 09 условие оптимальности
def fig_optimality():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    c = np.array([2.6, 2.2])                  # центр линий уровня f = |x - c|^2 / 2
    P = np.array([[-0.5, -0.3], [1.6, -0.6], [2.1, 0.9], [0.9, 1.9], [-0.8, 1.2]])   # многоугольник
    th = np.linspace(0, 2 * np.pi, 200)

    def draw_omega(ax):
        ax.add_patch(Polygon(P, color=BLUE, alpha=0.15, lw=0))
        ax.add_patch(Polygon(P, fill=False, color=BLUE, lw=1.5))

    # решение: проекция c на многоугольник — численно
    from matplotlib.path import Path as MplPath
    poly = MplPath(P)
    cons = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n = np.array([b[1] - a[1], -(b[0] - a[0])])       # нормаль наружу (обход против часовой)
        cons.append({"type": "ineq", "fun": (lambda x, a=a, n=n: -(n @ (x - a)))})
    xstar = minimize(lambda x: 0.5 * np.sum((x - c) ** 2), x0=P.mean(0), constraints=cons, method="SLSQP").x

    ax = axes[0]
    draw_omega(ax)
    for rr in (0.4, 0.8, 1.2, 1.6, 2.0):
        ax.plot(c[0] + rr * np.cos(th), c[1] + rr * np.sin(th), color=GRAY, lw=0.8, ls=":")
    ax.plot(*c, "+", color=GRAY, ms=10, mew=1.5)
    ax.plot(*xstar, "o", color=RED, ms=9, zorder=6)
    g = xstar - c                                   # градиент в x*
    g = g / np.linalg.norm(g)
    ax.annotate("", xy=xstar - 0.9 * g, xytext=xstar, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(*(xstar - 0.5 * g + np.array([0.12, -0.12])), "$-\\nabla f(x^\\ast)$", color=RED, fontsize=10)
    for y in (P[0], P[3], P.mean(0), P[4]):
        d = y - xstar
        ax.annotate("", xy=xstar + 0.85 * d / np.linalg.norm(d), xytext=xstar,
                    arrowprops=dict(arrowstyle="->", color=AQUA, lw=1.5))
    ax.text(0.15, 0.35, "$y - x^\\ast$", color=AQUA, fontsize=10)
    ax.text(*(xstar + np.array([0.12, -0.28])), "$x^\\ast$", fontsize=11)
    ax.text(-1.1, -1.05, "$\\nabla f(x^\\ast)^\\top (y - x^\\ast) \\geq 0$ для всех $y \\in \\Omega$",
            fontsize=10, color=INK)
    ax.set(title="решение на границе: антиградиент «смотрит наружу»", xlim=(-1.2, 3.4), ylim=(-1.2, 2.8),
           aspect="equal", xticks=[], yticks=[])

    ax = axes[1]
    c2 = np.array([0.7, 0.6])
    draw_omega(ax)
    for rr in (0.3, 0.6, 0.9, 1.2):
        ax.plot(c2[0] + rr * np.cos(th), c2[1] + rr * np.sin(th), color=GRAY, lw=0.8, ls=":")
    ax.plot(*c2, "o", color=RED, ms=9, zorder=6)
    ax.text(c2[0] + 0.12, c2[1] - 0.28, "$x^\\ast$", fontsize=11)
    ax.text(-0.55, 1.55, "решение внутри: $\\nabla f(x^\\ast) = 0$", fontsize=10, color=INK)
    ax.set(title="решение внутри $\\Omega$: ограничения не мешают", xlim=(-1.2, 3.4), ylim=(-1.2, 2.8),
           aspect="equal", xticks=[], yticks=[])
    for a in axes:
        a.grid(False)
    save("09_optimality.png")


# ---------------------------------------------------------------- 10 проекции
def fig_projection():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    th = np.linspace(0, 2 * np.pi, 200)

    ax = axes[0]
    ax.add_patch(Circle((0, 0), 1, color=BLUE, alpha=0.15, lw=0))
    ax.add_patch(Circle((0, 0), 1, fill=False, color=BLUE, lw=1.5))
    p = np.array([2.1, 1.4])
    x = p / np.linalg.norm(p)
    ax.plot([0, p[0]], [0, p[1]], color=GRAY, ls=":", lw=1)
    ax.plot(*p, "s", color=INK, ms=6)
    ax.plot(*x, "o", color=RED, ms=8, zorder=5)
    ax.text(p[0] + 0.08, p[1], "$p$", fontsize=11)
    ax.text(x[0] + 0.12, x[1] - 0.32, "$x^\\ast = p / \\Vert p\\Vert$", fontsize=10)
    for y in (np.array([-0.5, 0.6]), np.array([0.2, -0.8]), np.array([-0.7, -0.3])):
        d = y - x
        ax.annotate("", xy=x + 0.9 * d, xytext=x, arrowprops=dict(arrowstyle="->", color=AQUA, lw=1.3))
    ax.annotate("", xy=x + 0.6 * (p - x) / np.linalg.norm(p - x), xytext=x,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(0.95, 1.05, "$-\\nabla f$", color=RED, fontsize=10)
    ax.set(title="проекция на шар: $\\min \\frac{1}{2}\\Vert x - p\\Vert^2$, $\\Vert x\\Vert \\leq 1$",
           xlim=(-1.4, 2.6), ylim=(-1.4, 1.9), aspect="equal", xticks=[], yticks=[])

    ax = axes[1]
    ax.add_patch(Rectangle((-1, -1), 2, 2, color=BLUE, alpha=0.15, lw=0))
    ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, color=BLUE, lw=1.5))
    pts = [np.array([1.9, 0.4]), np.array([1.7, 1.6]), np.array([-0.3, 1.8]), np.array([0.4, -0.5])]
    for p in pts:
        x = np.clip(p, -1, 1)
        ax.plot([p[0], x[0]], [p[1], x[1]], color=GRAY, ls=":", lw=1)
        ax.plot(*p, "s", color=INK, ms=5)
        ax.plot(*x, "o", color=RED, ms=7, zorder=5)
    ax.text(-1.45, -1.38, "$x^\\ast_i = \\min(\\max(p_i, l_i), u_i)$ — покоординатно", fontsize=9.5)
    ax.set(title="проекция на параллелепипед $l \\leq x \\leq u$: clip",
           xlim=(-1.5, 2.4), ylim=(-1.5, 2.2), aspect="equal", xticks=[], yticks=[])
    for a in axes:
        a.grid(False)
    save("10_projection.png")


if __name__ == "__main__":
    print("Сохраняю в", IMG)
    fig_sets()
    fig_intersection()
    fig_secant_epigraph()
    fig_tangent()
    fig_hessian_map()
    fig_max_affine()
    fig_sublevel()
    fig_local_global_proof()
    fig_optimality()
    fig_projection()
