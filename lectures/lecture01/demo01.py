"""
Демонстрации к лекции 1 «Введение: постановка задачи, классы задач, примеры».

Части:
  (a) линейный МНК: numpy.linalg.lstsq против нормальных уравнений;
  (b) LP планирования производства: scipy.optimize.linprog;
  (c) подвешенная цепь: безусловная QP и QP с линейными неравенствами;
  (d) невыпуклая функция одной переменной: зависимость результата от начальной точки.

Запуск:  python3 lectures/lecture01/demo01.py
Графики сохраняются в папку figures/ рядом со скриптом.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без окна: сохраняем в файлы
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(1)


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# (a) Линейный метод наименьших квадратов
# ---------------------------------------------------------------------------
def demo_least_squares() -> None:
    header("(a) Линейный МНК: подгонка полинома")
    N = 40
    t = np.linspace(0, 1, N)
    y_true = 1.0 + 2.0 * t - 3.0 * t**2
    y = y_true + 0.15 * rng.standard_normal(N)

    def design(t, deg):
        # столбцы: 1, t, t^2, ..., t^deg
        return np.vander(t, deg + 1, increasing=True)

    A = design(t, 2)

    # 1) QR-подход: numpy.linalg.lstsq (рекомендуемый)
    x_lstsq, *_ = np.linalg.lstsq(A, y, rcond=None)
    # 2) нормальные уравнения A^T A x = A^T y (для сравнения)
    x_normal = np.linalg.solve(A.T @ A, A.T @ y)

    print("истинные коэффициенты      :", np.array([1.0, 2.0, -3.0]))
    print("lstsq (QR)                 :", x_lstsq.round(4))
    print("нормальные уравнения       :", x_normal.round(4))
    print("разница между способами    : %.2e" % np.linalg.norm(x_lstsq - x_normal))
    print("cond(A) = %.1e,  cond(A^T A) = %.1e"
          % (np.linalg.cond(A), np.linalg.cond(A.T @ A)))

    # Переобучение: степень 15 на тех же данных
    A15 = design(t, 15)
    x15, *_ = np.linalg.lstsq(A15, y, rcond=None)
    print("cond(A) при степени 15 = %.1e" % np.linalg.cond(A15))

    tt = np.linspace(0, 1, 400)
    plt.figure(figsize=(6, 4))
    plt.plot(t, y, "o", ms=4, label="данные")
    plt.plot(tt, design(tt, 2) @ x_lstsq, label="степень 2")
    plt.plot(tt, design(tt, 15) @ x15, "--", label="степень 15")
    plt.plot(tt, 1 + 2 * tt - 3 * tt**2, ":", color="k", label="истина")
    plt.ylim(-1, 3)
    plt.legend()
    plt.title("Линейный МНК")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "a_least_squares.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# (b) LP: планирование производства
# ---------------------------------------------------------------------------
def demo_lp() -> None:
    header("(b) LP: планирование производства")
    # max 3 x1 + 5 x2  <=>  min -3 x1 - 5 x2
    c = np.array([-3.0, -5.0])
    A_ub = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]])
    b_ub = np.array([4.0, 12.0, 18.0])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)],
                  method="highs")
    x = res.x
    print("статус:", res.message)
    print("x* =", x.round(6), "  прибыль =", round(-res.fun, 6))
    slack = b_ub - A_ub @ x
    for i, s in enumerate(slack, 1):
        print(f"ограничение {i}: невязка {s: .3f} -> {'активно' if abs(s) < 1e-9 else 'неактивно'}")

    # Графическая иллюстрация
    x1 = np.linspace(0, 7, 200)
    plt.figure(figsize=(5, 5))
    plt.fill_between(x1, 0, np.minimum(6, (18 - 3 * x1) / 2), where=x1 <= 4,
                     alpha=0.25, label="допустимое множество")
    plt.axvline(4, color="gray", lw=1)
    plt.axhline(6, color="gray", lw=1)
    plt.plot(x1, (18 - 3 * x1) / 2, color="gray", lw=1)
    for val in (10, 20, 30, 36):
        plt.plot(x1, (val - 3 * x1) / 5, "--", lw=0.8, color="C1")
    plt.plot(*x, "r*", ms=14, label="x* = (2, 6)")
    plt.xlim(0, 7)
    plt.ylim(0, 9)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend(loc="upper right")
    plt.title("LP: линии уровня $3x_1 + 5x_2$")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "b_lp.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# (c) Подвешенная цепь
# ---------------------------------------------------------------------------
def demo_hanging_chain() -> None:
    header("(c) Подвешенная цепь: безусловная QP и QP с неравенствами")
    N = 40                      # число подвижных грузов
    m = 4.0 / N                 # масса груза, кг
    D = 70.0                    # жёсткость пружины, Н/м
    g = 9.81
    p_left = np.array([-2.0, 1.0])
    p_right = np.array([2.0, 1.0])

    def unpack(v):
        # v = [y_1..y_N, z_1..z_N]; добавляем закреплённые концы
        y = np.concatenate(([p_left[0]], v[:N], [p_right[0]]))
        z = np.concatenate(([p_left[1]], v[N:], [p_right[1]]))
        return y, z

    def energy(v):
        y, z = unpack(v)
        spring = 0.5 * D * np.sum(np.diff(y) ** 2 + np.diff(z) ** 2)
        gravity = m * g * np.sum(z[1:-1])
        return spring + gravity

    def grad(v):
        y, z = unpack(v)
        # d/dp_i  0.5 D sum ||p_{i+1}-p_i||^2 = D (2 p_i - p_{i-1} - p_{i+1})
        gy = D * (2 * y[1:-1] - y[:-2] - y[2:])
        gz = D * (2 * z[1:-1] - z[:-2] - z[2:]) + m * g
        return np.concatenate((gy, gz))

    # Энергия квадратична: E(v) = 1/2 v^T H v + c^T v + const.
    # Гессиан H — блочно-диагональный из двух трёхдиагональных блоков.
    T = D * (2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1))
    H = np.kron(np.eye(2), T)
    c = np.zeros(2 * N)
    c[0] -= D * p_left[0];  c[N - 1] -= D * p_right[0]        # концы, y
    c[N] -= D * p_left[1];  c[2 * N - 1] -= D * p_right[1]    # концы, z
    c[N:] += m * g                                            # сила тяжести
    assert np.allclose(H @ np.ones(2 * N) + c, grad(np.ones(2 * N)))

    # 1) без ограничений: минимум выпуклой QP = решение линейной системы H v = -c
    v_lin = np.linalg.solve(H, -c)
    print("без ограничений (линейная система): E=%.4f, ||grad||=%.1e, z_min=%.3f"
          % (energy(v_lin), np.linalg.norm(grad(v_lin)), v_lin[N:].min()))

    # то же итерационно, методом BFGS, из прямой между концами
    v0 = np.concatenate((np.linspace(p_left[0], p_right[0], N + 2)[1:-1],
                         np.linspace(p_left[1], p_right[1], N + 2)[1:-1]))
    r1 = minimize(energy, v0, jac=grad, method="BFGS", options={"gtol": 1e-6})
    print("без ограничений (BFGS)            : итераций=%d, E=%.4f, ||v - v_lin||=%.1e"
          % (r1.nit, r1.fun, np.linalg.norm(r1.x - v_lin)))

    # 2) с полом z_i >= 0.5 + 0.1 y_i  <=>  z_i - 0.1 y_i >= 0.5
    A = np.hstack((-0.1 * np.eye(N), np.eye(N)))
    floor = LinearConstraint(A, lb=0.5 * np.ones(N), ub=np.inf)
    r2 = minimize(energy, v0, jac=grad, method="SLSQP", constraints=[floor],
                  options={"ftol": 1e-12, "maxiter": 500})
    y2, z2 = unpack(r2.x)
    active = np.abs(A @ r2.x - 0.5) < 1e-6
    print("с полом (SLSQP)                   : успех=%s, итераций=%d, E=%.4f, "
          "активных ограничений=%d из %d"
          % (r2.success, r2.nit, r2.fun, active.sum(), N))

    y1, z1 = unpack(v_lin)
    yy = np.linspace(-2, 2, 2)
    plt.figure(figsize=(6, 4))
    plt.plot(y1, z1, "o-", ms=3, label="без ограничений")
    plt.plot(y2, z2, "s-", ms=3, label="с полом")
    plt.plot(yy, 0.5 + 0.1 * yy, "k--", lw=1, label="пол $z = 0.5 + 0.1y$")
    plt.xlabel("$y$")
    plt.ylabel("$z$")
    plt.legend()
    plt.title("Подвешенная цепь, N = %d" % N)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "c_hanging_chain.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# (d) Невыпуклая функция одной переменной
# ---------------------------------------------------------------------------
def demo_nonconvex() -> None:
    header("(d) Невыпуклая функция: результат зависит от начальной точки")

    def f(x):
        return (x**2 - 1) ** 2 + 0.3 * x

    def df(x):
        return 4 * x * (x**2 - 1) + 0.3

    results = {}
    for x0 in (-1.5, 0.3, 1.5):
        r = minimize(lambda v: f(v[0]), [x0], jac=lambda v: [df(v[0])], method="BFGS")
        results[x0] = (r.x[0], r.fun)
        print("x0 = %5.2f  ->  x* = %8.5f,  f(x*) = %8.5f" % (x0, r.x[0], r.fun))

    xs = np.linspace(-1.8, 1.8, 400)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, f(xs), label="$f(x) = (x^2-1)^2 + 0.3x$")
    for x0, (xs_, fs_) in results.items():
        plt.plot(x0, f(x0), "ko", ms=5)
        plt.plot(xs_, fs_, "r*", ms=12)
        plt.annotate("", xy=(xs_, fs_), xytext=(x0, f(x0)),
                     arrowprops=dict(arrowstyle="->", color="gray"))
    plt.legend()
    plt.title("Старты (точки) и найденные минимумы (звёзды)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "d_nonconvex.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    demo_least_squares()
    demo_lp()
    demo_hanging_chain()
    demo_nonconvex()
    print("\nГрафики сохранены в", FIG_DIR)
