import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "figure.facecolor": "#0d1117",
        "savefig.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "axes.titlecolor": "#e6edf3",
        "text.color": "#c9d1d9",
        "xtick.color": "#c9d1d9",
        "ytick.color": "#c9d1d9",
        "grid.alpha": 0.3,
    }
)


def _save(fig, ax, path: str) -> None:
    ax.grid()
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_series(f, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(f, lw=0.8, label="Трафик")
    ax.axhline(f.mean(), color="orange", ls="--", label=f"среднее = {f.mean():.1f}")
    ax.set(title="Исходный ряд f(x)", xlabel="x (номер отсчёта)", ylabel="пакетов/с")
    _save(fig, ax, path)


def plot_spectrum(A, threshold, path):
    n = np.arange(1, len(A))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(n, A[1:], width=0.8)
    ax.axhline(threshold, color="red", ls="--", label=f"mean + 3·std = {threshold:.2f}")
    ax.set(
        title=f"Амплитудный спектр (n = 1..{n[-1]})",
        xlabel="n (номер гармоники)",
        ylabel="$A_n$",
    )
    _save(fig, ax, path)


def plot_approximations(f, approx, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(f, lw=0.6, color="gray", alpha=0.7, label="исходный ряд")
    for K, s in approx.items():
        ax.plot(s, lw=1.5, label=f"K = {K}")
    ax.set(
        title="Тригонометрические многочлены $S_K(x)$",
        xlabel="x (номер отсчёта)",
        ylabel="пакетов/с",
    )
    _save(fig, ax, path)
