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


def _new_axes(title: str, xlabel: str = "index", ylabel: str = "пакетов/с"):
    """Создаёт полотно с подписями."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return ax


def _save(ax, path: str, grid: str = "both") -> None:
    """Оформляет и сохраняет график."""
    ax.grid(axis=grid)
    ax.legend()
    fig = ax.figure
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_series(t: np.ndarray, path: str) -> None:
    ax = _new_axes("Исходный ряд")
    ax.plot(t, lw=0.8, label="traffic")
    ax.axhline(t.mean(), color="red", ls="--", label=f"среднее = {t.mean():.1f}")
    _save(ax, path)


def plot_spectrum(amp: np.ndarray, threshold: float, path: str) -> None:
    k = np.arange(1, len(amp))
    ax = _new_axes(f"Амплитудный спектр (k = 1..{k[-1]})", "k", "A_k")
    ax.bar(k, amp[1:], width=0.8)
    ax.axhline(threshold, color="red", ls="--", label=f"порог = {threshold:.2f}")
    _save(ax, path, grid="y")


def plot_approximations(
    t: np.ndarray, approx: dict[int, np.ndarray], path: str
) -> None:
    ax = _new_axes("Тригонометрические многочлены")
    ax.plot(t, lw=0.6, color="gray", alpha=0.7, label="исходный ряд")
    for order, values in approx.items():
        ax.plot(values, lw=1.5, label=f"K = {order}")
    _save(ax, path)
