import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.dft import amplitudes, find_peaks, fourier_coefficients, rmse, trig_polynomial
from src.plots import plot_approximations, plot_series, plot_spectrum

DATA_FILE = Path("data/var_01.csv")
FIG_DIR = Path("figures")
K_MAX = 50
ORDERS = (2, 20)


def main() -> None:
    """Выполняет шаги 1–5 задания и печатает результаты."""
    df = pd.read_csv(DATA_FILE)
    df = df.sort_values("index")["traffic"].to_numpy(dtype=float)
    FIG_DIR.mkdir(exist_ok=True)

    print(
        f"N = {len(df)}, среднее = {df.mean():.2f}, СКО = {df.std():.2f}, "
        f"min = {df.min():.2f}, max = {df.max():.2f}"
    )
    plot_series(df, FIG_DIR / "series.png")

    a, b = fourier_coefficients(df, K_MAX)
    amp = amplitudes(a, b)
    print(f"\na0/2 = {a[0] / 2:.2f}")
    print(f"{'k':>3} {'a_k':>8} {'b_k':>8} {'A_k':>8}")
    for k in range(1, K_MAX + 1):
        print(f"{k:>3} {a[k]:>8.3f} {b[k]:>8.3f} {amp[k]:>8.3f}")

    peaks, threshold = find_peaks(amp)
    plot_spectrum(amp, threshold, FIG_DIR / "spectrum.png")
    print(f"\nПорог (mean + 3*std) = {threshold:.3f}")
    print(f"Аномальные пики: {peaks if peaks else 'нет'}")
    top = sorted(range(1, K_MAX + 1), key=lambda k: amp[k], reverse=True)[:5]
    print("Топ-5 гармоник:", ", ".join(f"k={k} (A={amp[k]:.2f})" for k in top))

    approx = {order: trig_polynomial(a, b, order, len(df)) for order in ORDERS}
    plot_approximations(df, approx, FIG_DIR / "approximation.png")
    print(f"\nRMSE константы (a0/2) = {rmse(df, np.full_like(df, a[0] / 2)):.2f}")
    for order, values in approx.items():
        print(f"RMSE K = {order:>2}: {rmse(df, values):.2f}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}")
        sys.exit(1)
