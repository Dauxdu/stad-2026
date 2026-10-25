import argparse

import numpy as np
import pandas as pd

from src.dft import (
    amplitudes,
    fourier_coefficients,
    rmse,
    three_sigma_peaks,
    trig_polynomial,
)
from src.plots import plot_approximations, plot_series, plot_spectrum

parser = argparse.ArgumentParser()
parser.add_argument("data", help="CSV с колонками index, traffic")
parser.add_argument("--k-max", type=int, default=50, help="число гармоник")
parser.add_argument(
    "--orders", type=int, nargs="+", default=[2, 20], help="порядки K для шага 5"
)
args = parser.parse_args()
if max(args.orders) > args.k_max:
    parser.error("K не может быть больше --k-max")

# Шаг 1. Исходный ряд
f = pd.read_csv(args.data).sort_values("index")["traffic"].to_numpy()
N = len(f)
sigma = f.std()
slope = np.polyfit(np.arange(N), f, 1)[0]  # наклон прямой тренда
lag1 = np.corrcoef(f[:-1], f[1:])[
    0, 1
]  # автокорреляция: ≈ 0 у шума, ≈ 1 у гладкой волны
print(
    f"N = {N}, среднее = {f.mean():.2f}, СКО = {sigma:.2f}, min = {f.min():.2f}, max = {f.max():.2f}"
)
print(f"Наклон тренда = {slope:.4f}, автокорреляция (лаг 1) = {lag1:.3f}")
plot_series(f, "figures/series.png")

# Шаг 2. Коэффициенты Фурье
a, b = fourier_coefficients(f, args.k_max)
A = amplitudes(a, b)
print(f"\na0/2 = {a[0] / 2:.2f}")
print(f"{'k':>3} {'a_k':>8} {'b_k':>8} {'A_k':>8}")
for k in range(1, args.k_max + 1):
    print(f"{k:>3} {a[k]:>8.3f} {b[k]:>8.3f} {A[k]:>8.3f}")

# Шаги 3–4. Спектр и аномальные гармоники
peaks, threshold = three_sigma_peaks(A)
plot_spectrum(A, threshold, "figures/spectrum.png")
print(f"\nmean(A) = {A[1:].mean():.3f}, std(A) = {A[1:].std():.3f}")
print(f"Порог mean + 3*std = {threshold:.3f}, пики: {peaks or 'нет'}")
top = sorted(range(1, args.k_max + 1), key=lambda k: A[k], reverse=True)[:5]
print("Топ-5:", ", ".join(f"k={k} (A={A[k]:.2f}, N/k={N / k:.1f})" for k in top))

# Шаг 5. Тригонометрические многочлены
approx = {K: trig_polynomial(a, b, K, N) for K in args.orders}
plot_approximations(f, approx, "figures/approximation.png")
print(f"\nТолько среднее: RMSE = {sigma:.2f}")
for K, s in approx.items():
    error = rmse(f, s)
    print(f"K = {K:>2}: RMSE = {error:.2f}, R^2 = {1 - error**2 / sigma**2:.1%}")
