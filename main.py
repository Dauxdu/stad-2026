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

N_MAX = 50  # число гармоник по заданию
ORDERS = (2, 20)  # порядки многочленов для шага 5

# Шаг 1. Исходный ряд
f = pd.read_csv("data/var_01.csv").sort_values("index")["traffic"].to_numpy()
N = len(f)
sigma = f.std()
slope = np.polyfit(np.arange(N), f, 1)[0]  # наклон прямой тренда (МНК)
lag1 = np.corrcoef(f[:-1], f[1:])[
    0, 1
]  # автокорреляция: ≈ 0 у шума, ≈ 1 у гладкой волны
print(
    f"N = {N}, среднее = {f.mean():.2f}, СКО = {sigma:.2f}, min = {f.min():.2f}, max = {f.max():.2f}"
)
print(f"Наклон тренда = {slope:.4f}, автокорреляция (лаг 1) = {lag1:.3f}")
plot_series(f, "figures/series.png")

# Шаг 2. Коэффициенты Фурье
a, b = fourier_coefficients(f, N_MAX)
A = amplitudes(a, b)
print(f"\na0/2 = {a[0] / 2:.2f}")
print(f"{'n':>3} {'a_n':>8} {'b_n':>8} {'A_n':>8}")
for n in range(1, N_MAX + 1):
    print(f"{n:>3} {a[n]:>8.3f} {b[n]:>8.3f} {A[n]:>8.3f}")

# Шаги 3–4. Спектр и аномальные гармоники
peaks, threshold = three_sigma_peaks(A)
plot_spectrum(A, threshold, "figures/spectrum.png")
print(f"\nmean(A) = {A[1:].mean():.3f}, std(A) = {A[1:].std():.3f}")
print(f"Порог mean + 3*std = {threshold:.3f}, пики: {peaks or 'нет'}")
top = sorted(range(1, N_MAX + 1), key=lambda n: A[n], reverse=True)[:5]
print("Топ-5:", ", ".join(f"n={n} (A={A[n]:.2f}, N/n={N / n:.1f})" for n in top))

# Шаг 5. Тригонометрические многочлены
approx = {K: trig_polynomial(a, b, K, N) for K in ORDERS}
plot_approximations(f, approx, "figures/approximation.png")
print(f"\nТолько среднее: RMSE = {sigma:.2f}")
for K, s in approx.items():
    error = rmse(f, s)
    print(f"K = {K:>2}: RMSE = {error:.2f}, R^2 = {1 - error**2 / sigma**2:.1%}")
