import numpy as np


def fourier_coefficients(f: np.ndarray, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """a_n = 1/l * sum f(x_j) cos(pi*n*x_j/l),  b_n = 1/l * sum f(x_j) sin(pi*n*x_j/l).

    Это формулы Эйлера–Фурье, где интеграл заменён суммой прямоугольников с шагом 1.
    """
    N = len(f)
    if n_max >= N / 2:  # по N точкам различимы только гармоники n < N/2
        raise ValueError("n_max должно быть меньше N/2")
    l = N / 2
    x = np.arange(N)
    a = np.zeros(n_max + 1)
    b = np.zeros(n_max + 1)
    for n in range(n_max + 1):
        a[n] = np.sum(f * np.cos(np.pi * n * x / l)) / l
        b[n] = np.sum(f * np.sin(np.pi * n * x / l)) / l
    return a, b  # a[0] / 2 — среднее значение ряда


def amplitudes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A_n = sqrt(a_n^2 + b_n^2) — амплитуда n-й гармоники."""
    return np.sqrt(a**2 + b**2)


def trig_polynomial(a: np.ndarray, b: np.ndarray, K: int, N: int) -> np.ndarray:
    """S_K(x) = a_0/2 + sum_{n=1..K} (a_n cos(pi*n*x/l) + b_n sin(pi*n*x/l))."""
    l = N / 2
    x = np.arange(N)
    s = np.full(N, a[0] / 2)
    for n in range(1, K + 1):
        s += a[n] * np.cos(np.pi * n * x / l) + b[n] * np.sin(np.pi * n * x / l)
    return s


def three_sigma_peaks(amp: np.ndarray) -> tuple[list[int], float]:
    """Критерий 1: A_n > mean(A) + 3 * std(A), n = 1..n_max (a_0 не гармоника)."""
    A = amp[1:]
    threshold = A.mean() + 3 * A.std()
    peaks = [n for n in range(1, len(amp)) if amp[n] > threshold]
    return peaks, threshold


def rmse(f: np.ndarray, s: np.ndarray) -> float:
    """Среднеквадратичная ошибка = sqrt(1/N * sum (f_j - s_j)^2)."""
    return np.sqrt(np.mean((f - s) ** 2))
