import numpy as np


def fourier_coefficients(T: np.ndarray, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    """a_k = 2/N * sum T(j) cos(2*pi*k*j/N),  b_k = 2/N * sum T(j) sin(2*pi*k*j/N).

    Считается по формулам ДПФ, k = 0..k_max.
    """
    N = len(T)
    if k_max >= N / 2:  # по N точкам различимы только гармоники k < N/2
        raise ValueError("k_max должно быть меньше N/2")
    j = np.arange(N)
    a = np.zeros(k_max + 1)
    b = np.zeros(k_max + 1)
    for k in range(k_max + 1):
        a[k] = 2 / N * np.sum(T * np.cos(2 * np.pi * k * j / N))
        b[k] = 2 / N * np.sum(T * np.sin(2 * np.pi * k * j / N))
    return a, b  # a[0] / 2 — среднее значение ряда


def amplitudes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A_k = sqrt(a_k^2 + b_k^2) — амплитуда k-й гармоники."""
    return np.sqrt(a**2 + b**2)


def trig_polynomial(a: np.ndarray, b: np.ndarray, K: int, N: int) -> np.ndarray:
    """T(t) ≈ a_0/2 + sum_{k=1..K} (a_k cos(2*pi*k*t/N) + b_k sin(2*pi*k*t/N))."""
    t = np.arange(N)
    s = np.full(N, a[0] / 2)
    for k in range(1, K + 1):
        s += a[k] * np.cos(2 * np.pi * k * t / N) + b[k] * np.sin(2 * np.pi * k * t / N)
    return s


def three_sigma_peaks(A: np.ndarray) -> tuple[list[int], float]:
    """Пик: A_k > mean(A) + 3 * std(A), k = 1..k_max (a_0 не гармоника)."""
    threshold = A[1:].mean() + 3 * A[1:].std()
    peaks = [k for k in range(1, len(A)) if A[k] > threshold]
    return peaks, threshold


def rmse(T: np.ndarray, s: np.ndarray) -> float:
    """Среднеквадратичная ошибка = sqrt(1/N * sum (T(j) - s(j))^2)."""
    return np.sqrt(np.mean((T - s) ** 2))
