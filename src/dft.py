import numpy as np


def fourier_coefficients(t: np.ndarray, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Считает a_k и b_k для k = 0..k_max по формулам ДПФ.

    a_k = 2/N * sum T(j) cos(2*pi*k*j/N),  b_k = 2/N * sum T(j) sin(2*pi*k*j/N).
    При k = 0 получается a_0, а среднее значение ряда равно a_0 / 2.
    """
    n = len(t)
    j = np.arange(n)
    a = np.zeros(k_max + 1)
    b = np.zeros(k_max + 1)
    for k in range(k_max + 1):
        angle = 2 * np.pi * k * j / n
        a[k] = 2 / n * np.sum(t * np.cos(angle))
        b[k] = 2 / n * np.sum(t * np.sin(angle))
    return a, b


def amplitudes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Амплитуды гармоник A_k = sqrt(a_k^2 + b_k^2)."""
    return np.sqrt(a**2 + b**2)


def trig_polynomial(a: np.ndarray, b: np.ndarray, order: int, n: int) -> np.ndarray:
    """Тригонометрический многочлен порядка K = order в точках t = 0..n-1."""
    t = np.arange(n)
    result = np.full(n, a[0] / 2)
    for k in range(1, order + 1):
        angle = 2 * np.pi * k * t / n
        result += a[k] * np.cos(angle) + b[k] * np.sin(angle)
    return result


def find_peaks(amp: np.ndarray, n_sigma: float = 3.0) -> tuple[list[int], float]:
    """Ищет гармоники k >= 1, амплитуда которых выше mean + n_sigma * std.

    Возвращает список номеров k и значение порога.
    """
    harmonics = amp[1:]
    threshold = harmonics.mean() + n_sigma * harmonics.std()
    peaks = [k for k in range(1, len(amp)) if amp[k] > threshold]
    return peaks, threshold


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """Среднеквадратичная ошибка между двумя рядами."""
    return float(np.sqrt(np.mean((x - y) ** 2)))
