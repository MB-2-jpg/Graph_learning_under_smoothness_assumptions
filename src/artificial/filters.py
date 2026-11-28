import numpy as np
from typing import Callable
from numpy.typing import NDArray

from src.artificial import control_errors as ce


def h_tikhonov(a: float) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    ce.error_filter_parameter(a)

    return lambda lam: 1.0 / (1.0 + a * np.asarray(lam, dtype=np.float64))


def h_generative() -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def h(lam: NDArray[np.float64]) -> NDArray[np.float64]:
        lam = np.asarray(lam, dtype=np.float64)
        out = np.zeros_like(lam)
        pos = lam > 1e-12
        out[pos] = 1.0 / np.sqrt(lam[pos])

        return out

    return h


def h_heat(a: float) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    ce.error_filter_parameter(a)

    return lambda lam: np.exp(-a * np.asarray(lam, dtype=np.float64))


def apply_filter(
    initial_signals: NDArray[np.float64],
    eigenvalues: NDArray[np.float64],
    eigenvectors: NDArray[np.float64],
    filter_function: Callable[[NDArray[np.float64]], NDArray[np.float64]]
) -> NDArray[np.float64]:

    ce.error_filter_function(filter_function)

    response = filter_function(eigenvalues)
    ce.error_filter_response(response, eigenvalues)

    return eigenvectors @ (response[:, None] * (eigenvectors.T @ initial_signals))