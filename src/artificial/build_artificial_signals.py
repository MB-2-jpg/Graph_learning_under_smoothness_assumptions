import numpy as np
from typing import Callable
from numpy.typing import NDArray

from src.artificial import filters as fil
from src.common import interim_calcul as ic
from src.artificial import control_errors as ce


def add_noise(
    signals: NDArray[np.float64],
    law: np.random.Generator,
    noise_level: float
) -> NDArray[np.float64]:

    ce.error_generator(law)
    ce.error_noise_level(noise_level)

    noise = law.normal(size=signals.shape)

    norm_signals = np.linalg.norm(signals)
    norm_noise = np.linalg.norm(noise)

    if norm_signals == 0 or norm_noise == 0:
        return signals.copy()

    return signals + noise * (
        noise_level * norm_signals / norm_noise
    )


def smooth_signals(
    weights_matrix: NDArray[np.float64],
    numb_signals: int,
    filter_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    law: np.random.Generator,
    noise_level: float
) -> NDArray[np.float64]:

    ce.error_numb_signals(numb_signals)
    ce.error_generator(law)

    normalized_weights_matrix = ic.normalized_laplacian(weights_matrix)

    eigenvalues, eigenvectors = ic.spectrum(normalized_weights_matrix)

    initial_signals = law.normal(
        size=(weights_matrix.shape[0], numb_signals)
    )

    clean_signals = fil.apply_filter(
        initial_signals,
        eigenvalues,
        eigenvectors,
        filter_function
    )

    signals = add_noise(clean_signals, law, noise_level)

    ce.error_signals_values(signals)

    return signals
