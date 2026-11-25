import numpy as np
from typing import Callable
from numpy.typing import NDArray
import filters as fil
import interim_calcul as ic


def add_noise(
    signals: NDArray[np.float64],
    law: np.random.Generator,
    noise_level: float = 0.1
) -> NDArray[np.float64]:

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
    noise_level: float = 0.1
) -> NDArray[np.float64]:

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

    return add_noise(clean_signals, law, noise_level)