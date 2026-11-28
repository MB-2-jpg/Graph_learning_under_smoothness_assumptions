import numpy as np
from numpy.typing import NDArray
from typing import Callable


def error_alpha(alpha: float) -> None:
    if not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise TypeError("alpha must be a real or integer number")

    if not np.isfinite(alpha) or alpha <= -1/5 or alpha == 0:
        raise ValueError("alpha must belong to ]-1/5, 0[ U ]0, +inf[")


def error_sigma(sigma: float) -> None:
    if not isinstance(sigma, (int, float, np.integer, np.floating)):
        raise TypeError("sigma must be a real or integer number")

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and strictly positive")


def error_numb_nodes(numb_nodes: int) -> None:
    if not isinstance(numb_nodes, (int, np.integer)):
        raise TypeError("numb_nodes must be an integer")

    if numb_nodes < 2:
        raise ValueError("numb_nodes must be greater than or equal to 2")


def error_weight_threshold(weight_threshold: float) -> None:
    if not isinstance(weight_threshold, (int, float, np.integer, np.floating)):
        raise TypeError("weight_threshold must be a real or integer number")

    if not np.isfinite(weight_threshold) or not 0 <= weight_threshold <= 1:
        raise ValueError("weight_threshold must belong to [0, 1]")


def error_generator(law: np.random.Generator) -> None:
    if not isinstance(law, np.random.Generator):
        raise TypeError("law must be a numpy random Generator")


def error_c(c: float, numb_nodes: int) -> None:
    if not isinstance(c, (int, float, np.integer, np.floating)):
        raise TypeError("c must be a real or integer number")

    if not np.isfinite(c) or c <= 0 or c > numb_nodes:
        raise ValueError("c must be strictly positive and less than or equal to numb_nodes")


def error_numb_connections(numb_connections: int, numb_nodes: int) -> None:
    if not isinstance(numb_connections, (int, np.integer)):
        raise TypeError("numb_connections must be an integer")

    if numb_connections <= 0 or numb_connections >= numb_nodes:
        raise ValueError("numb_connections must be between 1 and numb_nodes - 1")


def error_numb_signals(numb_signals: int) -> None:
    if not isinstance(numb_signals, (int, np.integer)):
        raise TypeError("numb_signals must be an integer")

    if numb_signals <= 0:
        raise ValueError("numb_signals must be strictly positive")


def error_noise_level(noise_level: float) -> None:
    if not isinstance(noise_level, (int, float, np.integer, np.floating)):
        raise TypeError("noise_level must be a real or integer number")

    if not np.isfinite(noise_level) or noise_level < 0:
        raise ValueError("noise_level must be finite and non-negative")


def error_filter_parameter(a: float) -> None:
    if not isinstance(a, (int, float, np.integer, np.floating)):
        raise TypeError("a must be a real or integer number")

    if not np.isfinite(a) or a < 0:
        raise ValueError("a must be finite and non-negative")


def error_filter_function(filter_function: Callable) -> None:
    if not callable(filter_function):
        raise TypeError("filter_function must be callable")


def error_node_size(node_size: int) -> None:
    if not isinstance(node_size, (int, np.integer)):
        raise TypeError("node_size must be an integer")

    if node_size <= 0:
        raise ValueError("node_size must be strictly positive")


def error_font_size(font_size: int) -> None:
    if not isinstance(font_size, (int, np.integer)):
        raise TypeError("font_size must be an integer")

    if font_size <= 0:
        raise ValueError("font_size must be strictly positive")


def error_file_name(file_name: str) -> None:
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string")

    if not file_name.strip():
        raise ValueError("file_name must not be empty")

    if "/" in file_name or "\\" in file_name:
        raise ValueError("file_name must not contain path separators")


def error_laplacian_norm(nrm: np.floating) -> None:
    if not np.isfinite(nrm) or nrm <= 0:
        raise ValueError("Laplacian norm must be finite and strictly positive")


def error_adaptive_threshold(weight_boundary: float) -> None:
    if not np.isfinite(weight_boundary) or weight_boundary <= 0:
        raise ValueError("Adaptive threshold must be finite and strictly positive")


def error_eigenvalues(eigenvalues: NDArray[np.float64]) -> None:
    if np.any(eigenvalues < -1e-12):
        raise ValueError("Laplacian eigenvalues must be non-negative")


def error_filter_response(
    response: NDArray[np.float64],
    eigenvalues: NDArray[np.float64]
) -> None:

    if response.shape != eigenvalues.shape:
        raise ValueError("Filter response and eigenvalues must have the same shape")

    if not np.all(np.isfinite(response)):
        raise ValueError("Filter response must contain only finite values")


def error_graph_weights_compatibility(
    weights_matrix: NDArray[np.float64],
    graph_weights: NDArray[np.float64]
) -> None:

    if not np.allclose(weights_matrix, graph_weights):
        raise ValueError("Graph and weights matrix are inconsistent")


def error_signals_values(signals: NDArray[np.float64]) -> None:
    if not np.all(np.isfinite(signals)):
        raise ValueError("Signals must contain only finite values")
