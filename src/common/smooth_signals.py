from typing import Callable, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "laplacian", "normalize_laplacian", "spectrum",
    "h_tikhonov", "h_generative", "h_heat", "FILTERS",
    "graph_filter", "apply_filter", "add_noise", "smooth_signals",
    "smoothness", "normalized_smoothness", "graph_fourier",
    "gaussian_weights", "rgg_graph", "non_uniform_graph",
    "erdos_renyi_graph", "barabasi_albert_graph", "GRAPHS",
]


def laplacian(W: np.ndarray) -> np.ndarray:
    return np.diag(W.sum(1)) - W


def normalize_laplacian(W: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(laplacian(W), 2)
    return W / nrm if nrm > 0 else W.copy()


def spectrum(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lam, U = np.linalg.eigh(laplacian(W))
    return np.clip(lam, 0.0, None), U          # kill the -1e-16 eigenvalue


def h_tikhonov(a: float = 10.0) -> Callable[[np.ndarray], np.ndarray]:
    return lambda lam: 1.0 / (1.0 + a * np.asarray(lam, float))


def h_generative() -> Callable[[np.ndarray], np.ndarray]:
    def h(lam):
        lam = np.asarray(lam, float)
        out = np.zeros_like(lam)
        pos = lam > 1e-12
        out[pos] = 1.0 / np.sqrt(lam[pos])
        return out
    return h


def h_heat(a: float = 10.0) -> Callable[[np.ndarray], np.ndarray]:
    return lambda lam: np.exp(-a * np.asarray(lam, float))


#: the exact three filters used in the artificial experiments of the paper
FILTERS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Tikhonov":         h_tikhonov(10.0),
    "Generative Model": h_generative(),
    "Heat Diffusion":   h_heat(10.0),
}


def graph_filter(lam: np.ndarray, U: np.ndarray, h: Callable) -> np.ndarray:
    return (U * h(lam)[None, :]) @ U.T


def apply_filter(X0: np.ndarray, lam: np.ndarray, U: np.ndarray,
                 h: Callable) -> np.ndarray:
    return U @ (h(lam)[:, None] * (U.T @ X0))


def graph_fourier(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U.T @ X


def add_noise(X: np.ndarray, level: float = 0.1, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    N = rng.normal(size=X.shape)
    nx, nn = np.linalg.norm(X), np.linalg.norm(N)
    if nn == 0 or nx == 0:
        return X.copy()
    return X + N * (level * nx / nn)


def smooth_signals(W: np.ndarray, n: int = 1000, filt="Tikhonov",
                   noise: float = 0.1, rng=None, mean: Optional[np.ndarray] = None,
                   return_clean: bool = False):
    rng = np.random.default_rng(rng)
    h = FILTERS[filt] if isinstance(filt, str) else filt
    lam, U = spectrum(W)
    X0 = rng.normal(size=(W.shape[0], n))
    X_clean = apply_filter(X0, lam, U, h)
    if mean is not None:
        X_clean = X_clean + np.asarray(mean).reshape(-1, 1)
    X = add_noise(X_clean, noise, rng) if noise else X_clean.copy()
    return (X, X_clean) if return_clean else X


def smoothness(X: np.ndarray, W: np.ndarray) -> float:
    return float(np.trace(X.T @ laplacian(W) @ X))


def normalized_smoothness(X: np.ndarray, W: np.ndarray) -> float:
    nrm = np.linalg.norm(X) ** 2
    return smoothness(X, W) / nrm if nrm > 0 else 0.0


def gaussian_weights(coords: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    sq = (coords ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * coords @ coords.T, 0.0)
    W = np.exp(-D2 / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    return W


def rgg_graph(m: int = 100, sigma: float = 0.2, thresh: float = 0.6,
              rng=None, normalize: bool = True):
    rng = np.random.default_rng(rng)
    coords = rng.uniform(size=(m, 2))
    W = gaussian_weights(coords, sigma)
    W[W < thresh] = 0.0
    return (normalize_laplacian(W) if normalize else W), coords


def non_uniform_graph(m: int = 100, sigma: float = 0.2, a: float = 1.0,
                      rng=None, normalize: bool = True):
    rng = np.random.default_rng(rng)
    # inverse-cdf sampling of p(x2) ~ 1/(1+a*x2) on [0, 5]
    u = rng.uniform(size=m)
    x2 = (np.exp(u * np.log1p(a * 5.0)) - 1.0) / a
    x1 = rng.uniform(size=m)
    coords = np.c_[x1, x2]
    W = gaussian_weights(coords, sigma)
    t = W.max(1).min()
    W[W < t] = 0.0
    return (normalize_laplacian(W) if normalize else W), coords


def erdos_renyi_graph(m: int = 100, p: Optional[float] = None, rng=None,
                      normalize: bool = True):
    rng = np.random.default_rng(rng)
    p = 3.0 / m if p is None else p
    i, j = np.triu_indices(m, k=1)
    w = (rng.uniform(size=i.size) < p).astype(float)
    W = np.zeros((m, m))
    W[i, j] = w
    W = W + W.T
    return (normalize_laplacian(W) if normalize else W), None


def barabasi_albert_graph(m: int = 100, m_attach: int = 2, rng=None,
                          normalize: bool = True):
    import networkx as nx
    seed = int(np.random.default_rng(rng).integers(0, 2 ** 31 - 1))
    G = nx.barabasi_albert_graph(m, m_attach, seed=seed)
    W = nx.to_numpy_array(G, nodelist=range(m))
    return (normalize_laplacian(W) if normalize else W), None


#: name -> builder(m, rng) -> (W, coords or None), in the order of Table 2
GRAPHS: Dict[str, Callable] = {
    "Rand. Geometric": lambda m, rng: rgg_graph(m, rng=rng),
    "Non Uniform":     lambda m, rng: non_uniform_graph(m, rng=rng),
    "Erdos Renyi":     lambda m, rng: erdos_renyi_graph(m, rng=rng),
    "Barabasi-Albert": lambda m, rng: barabasi_albert_graph(m, rng=rng),
}
