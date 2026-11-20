from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Z_vec", "pairwise_sq_dists", "S_op", "EdgeNodeOp", "as_op",
    "vec_to_W", "W_to_vec", "degrees",
    "objective_log_degree", "objective_dong",
    "fbf_primal_dual", "fbf_primal_dual_dong",
    "GridResult", "make_grid", "run_grid_parallel",
]


def Z_vec(signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = signals.shape[0]
    Z = ((signals[:, None, :] - signals[None, :, :]) ** 2).sum(-1)
    i, j = np.triu_indices(m, k=1)
    return Z[i, j], i, j


def pairwise_sq_dists(signals: np.ndarray) -> np.ndarray:
    sq = (signals ** 2).sum(1)
    Z = sq[:, None] + sq[None, :] - 2.0 * (signals @ signals.T)
    np.maximum(Z, 0.0, out=Z)          # kill the negative round-off
    i, j = np.triu_indices(signals.shape[0], k=1)
    return Z[i, j]


def S_op(m: int) -> np.ndarray:
    i, j = np.triu_indices(m, k=1)
    E = len(i)
    S = np.zeros((m, E))
    e = np.arange(E)
    S[i, e] = 1.0
    S[j, e] = 1.0
    return S


class EdgeNodeOp:
    def __init__(self, m: int):
        self.m = int(m)
        self.i, self.j = np.triu_indices(self.m, k=1)
        self.E = self.i.size
        self.shape = (self.m, self.E)
        self.norm = np.sqrt(2.0 * (self.m - 1))

    def dot(self, w: np.ndarray) -> np.ndarray:                      # S @ w
        return (np.bincount(self.i, weights=w, minlength=self.m)
                + np.bincount(self.j, weights=w, minlength=self.m))

    def T_dot(self, d: np.ndarray) -> np.ndarray:                    # S.T @ d
        return d[self.i] + d[self.j]


class _DenseOp:
    def __init__(self, S: np.ndarray):
        self.S = S
        self.m, self.E = S.shape
        self.shape = S.shape
        self.norm = np.linalg.norm(S, 2)

    def dot(self, w):
        return self.S @ w

    def T_dot(self, d):
        return self.S.T @ d


def as_op(S) -> Any:
    if isinstance(S, (EdgeNodeOp, _DenseOp)):
        return S
    if isinstance(S, np.ndarray):
        return _DenseOp(S)
    return EdgeNodeOp(int(S))


def vec_to_W(w: np.ndarray, m: int) -> np.ndarray:
    i, j = np.triu_indices(m, k=1)
    W = np.zeros((m, m))
    W[i, j] = w
    return W + W.T


def W_to_vec(W: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(W.shape[0], k=1)
    return W[i, j]


def degrees(w: np.ndarray, m: int) -> np.ndarray:
    return as_op(m).dot(w)


def objective_log_degree(w, z, alpha, beta, S) -> float:
    d = as_op(S).dot(w)
    if np.any(d <= 0):
        return np.inf
    return float(2.0 * z @ w - alpha * np.log(d).sum() + beta * (w @ w))


def objective_dong(w, z, alpha, S) -> float:
    d = as_op(S).dot(w)
    return float(2.0 * z @ w + alpha * (2.0 * (w @ w) + d @ d))


def fbf_primal_dual(z, alpha, beta, S, max_iter=100_000, eps=1e-10,
                    gamma=None, w0=None, d0=None, clip=True, warn=True):
    op = as_op(S)
    m, E = op.shape
    if gamma is None:
        gamma = 0.9 / (2.0 * beta + op.norm)      # gamma < 1/(zeta + ||K||)

    w = np.zeros(E) if w0 is None else np.asarray(w0, float).copy()
    d = np.zeros(m) if d0 is None else np.asarray(d0, float).copy()

    for it in range(max_iter):
        y   = w - gamma * (2.0 * beta * w + op.T_dot(d))        # 3: primal forward
        y_b = d + gamma * op.dot(w)                             # 4: dual forward
        p   = np.maximum(0.0, y - 2.0 * gamma * z)              # 5: prox_{gamma f1}
        p_b = (y_b - np.sqrt(y_b ** 2 + 4.0 * alpha * gamma)) / 2.0   # 6: prox_{gamma f2*}
        q   = p - gamma * (2.0 * beta * p + op.T_dot(p_b))      # 7: correction step
        q_b = p_b + gamma * op.dot(p)                           # 8
        w_new = w - y + q                                       # 9: primal update
        d_new = d - y_b + q_b                                   # 10: dual update

        # guard the relative test against the w = 0, d = 0 start
        if it > 1 and np.linalg.norm(w_new - w) <= eps * max(np.linalg.norm(w), 1e-12) \
                  and np.linalg.norm(d_new - d) <= eps * max(np.linalg.norm(d), 1e-12):
            w, d, it_done = w_new, d_new, it + 1
            break
        w, d = w_new, d_new
    else:
        it_done = max_iter
        if warn:
            print("warning: Algorithm 1 hit max_iter without meeting the tolerance")

    if clip:
        # line 9 can leave weights at -1e-16; the feasible set is w >= 0
        w = np.maximum(w, 0.0)
    return w, d, it_done


def fbf_primal_dual_dong(z, alpha, s, S, max_iter=100_000, eps=1e-10,
                         gamma=None, w0=None, c0=0.0, rescale=False,
                         clip=True, zeta_paper=False, warn=True):
    op = as_op(S)
    m, E = op.shape
    zeta = 2.0 * alpha * (m + 1.0) if zeta_paper else 4.0 * alpha * m
    K_norm = 2.0 * np.sqrt(E)                      # ||K||_2 = 2*sqrt(m(m-1)/2)
    if gamma is None:
        gamma = 0.9 / (zeta + K_norm)              # gamma < 1/(zeta + ||K||)

    w = np.zeros(E) if w0 is None else np.asarray(w0, float).copy()
    c = float(c0)

    def grad_f3(v):
        return alpha * (4.0 * v + 2.0 * op.T_dot(op.dot(v)))

    for it in range(max_iter):
        y   = w - gamma * (grad_f3(w) + 2.0 * c)        # 3: primal forward (K^T c = 2c)
        y_b = c + gamma * (2.0 * w.sum())               # 4: dual forward   (K w = 2 sum w)
        p   = np.maximum(0.0, y - 2.0 * gamma * z)      # 5: prox_{gamma f1}
        p_b = y_b - gamma * s                           # 6: prox_{gamma f2*}
        q   = p - gamma * (grad_f3(p) + 2.0 * p_b)      # 7: correction step
        q_b = p_b + gamma * (2.0 * p.sum())             # 8
        w_new = w - y + q                               # 9: primal update
        c_new = c - y_b + q_b                           # 10: dual update

        if it > 1 and np.linalg.norm(w_new - w) <= eps * max(np.linalg.norm(w), 1e-12) \
                  and abs(c_new - c) <= eps * max(abs(c), 1e-12):
            w, c, it_done = w_new, c_new, it + 1
            break
        w, c = w_new, c_new
    else:
        it_done = max_iter
        if warn:
            print("warning: Algorithm 2 hit max_iter without meeting the tolerance")

    if clip:
        w = np.maximum(w, 0.0)      # remove round-off of order 1e-16
    if rescale:
        total = 2.0 * w.sum()
        if total > 0:
            w = w * (s / total)
    return w, c, it_done


@dataclass
class GridResult:
    alpha: float
    beta: float                 # beta for model "log", s for model "dong"
    model: str
    m: int
    w: np.ndarray = field(repr=False)
    dual: Any = field(repr=False)
    n_iter: int
    converged: bool
    objective: float
    n_edges: int
    density: float              # average number of non-zero edges per node
    min_degree: float
    n_components: int
    W: Optional[np.ndarray] = field(default=None, repr=False)

    def adjacency(self) -> np.ndarray:
        return self.W if self.W is not None else vec_to_W(self.w, self.m)


def _summarise(w, dual, n_iter, alpha, second, model, m, z, max_iter,
               rel_tol, return_W) -> GridResult:
    from scipy.sparse.csgraph import connected_components

    op = as_op(m)
    wmax = w.max() if w.size and w.max() > 0 else 1.0
    keep = w > rel_tol * wmax
    W = vec_to_W(w, m)
    A = vec_to_W(keep.astype(float), m)
    ncomp, _ = connected_components(A, directed=False)
    obj = (objective_log_degree(w, z, alpha, second, op) if model == "log"
           else objective_dong(w, z, alpha, op))
    return GridResult(
        alpha=float(alpha), beta=float(second), model=model, m=int(m),
        w=w, dual=dual, n_iter=int(n_iter), converged=bool(n_iter < max_iter),
        objective=obj, n_edges=int(keep.sum()),
        density=float(2 * keep.sum() / m), min_degree=float(op.dot(w).min()),
        n_components=int(ncomp), W=(W if return_W else None),
    )


def _solve_one(task):
    z, alpha, second, model, m, max_iter, eps, rel_tol, return_W, rescale = task
    if model == "log":
        w, dual, n_it = fbf_primal_dual(z, alpha, second, m,
                                        max_iter=max_iter, eps=eps, warn=False)
    elif model == "dong":
        w, dual, n_it = fbf_primal_dual_dong(z, alpha, second, m,
                                             max_iter=max_iter, eps=eps,
                                             rescale=rescale, warn=False)
    else:
        raise ValueError(f"unknown model {model!r} (use 'log' or 'dong')")
    return _summarise(w, dual, n_it, alpha, second, model, m, z,
                      max_iter, rel_tol, return_W)


def make_grid(alpha_values: Sequence[float],
              beta_values: Sequence[float]) -> Tuple[List[float], List[float]]:
    alphas = [float(a) for a in alpha_values for _ in beta_values]
    betas = [float(b) for _ in alpha_values for b in beta_values]
    return alphas, betas


def run_grid_parallel(z, alphas: Sequence[float], betas: Sequence[float], m: int,
                      model: str = "log", max_iter: int = 100_000, eps: float = 1e-10,
                      n_jobs: int = -1, backend: str = "loky",
                      rel_tol: float = 1e-6, return_W: bool = True,
                      rescale: bool = False, progress: bool = False
                      ) -> List[GridResult]:
    alphas = [float(a) for a in alphas]
    betas = [float(b) for b in betas]
    if len(alphas) != len(betas):
        raise ValueError(f"the two lists must have the same length N, "
                         f"got {len(alphas)} and {len(betas)}")
    z = np.ascontiguousarray(z, dtype=float)

    tasks = [(z, a, b, model, m, max_iter, eps, rel_tol, return_W, rescale)
             for a, b in zip(alphas, betas)]

    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            progress = False

    if n_jobs == 1:
        it = tqdm(tasks) if progress else tasks
        return [_solve_one(t) for t in it]

    from joblib import Parallel, delayed
    verbose = 5 if progress else 0
    return list(Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(_solve_one)(t) for t in tasks))
