from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

import control_errors as ce
import graph_learning as gl
import metrics as mt

#: solver settings of section 6.2 (the paper quotes ~5 s / 218 iterations at m = 1001)
EPS_DEFAULT = 1e-6
MAX_ITER_DEFAULT = 12_000


def algo_ours(dist_vector: NDArray[np.float64], numb_nodes: int, beta: float,
              alpha: float = 1.0, max_iter: int = MAX_ITER_DEFAULT,
              eps: float = EPS_DEFAULT) -> NDArray[np.float64]:
    weights_vector, _, _ = gl.fbf_primal_dual(dist_vector, alpha, beta, numb_nodes,
                                              max_iter=max_iter, eps=eps, warn=False)
    return weights_vector


def algo_dong(dist_vector: NDArray[np.float64], numb_nodes: int, alpha: float,
              s: Optional[float] = None, max_iter: int = MAX_ITER_DEFAULT,
              eps: float = EPS_DEFAULT) -> NDArray[np.float64]:
    scale = float(numb_nodes) if s is None else float(s)
    weights_vector, _, _ = gl.fbf_primal_dual_dong(dist_vector, alpha, scale, numb_nodes,
                                                   max_iter=max_iter, eps=eps, warn=False)
    return weights_vector


def algo_knn(dist_vector: NDArray[np.float64], numb_nodes: int, k: int,
             weights: str = "gaussian") -> NDArray[np.float64]:
    ce.error_numb_neighbours(k)
    return gl.W_to_vec(mt.knn_graph(dist_vector, numb_nodes, int(k), weights=weights))


def learn_grid(dist_vector: NDArray[np.float64], numb_nodes: int, model: str,
               params: Sequence[float], max_iter: int = MAX_ITER_DEFAULT,
               eps: float = EPS_DEFAULT, n_jobs: int = -1
               ) -> List[Tuple[float, NDArray[np.float64], int, bool]]:
    ce.error_model(model)
    if model == "log":
        first, second = [1.0] * len(params), [float(p) for p in params]
    else:
        first, second = [float(p) for p in params], [float(numb_nodes)] * len(params)

    results = gl.run_grid_parallel(dist_vector, first, second, numb_nodes, model=model,
                                   eps=eps, max_iter=max_iter, return_W=False,
                                   n_jobs=n_jobs)
    return [(float(r.alpha if model == "dong" else r.beta), r.w, r.n_iter, r.converged)
            for r in results]


def sweep_models(dist_vector: NDArray[np.float64], numb_nodes: int,
                 labels: NDArray[np.int64], betas: Sequence[float],
                 alphas: Sequence[float], ks: Sequence[int],
                 label_rate: float = 0.1, n_runs: int = 100,
                 cluster: bool = True, max_iter: int = MAX_ITER_DEFAULT,
                 eps: float = EPS_DEFAULT, n_jobs: int = -1, seed: int = 0,
                 keep_weights: bool = False, verbose: bool = True):
    ce.error_label_rate(label_rate)
    meta, weights, store = [], [], {}

    for name, model, params in [("Ours", "log", betas), ("Dong etal", "dong", alphas)]:
        for param, weights_vector, n_iter, converged in learn_grid(
                dist_vector, numb_nodes, model, params, max_iter, eps, n_jobs):
            meta.append({"model": name, "param": param,
                         "n_iter": n_iter, "converged": converged})
            weights.append(weights_vector)
            if keep_weights:
                store[(name, param)] = weights_vector
        if verbose:
            print(f"    {name:10s} {len(params):3d} graphs learnt")

    for k in ks:
        weights_vector = algo_knn(dist_vector, numb_nodes, k)
        meta.append({"model": "k-NN", "param": float(k), "n_iter": 0, "converged": True})
        weights.append(weights_vector)
        if keep_weights:
            store[("k-NN", float(k))] = weights_vector
    if verbose:
        print(f"    {'k-NN':10s} {len(ks):3d} graphs learnt")

    scored = _score_all(weights, numb_nodes, labels, label_rate, n_runs,
                        cluster, seed, n_jobs)
    if verbose:
        print(f"    scored {len(weights)} graphs")
    return [{**a, **b} for a, b in zip(meta, scored)], store


def score_weights(weights_vector: NDArray[np.float64], numb_nodes: int,
                  labels: NDArray[np.int64], label_rate: float = 0.1,
                  n_runs: int = 100, cluster: bool = True, seed: int = 0
                  ) -> Dict[str, float]:
    weights_matrix = gl.vec_to_W(weights_vector, numb_nodes)
    out = {"density": mt.density(weights_matrix), **mt.connectivity(weights_matrix)}
    if cluster:
        out["clustering_error"] = mt.spectral_clustering_error(
            weights_matrix, labels, n_runs=n_runs, rng=seed)
    propagation = mt.label_propagation_error(weights_matrix, labels,
                                             label_rate=label_rate,
                                             n_runs=n_runs, rng=seed)
    out["lp_miss"] = propagation["miss_rate"]
    out["lp_unclass"] = propagation["unclassifiable"]
    out["lp_unclass_rate"] = propagation["unclassifiable_rate"]
    return out


def _score_all(weights, numb_nodes, labels, label_rate, n_runs, cluster, seed, n_jobs):
    if n_jobs == 1:
        return [score_weights(w, numb_nodes, labels, label_rate, n_runs, cluster, seed)
                for w in weights]
    try:
        from joblib import Parallel, delayed
    except ImportError:
        return [score_weights(w, numb_nodes, labels, label_rate, n_runs, cluster, seed)
                for w in weights]
    return list(Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(score_weights)(w, numb_nodes, labels, label_rate, n_runs, cluster, seed)
        for w in weights))
