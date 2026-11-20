#!/usr/bin/env python3

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "EDGE_TOL",
    # artificial data
    "rescale_to", "relative_error", "edge_errors", "degree_errors",
    "f_measure", "best_threshold_f_measure", "artificial_metrics",
    "ARTIFICIAL_METRICS", "best_parameter_per_metric",
    # real data
    "density", "connectivity", "clustering_error", "subsample_digits",
    "spectral_clustering_error", "label_propagation_error", "real_data_metrics",
    # baselines / graph construction
    "gaussian_kernel_weights", "knn_graph", "tune_param_for_density",
]

_EPS = 1e-12

EDGE_TOL = 1e-6


def rescale_to(w: np.ndarray, w0: np.ndarray, ord: int = 1) -> np.ndarray:
    nw = np.linalg.norm(w, ord)
    return w * (np.linalg.norm(w0, ord) / nw) if nw > _EPS else np.zeros_like(w)


def relative_error(a: np.ndarray, b: np.ndarray, ord: int = 1) -> float:
    return float(np.linalg.norm(a - b, ord) / max(np.linalg.norm(b, ord), _EPS))


def edge_errors(w: np.ndarray, w0: np.ndarray) -> Dict[str, float]:
    return {
        "edge l-1": relative_error(rescale_to(w, w0, 1), w0, 1),
        "edge l-2": relative_error(rescale_to(w, w0, 2), w0, 2),
    }


def degree_errors(w: np.ndarray, w0: np.ndarray, m: int) -> Dict[str, float]:
    from graph_learning import degrees
    d0 = degrees(w0, m)
    return {
        "degree l-1": relative_error(degrees(rescale_to(w, w0, 1), m), d0, 1),
        "degree l-2": relative_error(degrees(rescale_to(w, w0, 2), m), d0, 2),
    }


def f_measure(w: np.ndarray, w0: np.ndarray, tol: float = EDGE_TOL) -> float:
    pred = w > tol * max(w.max(), _EPS)
    true = w0 > tol * max(w0.max(), _EPS)
    tp = float(np.count_nonzero(pred & true))
    if tp == 0:
        return 0.0
    precision = tp / np.count_nonzero(pred)
    recall = tp / np.count_nonzero(true)
    return 2.0 * precision * recall / (precision + recall)


def best_threshold_f_measure(w: np.ndarray, w0: np.ndarray,
                             n_thresholds: int = 200) -> Tuple[float, float]:
    cand = np.unique(np.quantile(w[w > 0], np.linspace(0, 1, n_thresholds))) \
        if np.any(w > 0) else np.array([0.0])
    best = (0.0, 0.0)
    for t in cand:
        f = f_measure(np.where(w > t, w, 0.0), w0, tol=0.0)
        if f > best[0]:
            best = (f, float(t))
    return best


def artificial_metrics(w: np.ndarray, w0: np.ndarray, m: int,
                       threshold_for_f: bool = False) -> Dict[str, float]:
    out = {}
    out["F-measure"] = (best_threshold_f_measure(w, w0)[0] if threshold_for_f
                        else f_measure(w, w0))
    out.update(edge_errors(w, w0))
    out.update(degree_errors(w, w0, m))
    return out


#: metric name -> True if bigger is better
ARTIFICIAL_METRICS: Dict[str, bool] = {
    "F-measure": True, "edge l-1": False, "edge l-2": False,
    "degree l-1": False, "degree l-2": False,
}


def best_parameter_per_metric(records: Sequence[dict],
                              param_keys: Sequence[str] = ("alpha", "beta"),
                              metrics: Optional[Dict[str, bool]] = None
                              ) -> Dict[str, dict]:
    metrics = ARTIFICIAL_METRICS if metrics is None else metrics
    by_param: Dict[tuple, List[dict]] = {}
    for r in records:
        by_param.setdefault(tuple(r[k] for k in param_keys), []).append(r)

    out = {}
    for metric, bigger_is_better in metrics.items():
        means = {p: float(np.mean([r[metric] for r in rs]))
                 for p, rs in by_param.items() if metric in rs[0]}
        if not means:
            continue
        pick = (max if bigger_is_better else min)(means, key=means.get)
        out[metric] = {"value": means[pick], "n_trials": len(by_param[pick]),
                       **dict(zip(param_keys, pick))}
    return out


def _support(W: np.ndarray, tol: Optional[float] = None) -> np.ndarray:
    A = np.abs(W)
    t = EDGE_TOL * A.max() if tol is None else tol
    out = A > t
    np.fill_diagonal(out, False)
    return out


def density(W: np.ndarray, tol: Optional[float] = None) -> float:
    return float(_support(W, tol).sum() / W.shape[0])


def connectivity(W: np.ndarray, tol: Optional[float] = None) -> Dict[str, int]:
    from scipy.sparse.csgraph import connected_components
    A = _support(W, tol).astype(np.int8)
    ncomp, _ = connected_components(A, directed=False)
    return {"n_components": int(ncomp),
            "n_disconnected_nodes": int((A.sum(1) == 0).sum())}


def clustering_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tl, pl = np.unique(y_true), np.unique(y_pred)
    C = np.zeros((pl.size, tl.size))
    for a, p in enumerate(pl):
        for b, t in enumerate(tl):
            C[a, b] = np.count_nonzero((y_pred == p) & (y_true == t))
    r, c = linear_sum_assignment(-C)
    return 1.0 - C[r, c].sum() / y_true.size


def spectral_clustering_error(W: np.ndarray, labels: np.ndarray,
                              k: Optional[int] = None, n_runs: int = 100,
                              rng=None) -> float:
    from sklearn.cluster import KMeans
    labels = np.asarray(labels)
    k = int(np.unique(labels).size) if k is None else int(k)

    L = np.diag(W.sum(1)) - W
    _, U = np.linalg.eigh(L)
    E = U[:, :k]                                  # k smallest eigenvectors
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.maximum(nrm, _EPS)                 # Ng et al.: unit-norm rows

    seeds = np.random.default_rng(rng).integers(0, 2 ** 31 - 1, size=n_runs)
    errs = [clustering_error(labels,
                             KMeans(n_clusters=k, n_init=1, random_state=int(s))
                             .fit_predict(E))
            for s in seeds]
    return float(np.mean(errs))


def label_propagation_error(W: np.ndarray, labels: np.ndarray,
                            label_rate: float = 0.1, n_runs: int = 100,
                            rng=None, stratified: bool = False) -> Dict[str, float]:
    from scipy.sparse.csgraph import connected_components
    from scipy.linalg import solve

    labels = np.asarray(labels)
    m = W.shape[0]
    classes = np.unique(labels)
    L = np.diag(W.sum(1)) - W
    _, comp = connected_components(_support(W).astype(np.int8), directed=False)

    rng = np.random.default_rng(rng)
    n_lab = max(int(round(label_rate * m)), 1)
    miss, unclassifiable = [], []

    for _ in range(n_runs):
        if stratified:
            idx = []
            for c in classes:
                pool = np.flatnonzero(labels == c)
                take = max(int(round(label_rate * pool.size)), 1)
                idx.append(rng.choice(pool, size=take, replace=False))
            lab_idx = np.concatenate(idx)
        else:
            lab_idx = rng.choice(m, size=n_lab, replace=False)

        is_lab = np.zeros(m, bool)
        is_lab[lab_idx] = True
        pred = np.empty(m, dtype=classes.dtype)
        done = np.zeros(m, bool)
        pred[is_lab] = labels[is_lab]
        done[is_lab] = True

        # a component is usable only if it holds at least one labeled node
        for c in np.unique(comp):
            nodes = np.flatnonzero(comp == c)
            lab_c = nodes[is_lab[nodes]]
            unl_c = nodes[~is_lab[nodes]]
            if unl_c.size == 0 or lab_c.size == 0:
                continue                              # nothing to do / hopeless
            Y = np.zeros((lab_c.size, classes.size))
            Y[np.arange(lab_c.size), np.searchsorted(classes, labels[lab_c])] = 1.0
            Luu = L[np.ix_(unl_c, unl_c)]
            B = W[np.ix_(unl_c, lab_c)] @ Y
            try:
                F = solve(Luu, B, assume_a="pos")
            except Exception:
                F = np.linalg.lstsq(Luu, B, rcond=None)[0]
            pred[unl_c] = classes[np.argmax(F, axis=1)]
            done[unl_c] = True

        test = ~is_lab
        bad = test & ~done                            # unclassifiable
        scored = test & ~bad
        unclassifiable.append(int(bad.sum()))
        miss.append(float(np.mean(pred[scored] != labels[scored]))
                    if scored.any() else np.nan)

    scored_any = np.isfinite(miss)
    return {"miss_rate": float(np.mean(np.asarray(miss)[scored_any]))
                         if scored_any.any() else float("nan"),
            "unclassifiable": float(np.mean(unclassifiable)),
            "unclassifiable_rate": float(np.mean(unclassifiable) / m)}


def real_data_metrics(W: np.ndarray, labels: np.ndarray, label_rate: float = 0.1,
                      n_runs: int = 100, rng=None) -> Dict[str, float]:
    out = {"density": density(W)}
    out.update(connectivity(W))
    out["clustering_error"] = spectral_clustering_error(W, labels, n_runs=n_runs,
                                                        rng=rng)
    lp = label_propagation_error(W, labels, label_rate=label_rate,
                                 n_runs=n_runs, rng=rng)
    out.update({"lp_miss_rate": lp["miss_rate"],
                "lp_unclassifiable": lp["unclassifiable"]})
    return out


def gaussian_kernel_weights(z: np.ndarray, sigma: float,
                            threshold: float = 0.0) -> np.ndarray:
    w = np.exp(-np.asarray(z, float) / (2.0 * sigma ** 2))
    return np.where(w > threshold, w, 0.0) if threshold > 0 else w


def knn_graph(z: np.ndarray, m: int, k: int, weights: str = "gaussian",
              symmetrize: str = "max") -> np.ndarray:
    from graph_learning import vec_to_W
    Z = vec_to_W(z, m)
    np.fill_diagonal(Z, np.inf)
    order = np.argsort(Z, axis=1)[:, :k]
    A = np.zeros((m, m), bool)
    A[np.repeat(np.arange(m), k), order.ravel()] = True

    if weights == "binary":
        Wk = A.astype(float)
    else:
        kth = np.sqrt(Z[np.arange(m), order[:, -1]])
        sigma = max(float(kth.mean()), _EPS)
        Wk = np.where(A, np.exp(-Z / (2.0 * sigma ** 2)), 0.0)

    W = np.maximum(Wk, Wk.T) if symmetrize == "max" else 0.5 * (Wk + Wk.T)
    np.fill_diagonal(W, 0.0)
    return W


def subsample_digits(factor: float = 2.6, rng=None, data=None):
    rng = np.random.default_rng(rng)
    if data is None:
        from sklearn.datasets import load_digits
        data = load_digits()
    Xall, yall = np.asarray(data.data, float), np.asarray(data.target)
    xs, ys = [], []
    for i, c in enumerate(np.unique(yall), start=1):
        pool = np.flatnonzero(yall == c)
        take = min(int(round(factor * i ** 2)), pool.size)
        idx = rng.choice(pool, size=take, replace=False)
        xs.append(Xall[idx])
        ys.append(np.full(take, c))
    return np.vstack(xs), np.concatenate(ys)


def tune_param_for_density(z: np.ndarray, m: int, target_density: float,
                           model: str = "log", alpha: float = 1.0,
                           s: Optional[float] = None,
                           bracket: Tuple[float, float] = (1e-4, 1e4),
                           n_iter: int = 14, max_iter: int = 3000,
                           eps: float = 1e-4, warm_start: bool = True,
                           return_w: bool = False):
    from graph_learning import fbf_primal_dual, fbf_primal_dual_dong, vec_to_W

    state = {}

    def solve_for(p):
        kw = dict(max_iter=max_iter, eps=eps, warn=False)
        if model == "log":
            w0, d0 = (state.get("w"), state.get("d")) if warm_start else (None, None)
            w, d, _ = fbf_primal_dual(z, alpha, p, m, w0=w0, d0=d0, **kw)
            state.update(w=w, d=d)
        elif model == "dong":
            w0 = state.get("w") if warm_start else None
            c0 = state.get("c", 0.0) if warm_start else 0.0
            w, c, _ = fbf_primal_dual_dong(z, p, (m if s is None else s), m,
                                           w0=w0, c0=c0, **kw)
            state.update(w=w, c=c)
        else:
            raise ValueError(f"unknown model {model!r} (use 'log' or 'dong')")
        return w, density(vec_to_W(w, m))

    lo, hi = bracket
    _, d_lo = solve_for(lo)
    for _ in range(4):                             # widen until lo is sparse enough
        if d_lo <= target_density:
            break
        lo /= 100.0
        _, d_lo = solve_for(lo)
    _, d_hi = solve_for(hi)
    for _ in range(4):                             # widen until hi is dense enough
        if d_hi >= target_density:
            break
        hi *= 100.0
        _, d_hi = solve_for(hi)
    if d_lo > target_density or d_hi < target_density:
        import warnings
        warnings.warn(f"density {target_density} is outside the reachable range "
                      f"[{d_lo:.2f}, {d_hi:.2f}] for model {model!r} on "
                      f"[{lo:.3g}, {hi:.3g}]; returning the closest endpoint")
        p = lo if abs(d_lo - target_density) < abs(d_hi - target_density) else hi
        w, dens = solve_for(p)
        return (float(p), dens, w) if return_w else (float(p), dens)

    w, dens = None, d_hi
    for _ in range(n_iter):
        mid = np.sqrt(lo * hi)                     # bisect in log scale
        w, dens = solve_for(mid)
        if dens < target_density:
            lo = mid
        else:
            hi = mid
    p = float(np.sqrt(lo * hi))
    w, dens = solve_for(p)
    return (p, dens, w) if return_w else (p, dens)


def demo_artificial(n_grid: int = 25, n: int = 1000, m: int = 100, seed: int = 0):
    import graph_learning as gl
    import smooth_signals as ss

    rng = np.random.default_rng(seed)
    W0, _ = ss.rgg_graph(m, rng=rng)
    w0 = gl.W_to_vec(W0)
    X = ss.smooth_signals(W0, n, "Tikhonov", noise=0.1, rng=rng)
    z = gl.pairwise_sq_dists(X)
    z = z / z.mean()
    print("=" * 78)
    print(f"ARTIFICIAL (section 6.1): Random Geometric Graph, m = {m} nodes, "
          f"n = {n} Tikhonov signals, 10% noise")
    print(f"ground truth: {int((w0 > 0).sum())} edges, "
          f"density {density(W0):.2f} edges/node")
    print("=" * 78)

    # one free parameter per model is enough (Propositions 2 and 3)
    betas = np.logspace(-2, 2, n_grid)             # model (12): alpha = 1, vary beta
    alphas = np.logspace(-4, 0, n_grid)            # model (14): s = m, vary alpha
    sigmas = np.logspace(-1, 1, n_grid)            # baseline eq. (11): vary sigma

    ours = gl.run_grid_parallel(z, [1.0] * n_grid, betas, m, model="log",
                                eps=1e-6, max_iter=20_000, return_W=False)
    dong = gl.run_grid_parallel(z, alphas, [float(m)] * n_grid, m, model="dong",
                                eps=1e-6, max_iter=20_000, return_W=False)

    recs = {"base (eq. 11)": [dict(alpha=1.0, beta=sg,
                                   **artificial_metrics(gaussian_kernel_weights(z, sg),
                                                        w0, m, threshold_for_f=True))
                              for sg in sigmas],
            "Dong etal (Alg 2)": [dict(alpha=r.alpha, beta=r.beta,
                                       **artificial_metrics(r.w, w0, m)) for r in dong],
            "Ours (Alg 1)": [dict(alpha=r.alpha, beta=r.beta,
                                  **artificial_metrics(r.w, w0, m)) for r in ours]}

    names = list(ARTIFICIAL_METRICS)
    print("best value per metric over the grid (the convention of Table 2):")
    print(f"{'':20s}" + "".join(f"{k:>12s}" for k in names))
    for label, rs in recs.items():
        best = best_parameter_per_metric(rs)
        print(f"{label:20s}" + "".join(f"{best[k]['value']:12.3f}" for k in names))
    return recs


def demo_real(n_runs: int = 20, densities=(4.0, 8.0), seed: int = 0,
              factor: float = 1.3):
    import graph_learning as gl

    X, labels = subsample_digits(factor, rng=seed)
    m = X.shape[0]
    sizes = np.bincount(labels)
    z = gl.pairwise_sq_dists(X)
    z = z / z.mean()
    print("\n" + "=" * 78)
    print(f"REAL DATA (section 6.2): {m} digit images, 10 classes of sizes "
          f"{sizes.min()}..{sizes.max()} (round({factor}*i^2), capped at what is "
          f"available)")
    print(f"labels known for 10% of the nodes, {n_runs} random draws / k-means runs")
    print("=" * 78)
    head = (f"{'model':20s}{'density':>8s}{'comps':>7s}{'isolat.':>8s}"
            f"{'clust.err':>11s}{'LP miss':>9s}{'unclass.':>10s}")
    print(head)

    for target in densities:
        graphs = {}
        b, _, w_ours = tune_param_for_density(z, m, target, model="log",
                                              alpha=1.0, return_w=True)
        graphs["Ours (Alg 1)"] = gl.vec_to_W(w_ours, m)
        a, _, w_dong = tune_param_for_density(z, m, target, model="dong",
                                              s=float(m), return_w=True)
        graphs["Dong etal (Alg 2)"] = gl.vec_to_W(w_dong, m)
        graphs["k-NN"] = knn_graph(z, m, max(int(round(target / 2)), 1))

        print(f"-- target density {target:g} edges/node")
        for label, W in graphs.items():
            r = real_data_metrics(W, labels, label_rate=0.1, n_runs=n_runs, rng=seed)
            print(f"  {label:18s}{r['density']:8.2f}{r['n_components']:7d}"
                  f"{r['n_disconnected_nodes']:8d}{r['clustering_error']:11.3f}"
                  f"{r['lp_miss_rate']:9.3f}{r['lp_unclassifiable']:10.1f}")


if __name__ == "__main__":
    import argparse, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artificial", action="store_true",
                    help="only the section 6.1 metrics")
    ap.add_argument("--real", action="store_true",
                    help="only the section 6.2 metrics")
    ap.add_argument("--fast", action="store_true",
                    help="coarser grid and fewer repetitions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    both = not (args.artificial or args.real)

    if args.artificial or both:
        demo_artificial(n_grid=9 if args.fast else 25, seed=args.seed)
    if args.real or both:
        demo_real(n_runs=5 if args.fast else 20,
                  densities=(6.0,) if args.fast else (4.0, 8.0),
                  factor=0.9 if args.fast else 1.3, seed=args.seed)
