#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / "common")]

import graph_learning as gl       # noqa: E402
import metrics as mt              # noqa: E402
import paths                      # noqa: E402
import smooth_signals as ss       # noqa: E402

M = 100          # nodes, as in the paper
NOISE = 0.10     # 10% in the l-2 sense
N_TRIALS = 5     # the paper averages over 20
N_GRID = 13
EPS = 1e-6
MAX_ITER = 20_000


def _grids(n_grid: int):
    return (np.logspace(-2, 2, n_grid),    # Ours:      alpha = 1, sweep beta
            np.logspace(-2, 1, n_grid),    # Dong etal: s = m,     sweep alpha
            np.logspace(-1, 1, n_grid))    # baseline eq. (11):    sweep sigma


def one_trial(graph_name: str, filter_name: str, seed: int, n_signals: int,
              n_grid: int = N_GRID, n_jobs: int = -1):
    law = np.random.default_rng(seed)
    betas, alphas, sigmas = _grids(n_grid)

    weights_matrix_0, _ = ss.GRAPHS[graph_name](M, law)
    weights_vector_0 = gl.W_to_vec(weights_matrix_0)

    signals_matrix = ss.smooth_signals(weights_matrix_0, n_signals, filter_name,
                                       noise=NOISE, rng=law)
    dist_vector = gl.pairwise_sq_dists(signals_matrix)
    dist_vector = dist_vector / dist_vector.mean()

    records = []
    ours = gl.run_grid_parallel(dist_vector, [1.0] * n_grid, betas, M, model="log",
                                eps=EPS, max_iter=MAX_ITER, return_W=False, n_jobs=n_jobs)
    dong = gl.run_grid_parallel(dist_vector, alphas, [float(M)] * n_grid, M,
                                model="dong", eps=EPS, max_iter=MAX_ITER,
                                return_W=False, n_jobs=n_jobs)

    for result in ours:
        records.append(dict(model="Ours", param=result.beta, density=result.density,
                            n_components=result.n_components,
                            converged=result.converged,
                            **mt.artificial_metrics(result.w, weights_vector_0, M)))
    for result in dong:
        records.append(dict(model="Dong etal", param=result.alpha,
                            density=result.density,
                            n_components=result.n_components,
                            converged=result.converged,
                            **mt.artificial_metrics(result.w, weights_vector_0, M)))
    for sigma in sigmas:
        weights_vector = mt.gaussian_kernel_weights(dist_vector, sigma)
        records.append(dict(model="base", param=float(sigma), density=float(M - 1),
                            n_components=1, converged=True,
                            **mt.artificial_metrics(weights_vector, weights_vector_0,
                                                    M, threshold_for_f=True)))

    for record in records:
        record.update(graph=graph_name, filter=filter_name, trial=seed)
    return records


def run_all(n_signals: int, n_trials: int, n_grid: int, graphs, filters, n_jobs: int):
    records, todo = [], [(g, f) for g in graphs for f in filters]
    for k, (graph_name, filter_name) in enumerate(todo, 1):
        start = time.perf_counter()
        for trial in range(n_trials):
            records.extend(one_trial(graph_name, filter_name, trial, n_signals,
                                     n_grid, n_jobs))
        print(f"  [{k:2d}/{len(todo)}] {graph_name:16s} {filter_name:17s} "
              f"{time.perf_counter() - start:6.1f} s")
    return records


def summarize(records, graphs, filters):
    metrics = list(mt.ARTIFICIAL_METRICS)
    rows = []
    for graph_name in graphs:
        for filter_name in filters:
            for model in ("base", "Dong etal", "Ours"):
                subset = [r for r in records if r["graph"] == graph_name
                          and r["filter"] == filter_name and r["model"] == model]
                if not subset:
                    continue
                best = mt.best_parameter_per_metric(subset, param_keys=("param",))
                row = {"graph": graph_name, "filter": filter_name, "model": model}
                for metric in metrics:
                    if metric in best:
                        row[metric] = round(best[metric]["value"], 4)
                        row[f"{metric} @param"] = round(best[metric]["param"], 4)
                rows.append(row)
    return rows


def _save_table(rows, file_name: str):
    directory = paths.ensure(paths.REPORTS_ARTIFICIAL)
    destination = directory / f"{file_name}.csv"
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(destination, index=False)
    except ImportError:
        keys = sorted({k for r in rows for k in r})
        with open(destination, "w") as handle:
            handle.write(",".join(keys) + "\n")
            for row in rows:
                handle.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    print(f"  wrote {destination}")


def display_summary(table_rows, n_signals: int, file_name: str) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    metrics = list(mt.ARTIFICIAL_METRICS)
    colors = {"base": "tab:green", "Dong etal": "tab:orange", "Ours": "tab:blue"}
    pairs = sorted({(r["graph"], r["filter"]) for r in table_rows},
                   key=lambda p: (p[0], p[1]))
    labels = [f"{g}\n{f}" for g, f in pairs]
    x = np.arange(len(pairs))

    figure, axes = plt.subplots(len(metrics), 1, figsize=(15, 2.9 * len(metrics)),
                                sharex=True)
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        for k, model in enumerate(["base", "Dong etal", "Ours"]):
            values = []
            for graph_name, filter_name in pairs:
                hit = [r for r in table_rows if r["graph"] == graph_name
                       and r["filter"] == filter_name and r["model"] == model]
                values.append(hit[0].get(metric, np.nan) if hit else np.nan)
            ax.bar(x + (k - 1) * 0.27, values, width=0.27, label=model,
                   color=colors[model])
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(ncol=3, fontsize=9)
    axes[-1].set_xticks(x, labels, fontsize=7, rotation=0)
    figure.suptitle(f"Artificial experiments, section 6.1, n = {n_signals} signals",
                    fontsize=12)
    figure.tight_layout()

    directory = paths.ensure(paths.OUTPUTS_ARTIFICIAL)
    destination = directory / f"{file_name}.png"
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    print(f"  wrote {destination}")
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "ps", "svg", "template"):
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Artificial experiments of section 6.1 (Tables 2 and 4).")
    parser.add_argument("--n-signals", type=int, default=1000,
                        help="1000 reproduces Table 2, 100 reproduces Table 4")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--grid", type=int, default=N_GRID)
    parser.add_argument("--fast", action="store_true",
                        help="1 graph family, 1 filter, 1 trial, coarse grid")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    if args.no_figures:
        import matplotlib
        matplotlib.use("Agg")

    graphs, filters = list(ss.GRAPHS), list(ss.FILTERS)
    n_trials, n_grid = args.trials, args.grid
    if args.fast:
        graphs, filters, n_trials, n_grid = graphs[:1], filters[:1], 1, 5

    print(f"=== artificial experiments, n = {args.n_signals} signals, "
          f"{n_trials} trial(s), {len(graphs)}x{len(filters)} settings ===")
    start = time.perf_counter()
    records = run_all(args.n_signals, n_trials, n_grid, graphs, filters, args.n_jobs)
    print(f"  {len(records)} records in {time.perf_counter() - start:.1f} s")

    tag = f"n{args.n_signals}"
    _save_table(records, f"artificial_raw_records_{tag}")
    table_rows = summarize(records, graphs, filters)
    _save_table(table_rows, f"artificial_table_{tag}")

    for row in table_rows:
        print(f"  {row['graph']:16s} {row['filter']:17s} {row['model']:10s} "
              + "  ".join(f"{m}={row.get(m, float('nan')):.3f}"
                          for m in mt.ARTIFICIAL_METRICS))

    if not args.no_figures:
        display_summary(table_rows, args.n_signals, f"artificial_summary_{tag}")

    print(f"\ntables  -> {paths.REPORTS_ARTIFICIAL}")
    print(f"figures -> {paths.OUTPUTS_ARTIFICIAL}")


if __name__ == "__main__":
    main()
