#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / "common")]

import algorithms as al            # noqa: E402
import build_reel_graphs as brg    # noqa: E402
import interim_calcul as ic        # noqa: E402
import paths                       # noqa: E402
import plot_reel_graphs as prg     # noqa: E402

GRIDS = {
    "USPS": dict(betas=np.logspace(np.log10(0.015), np.log10(0.35), 12),
                 alphas=np.logspace(np.log10(0.015), np.log10(0.60), 12),
                 ks=[2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22]),
    "MNIST": dict(betas=np.logspace(np.log10(0.02), np.log10(1.5), 10),
                  alphas=np.logspace(np.log10(0.03), np.log10(2.0), 10),
                  ks=[1, 2, 3, 4, 5, 6, 8, 10]),
    "COIL20": dict(betas=np.logspace(np.log10(0.02), np.log10(1.0), 11),
                   alphas=np.logspace(np.log10(0.03), np.log10(1.5), 11),
                   ks=[2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25]),
}

N_RUNS = 100        # k-means restarts and label subsets, as in the paper
N_DRAWS_MNIST = 40  # draws per proportion (figure 2)
N_REP_COIL = 5      # samples (figure 6)


def _thin(values, keep: int):
    values = list(values)
    if len(values) <= keep:
        return values
    index = np.linspace(0, len(values) - 1, keep).round().astype(int)
    return [values[k] for k in index]


def _save_table(records, file_name: str):
    directory = paths.ensure(paths.REPORTS_REEL)
    destination = directory / f"{file_name}.csv"
    try:
        import pandas as pd
        frame = pd.DataFrame(records)
        frame.to_csv(destination, index=False)
    except ImportError:
        keys = sorted({k for r in records for k in r})
        with open(destination, "w") as handle:
            handle.write(",".join(keys) + "\n")
            for record in records:
                handle.write(",".join(str(record.get(k, "")) for k in keys) + "\n")
        frame = None
    print(f"  wrote {destination}")
    return frame


def experiment_usps(fast, figures, n_jobs, seed=0, max_nodes=None,
                    max_iter=al.MAX_ITER_DEFAULT):
    print("\n=== USPS, 1001 images, non-uniform class sizes (section 6.2) ===")
    grid = GRIDS["USPS"]
    betas, alphas, ks = grid["betas"], grid["alphas"], grid["ks"]
    n_runs = N_RUNS
    if fast:
        betas, alphas, ks = _thin(betas, 5), _thin(alphas, 5), _thin(ks, 5)
        n_runs = 10

    law = np.random.default_rng(seed)
    signals_matrix, labels, dist_vector = brg.build_usps_sample(law)
    signals_matrix, labels, dist_vector = brg.subsample_nodes(
        signals_matrix, labels, max_nodes, law)
    numb_nodes = signals_matrix.shape[0]
    print(f"  {numb_nodes} nodes, classes {np.bincount(labels).tolist()}")

    start = time.perf_counter()
    records, store = al.sweep_models(dist_vector, numb_nodes, labels, betas, alphas, ks,
                                     label_rate=0.1, n_runs=n_runs, cluster=True,
                                     max_iter=max_iter, n_jobs=n_jobs, seed=seed,
                                     keep_weights=figures)
    for record in records:
        record["dataset"] = "USPS"
    print(f"  done in {time.perf_counter() - start:.1f} s")
    _save_table(records, "usps_density_sweep")

    for model in ("Ours", "Dong etal", "k-NN"):
        best = min((r for r in records if r["model"] == model),
                   key=lambda r: r["clustering_error"])
        print(f"  best clustering error, {model:10s} {best['clustering_error']:.3f} "
              f"at {best['density']:.1f} edges/node")
        near = prg.pick_at_density(records, model, 6.0)
        print(f"    at ~6 edges/node: {near['n_components']} components, "
              f"{near['n_disconnected_nodes']} isolated nodes")

    if figures:
        prg.display_density_curves(
            records, ["clustering_error", "lp_miss", "n_components"],
            "USPS, 1001 images", "usps_figure1")
        coords = ic.embed_2d(signals_matrix, seed)
        prg.display_graphs_by_label(store, records, coords, labels, 6.0,
                                    "USPS at ~6 edges per node, nodes by digit",
                                    "usps_graphs_by_digit")
        prg.display_graphs_by_degree(store, records, coords, numb_nodes, 6.0,
                                     "USPS at ~6 edges per node, nodes by weighted degree",
                                     "usps_graphs_by_degree")
    return records


def experiment_mnist(fast, figures, n_jobs, seed=0, max_nodes=None,
                     max_iter=al.MAX_ITER_DEFAULT):
    print('\n=== MNIST "1" vs "2", three class proportions (section 6.2) ===')
    grid = GRIDS["MNIST"]
    betas, alphas, ks = grid["betas"], grid["alphas"], grid["ks"]
    n_draws, n_runs = N_DRAWS_MNIST, N_RUNS
    if fast:
        betas, alphas, ks = _thin(betas, 4), _thin(alphas, 4), _thin(ks, 4)
        n_draws, n_runs = 3, 10

    records = []
    for numb_ones, numb_twos in brg.MNIST_PROPORTIONS:
        key = f"{numb_ones}:{numb_twos}"
        print(f"  proportion {key}  ({n_draws} draws)")
        start = time.perf_counter()
        for draw in range(n_draws):
            law = np.random.default_rng(1000 * seed + draw)
            signals_matrix, labels, dist_vector = brg.build_mnist_sample(
                numb_ones, numb_twos, law)
            signals_matrix, labels, dist_vector = brg.subsample_nodes(
                signals_matrix, labels, max_nodes, law)
            rows, _ = al.sweep_models(dist_vector, signals_matrix.shape[0], labels,
                                      betas, alphas, ks, label_rate=0.1,
                                      n_runs=n_runs, cluster=True, max_iter=max_iter,
                                      n_jobs=n_jobs, seed=draw, verbose=False)
            for row in rows:
                row.update(dataset="MNIST", proportion=key, draw=draw)
            records.extend(rows)
        print(f"    {time.perf_counter() - start:.1f} s")

    _save_table(records, "mnist_1vs2_density_sweep")

    print("  miss rate at ~4 edges per node:")
    for key in [f"{a}:{b}" for a, b in brg.MNIST_PROPORTIONS]:
        subset = [r for r in records if r["proportion"] == key]
        for model in ("Ours", "Dong etal", "k-NN"):
            near = prg.pick_at_density(subset, model, 4.0)
            if near:
                print(f"    {key:9s} {model:10s} {near['lp_miss']:.3f} "
                      f"({near['density']:.1f} edges/node)")

    if figures:
        for key in [f"{a}:{b}" for a, b in brg.MNIST_PROPORTIONS]:
            subset = [r for r in records if r["proportion"] == key]
            prg.display_density_curves(
                subset, ["lp_miss", "clustering_error", "n_components"],
                f'MNIST "1" vs "2", {key}, averaged over {n_draws} draws',
                f"mnist_figure2_{key.replace(':', '_')}", xmax=12.0)
    return records


def experiment_coil20(fast, figures, n_jobs, seed=0, max_nodes=None,
                      max_iter=al.MAX_ITER_DEFAULT):
    print("\n=== COIL-20, unbalanced class sizes (appendix C.1) ===")
    grid = GRIDS["COIL20"]
    betas, alphas, ks = grid["betas"], grid["alphas"], grid["ks"]
    n_rep, n_runs = N_REP_COIL, N_RUNS
    if fast:
        betas, alphas, ks = _thin(betas, 4), _thin(alphas, 4), _thin(ks, 4)
        n_rep, n_runs = 2, 10

    records, last = [], None
    for rep in range(n_rep):
        law = np.random.default_rng(100 * seed + rep)
        signals_matrix, labels, dist_vector = brg.build_coil20_sample(law)
        signals_matrix, labels, dist_vector = brg.subsample_nodes(
            signals_matrix, labels, max_nodes, law)
        numb_nodes = signals_matrix.shape[0]
        if rep == 0:
            print(f"  {numb_nodes} nodes, {np.unique(labels).size} objects")
        rows, store = al.sweep_models(dist_vector, numb_nodes, labels, betas, alphas, ks,
                                      label_rate=0.1, n_runs=n_runs, cluster=True,
                                      max_iter=max_iter, n_jobs=n_jobs, seed=rep,
                                      keep_weights=(figures and rep == 0),
                                      verbose=False)
        for row in rows:
            row.update(dataset="COIL20", rep=rep)
        records.extend(rows)
        if rep == 0:
            last = (signals_matrix, labels, numb_nodes, store, rows)
        print(f"  sample {rep + 1}/{n_rep} done")

    _save_table(records, "coil20_density_sweep")

    if figures:
        prg.display_density_curves(
            records, ["clustering_error", "lp_miss", "n_components"],
            f"COIL-20, averaged over {n_rep} samples", "coil20_figure6", xmax=25.0)
        signals_matrix, labels, numb_nodes, store, rows = last
        coords = ic.embed_2d(signals_matrix, seed)
        prg.display_graphs_by_label(store, rows, coords, labels, 6.0,
                                    "COIL-20 at ~6 edges per node, nodes by object",
                                    "coil20_graphs_by_object")
    return records


EXPERIMENTS = {"USPS": experiment_usps, "MNIST": experiment_mnist,
               "COIL20": experiment_coil20}


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data experiments of section 6.2 and appendix C.1.")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "USPS", "MNIST", "COIL20"])
    parser.add_argument("--fast", action="store_true",
                        help="coarser grids and fewer repetitions (a smoke test)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="cap the number of nodes; cost grows like m^2")
    parser.add_argument("--max-iter", type=int, default=al.MAX_ITER_DEFAULT)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.no_figures:
        import matplotlib
        matplotlib.use("Agg")

    # --fast is a smoke test: the full USPS sweep is ~500k edges per solve
    max_nodes = args.max_nodes
    max_iter = args.max_iter
    if args.fast:
        max_nodes = 250 if max_nodes is None else max_nodes
        max_iter = min(max_iter, 2_000)

    wanted = list(EXPERIMENTS) if args.dataset == "all" else [args.dataset]
    start = time.perf_counter()
    for name in wanted:
        EXPERIMENTS[name](args.fast, not args.no_figures, args.n_jobs, args.seed,
                          max_nodes=max_nodes, max_iter=max_iter)
    print(f"\ntotal {time.perf_counter() - start:.1f} s")
    print(f"tables  -> {paths.REPORTS_REEL}")
    print(f"figures -> {paths.OUTPUTS_REEL}")


if __name__ == "__main__":
    main()
