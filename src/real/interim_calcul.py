import io
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from numpy.typing import NDArray

import control_errors as ce
import paths


def _cached(dataset: str, builder) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    cache = paths.data_interim(dataset) / f"{dataset.lower()}.npz"
    if cache.exists():
        store = np.load(cache)
        return store["X"].astype(np.float64), store["y"].astype(np.int64)

    signals_matrix, labels = builder()
    paths.ensure(cache.parent)
    np.savez_compressed(cache, X=signals_matrix.astype(np.float32),
                        y=labels.astype(np.int64))
    return signals_matrix.astype(np.float64), labels.astype(np.int64)


def load_usps() -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    def build():
        from scipy.io import arff
        raw = paths.data_raw("USPS") / "usps.arff"
        ce.error_raw_file_missing(raw, "USPS", "Expected usps.arff (LibSVM ARFF).")
        table, _ = arff.loadarff(raw)
        full = np.array(table.tolist(), dtype=np.float64)
        return full[:, 1:], (full[:, 0].astype(np.int64) % 10)

    return _cached("USPS", build)


def load_mnist() -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    def build():
        import pandas as pd
        from PIL import Image

        directory = paths.data_raw("MNIST")
        parquets = sorted(directory.glob("*.parquet"))
        if not parquets:
            ce.error_raw_file_missing(directory / "train-00000-of-00001.parquet",
                                      "MNIST", "Expected the HuggingFace parquet dump.")
        frame = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
        images = np.stack([
            np.asarray(Image.open(io.BytesIO(cell["bytes"])), dtype=np.uint8).ravel()
            for cell in frame["image"]
        ]).astype(np.float64)
        return images, frame["label"].to_numpy().astype(np.int64)

    return _cached("MNIST", build)


def load_coil20() -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    def build():
        raw = paths.data_raw("COIL20") / "coil20.npz"
        ce.error_raw_file_missing(raw, "COIL20", "Expected coil20.npz with arrays X, y.")
        store = np.load(raw, allow_pickle=True)
        return (np.asarray(store["X"], dtype=np.float64),
                np.asarray(store["y"]).astype(np.int64))

    return _cached("COIL20", build)


#: dataset name -> loader, in the order they appear in the paper
LOADERS = {"USPS": load_usps, "MNIST": load_mnist, "COIL20": load_coil20}


def load_dataset(dataset: str) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    ce.error_dataset(dataset)
    return LOADERS[dataset]()


def subsample_classes(signals_matrix: NDArray[np.float64], labels: NDArray[np.int64],
                      classes, sizes, law: np.random.Generator
                      ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    chosen_signals, chosen_labels = [], []
    for class_label, size in zip(classes, sizes):
        pool = np.flatnonzero(labels == class_label)
        take = int(min(size, pool.size))
        index = law.choice(pool, size=take, replace=False)
        chosen_signals.append(signals_matrix[index])
        chosen_labels.append(np.full(take, class_label, dtype=np.int64))
    return np.vstack(chosen_signals), np.concatenate(chosen_labels)


def embed_2d(signals_matrix: NDArray[np.float64], seed: int = 0) -> NDArray[np.float64]:
    from sklearn.manifold import TSNE
    numb_nodes = signals_matrix.shape[0]
    perplexity = float(min(30, max(5, (numb_nodes - 1) // 3)))
    return TSNE(n_components=2, init="pca", perplexity=perplexity,
                random_state=seed).fit_transform(signals_matrix)


def display_graph_scatter_interim(ax, weights_vector: NDArray[np.float64],
                                  coords: NDArray[np.float64],
                                  node_values: NDArray[np.float64], title: str,
                                  cmap: str = "tab10", rel_tol: float = 1e-6,
                                  vmin: Optional[float] = None,
                                  vmax: Optional[float] = None,
                                  edge_alpha: float = 0.30, node_size: int = 14):
    i, j = np.triu_indices(coords.shape[0], k=1)
    maximum = weights_vector.max() if weights_vector.size else 0.0
    keep = weights_vector > rel_tol * maximum if maximum > 0 else np.zeros_like(
        weights_vector, dtype=bool)

    if keep.any():
        segments = np.stack([coords[i[keep]], coords[j[keep]]], axis=1)
        widths = 1.6 * weights_vector[keep] / maximum
        ax.add_collection(LineCollection(segments, colors="0.45", linewidths=widths,
                                         alpha=edge_alpha, zorder=1))

    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=node_values, cmap=cmap,
                         s=node_size, vmin=vmin, vmax=vmax, zorder=2,
                         edgecolors="none")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.autoscale_view()
    return scatter


def save_figure(fig, file_name: str) -> Path:
    output_directory = paths.ensure(paths.OUTPUTS_REEL)
    destination = output_directory / f"{file_name}.png"
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    return destination
