import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from numpy.typing import NDArray
from pathlib import Path
from src.real import control_errors as ce2
import paths

from src.artificial import control_errors as ce


def generate_nodes_type_1(numb_nodes: int, law: np.random.Generator) -> NDArray[np.float64]:
    ce.error_numb_nodes(numb_nodes)
    ce.error_generator(law)

    signals_matrix_1 = law.uniform(0, 1, size=(numb_nodes,2))

    return signals_matrix_1


def generate_nodes_type_2(numb_nodes: int, alpha: float, law: np.random.Generator) -> NDArray[np.float64]:
    ce.error_numb_nodes(numb_nodes)
    ce.error_alpha(alpha)
    ce.error_generator(law)

    U = law.uniform(0, 1, size=numb_nodes)
    X1 = law.uniform(0, 1, size=numb_nodes)
    X2 = ( (1 + 5*alpha)**U - 1 ) / alpha
    signals_matrix_2 = np.column_stack((X1,X2))

    return signals_matrix_2


def generate_nodes_type_3(numb_nodes: int, c: float, law: np.random.Generator) -> int:
    p = c/numb_nodes
    X = law.binomial(1, p)

    return int(X)


def gaussian_weights(signals_matrix: NDArray[np.float64], sigma: float) -> NDArray[np.float64]: 
    ce.error_sigma(sigma)

    m = np.shape(signals_matrix)[0]
    dist_matrix = np.zeros((m, m), dtype=np.float64)

    for i in range(m):
        for j in range(m):
            if i!=j:
                dist = np.sum((signals_matrix[j, :] - signals_matrix[i, :])**2)
                dist_matrix[i,j] = dist

    weights_matrix = np.exp(-dist_matrix/(sigma**2))
    np.fill_diagonal(weights_matrix, 0)

    return weights_matrix


def weight_threshold_graph_type_2(weights_matrix: NDArray[np.float64]) -> float:
    lst_max = []
    m = np.shape(weights_matrix)[0]
    for i in range(m):
        lst_max.append(max(weights_matrix[i,:]))

    weight_boundary = min(lst_max)

    ce.error_adaptive_threshold(weight_boundary)

    return weight_boundary


def build_graph_struct_interim(signals_matrix: NDArray[np.float64], weights_matrix: NDArray[np.float64], weight_threshold: float) -> nx.Graph:
    ce.error_weight_threshold(weight_threshold)

    m = np.shape(signals_matrix)[0]
    lst_node_labels = [f"S{i}" for i in range(1, m+1)]
    G_interim = nx.Graph()
    G_interim.add_nodes_from(lst_node_labels)
    graph_struct = dict(zip(lst_node_labels, signals_matrix))
            
    for i in range(len(graph_struct)):
        for j in range(i+1, len(graph_struct)):
            if weights_matrix[i, j] > 0 and weights_matrix[i, j] >= weight_threshold:
                G_interim.add_edge(f"S{i+1}", f"S{j+1}", weight=weights_matrix[i, j])

    return G_interim


def display_graph_struct_interim(G: nx.Graph, pos: dict, font_size: int, node_size: int, file_name: str) -> None:
    ce.error_font_size(font_size)
    ce.error_node_size(node_size)
    ce.error_file_name(file_name)

    lst_weights = [G[i][j]["weight"] for i, j in G.edges()]
    plt.figure(figsize=(10,7))
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_size=font_size)
    edges = nx.draw_networkx_edges(G, pos, edge_color=lst_weights, width=lst_weights, edge_cmap=plt.get_cmap("coolwarm"))

    if isinstance(edges, LineCollection):
        plt.colorbar(edges, ax=plt.gca(), label="Poids des arêtes")

    plt.axis("off")

    output_directory = Path(__file__).resolve().parent.parent / "outputs" / "artificial_graphs"
    output_directory.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_directory / f"{file_name}.png", dpi=300, bbox_inches="tight")
    plt.show()


def laplacian(weights_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    
    return np.diag(weights_matrix.sum(1)) - weights_matrix


def normalized_laplacian(weights_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    nrm = np.linalg.norm(laplacian(weights_matrix), 2)

    ce.error_laplacian_norm(nrm)

    return (weights_matrix / nrm) if nrm > 0 else weights_matrix.copy()


def spectrum(weights_matrix: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian(weights_matrix))




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
    ce2.error_dataset(dataset)
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

    ce2.error_eigenvalues(eigenvalues)
    
    return np.clip(eigenvalues, 0.0, None), eigenvectors
