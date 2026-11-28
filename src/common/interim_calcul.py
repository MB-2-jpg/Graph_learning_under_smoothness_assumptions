import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from numpy.typing import NDArray
from pathlib import Path

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

    ce.error_eigenvalues(eigenvalues)
    
    return np.clip(eigenvalues, 0.0, None), eigenvectors