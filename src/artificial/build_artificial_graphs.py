import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.common import interim_calcul as ic
from src.artificial import control_errors as ce


def generate_graph_type_1(numb_nodes: int, weight_threshold: float, sigma: float, law: np.random.Generator) -> tuple[nx.Graph, NDArray[np.float64], NDArray[np.float64]]:
    ce.error_weight_threshold(weight_threshold)

    signals_matrix = ic.generate_nodes_type_1(numb_nodes, law)
    weights_matrix = ic.gaussian_weights(signals_matrix, sigma)

    weights_matrix[weights_matrix < weight_threshold] = 0.0

    G = ic.build_graph_struct_interim(signals_matrix, weights_matrix, weight_threshold)

    return (G, signals_matrix, weights_matrix)


def generate_graph_type_2(numb_nodes: int, alpha: float, sigma: float, law: np.random.Generator) -> tuple[nx.Graph, NDArray[np.float64], NDArray[np.float64]]:
    signals_matrix = ic.generate_nodes_type_2(numb_nodes, alpha, law)
    weights_matrix = ic.gaussian_weights(signals_matrix, sigma)

    weight_threshold = ic.weight_threshold_graph_type_2(weights_matrix)
    weights_matrix[weights_matrix < weight_threshold] = 0.0

    G = ic.build_graph_struct_interim(signals_matrix, weights_matrix, weight_threshold)

    return (G, signals_matrix, weights_matrix)


def generate_graph_type_3(numb_nodes: int, c: float, law: np.random.Generator) -> tuple[nx.Graph, NDArray[np.float64]]:
    ce.error_numb_nodes(numb_nodes)
    ce.error_c(c, numb_nodes)
    ce.error_generator(law)

    G = nx.Graph()
    lst_node_labels = [f"S{i}" for i in range(1, numb_nodes + 1)]
    G.add_nodes_from(lst_node_labels)

    weights_matrix = np.zeros((numb_nodes, numb_nodes), dtype=np.float64)

    for i in range(numb_nodes):
        for j in range(i + 1, numb_nodes):
            B = ic.generate_nodes_type_3(numb_nodes, c, law)

            if B == 1:
                weights_matrix[i, j] = 1
                weights_matrix[j, i] = 1
                G.add_edge(lst_node_labels[i], lst_node_labels[j], weight=1)

    return (G, weights_matrix)


def generate_graph_type_4(numb_nodes: int, numb_connections: int, law: np.random.Generator) -> tuple[nx.Graph, NDArray[np.float64]]:
    ce.error_numb_nodes(numb_nodes)
    ce.error_numb_connections(numb_connections, numb_nodes)
    ce.error_generator(law)

    G = nx.Graph()
    lst_node_labels = [f"S{i}" for i in range(1, numb_nodes + 1)]
    G.add_nodes_from(lst_node_labels)

    weights_matrix = np.zeros((numb_nodes, numb_nodes), dtype=np.float64)

    for i in range(1, numb_nodes):
        if i <= numb_connections:
            lst_connections = list(range(i))
        else:
            degrees = np.array([G.degree(lst_node_labels[j]) for j in range(i)], dtype=np.float64)
            probabilities = degrees / degrees.sum()
            lst_connections = law.choice(i, size=numb_connections, replace=False, p=probabilities)

        for j in lst_connections:
            weights_matrix[i, j] = 1.0
            weights_matrix[j, i] = 1.0
            G.add_edge(lst_node_labels[i], lst_node_labels[j], weight=1.0)

    return (G, weights_matrix)
