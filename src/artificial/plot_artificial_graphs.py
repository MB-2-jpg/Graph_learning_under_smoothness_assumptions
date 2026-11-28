import networkx as nx
import numpy as np

from src.common import interim_calcul as ic
from src.artificial import build_artificial_graphs as bag


def display_graph_type_1(law: np.random.Generator, node_size: int, font_size: int, numb_nodes: int, weight_threshold: float, sigma: float, file_name: str) -> None:

    G = bag.generate_graph_type_1(numb_nodes, weight_threshold, sigma, law)

    lst_nodes = list(G[0].nodes())
    signals_matrix = G[1]
    pos = dict(zip(lst_nodes, signals_matrix))

    ic.display_graph_struct_interim(G[0], pos, font_size, node_size, file_name)


def display_graph_type_2(law: np.random.Generator, node_size: int, font_size: int, numb_nodes: int, alpha: float, sigma: float, file_name: str) -> None:

    G = bag.generate_graph_type_2(numb_nodes, alpha, sigma, law)

    lst_nodes = list(G[0].nodes())
    signals_matrix = G[1]
    pos = dict(zip(lst_nodes, signals_matrix))

    ic.display_graph_struct_interim(G[0], pos, font_size, node_size, file_name)


def display_graph_type_3(law: np.random.Generator, node_size: int, font_size: int, numb_nodes: int, c: float, file_name: str) -> None:

    G = bag.generate_graph_type_3(numb_nodes, c, law)

    pos = nx.spring_layout(G[0], seed=42)

    ic.display_graph_struct_interim(G[0], pos, font_size, node_size, file_name)


def display_graph_type_4(law: np.random.Generator, node_size: int, font_size: int, numb_nodes: int, numb_connections: int, file_name: str) -> None:

    G = bag.generate_graph_type_4(numb_nodes, numb_connections, law)

    pos = nx.spring_layout(G[0], seed=42)

    ic.display_graph_struct_interim(G[0], pos, font_size, node_size, file_name)