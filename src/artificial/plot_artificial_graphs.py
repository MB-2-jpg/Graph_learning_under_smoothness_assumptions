import networkx as nx
import numpy as np
import interim_calcul as ic
import build_artificial_graphs as bag


def display_graph_type_1(law: np.random.Generator, node_size: int, font_size: int, weight_threshold: float, sigma: float, file_name: str) -> None: 
    G = bag.generate_graph_type_1(weight_threshold, sigma, law)
    lst_nodes = list(G[0].nodes())
    signals_matrix = G[1]
    pos = dict(zip(lst_nodes, signals_matrix))
    ic.display_graph_struct_interim(G[0], pos, font_size, node_size, file_name)


def display_graph_type_2(law: np.random.Generator, node_size: int, font_size: int, alpha: float, sigma: float, file_name: str) -> None:
    G = bag.generate_graph_type_2(alpha, sigma, law)
    lst_nodes = list(G[0].nodes())
    signals_matrix = G[1]
    pos = dict(zip(lst_nodes, signals_matrix))
    ic.display_graph_struct_interim(G[0], pos, font_size, node_size, file_name)
    

def display_graph_type_3(law: np.random.Generator, node_size: int, font_size: int, numb_nodes: int, file_name: str) -> None:
    G = bag.generate_graph_type_3(numb_nodes, law)
    pos = nx.spring_layout(G[0], seed=42)
    ic.display_graph_struct_interim(G[0], pos, font_size, node_size, file_name)

