
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

import build_artificial_graphs as bag
import build_artificial_signals as bas
import filters as fil


def prepare_data(law: np.random.Generator, numb_signals: int = 1000) -> None:

    project_root = Path(__file__).resolve().parents[2]

    signals_directory = project_root / "data" / "artificial" / "signals"
    signals_directory.mkdir(parents=True, exist_ok=True)

    graph_1, _, weights_matrix_1 = bag.generate_graph_type_1(
        weight_threshold=0.6,
        sigma=0.2,
        law=law
    )

    graph_2, _, weights_matrix_2 = bag.generate_graph_type_2(
        alpha=0.5,
        sigma=0.2,
        law=law
    )

    graph_3, weights_matrix_3 = bag.generate_graph_type_3(
        numb_nodes=100,
        law=law
    )

    graph_4, weights_matrix_4 = bag.generate_graph_type_4(
        numb_nodes=100,
        law=law
    )

    graphs = {
        "type_1": (graph_1, weights_matrix_1),
        "type_2": (graph_2, weights_matrix_2),
        "type_3": (graph_3, weights_matrix_3),
        "type_4": (graph_4, weights_matrix_4)
    }

    filters = {
        "tikhonov": fil.h_tikhonov(),
        "generative": fil.h_generative(),
        "heat": fil.h_heat()
    }

    for graph_name, (graph, weights_matrix) in graphs.items():

        graph_weights = nx.to_numpy_array(
            graph,
            nodelist=list(graph.nodes()),
            weight="weight"
        )

        if not np.allclose(weights_matrix, graph_weights):
            raise ValueError(f"Matrice de poids incohérente pour {graph_name}")

        for filter_name, filter_function in filters.items():

            signals = bas.smooth_signals(
                weights_matrix=weights_matrix,
                numb_signals=numb_signals,
                filter_function=filter_function,
                law=law,
                noise_level=0.1
            )

            dataframe = pd.DataFrame(
                signals,
                index=[f"S{i+1}" for i in range(signals.shape[0])],
                columns=[f"signal_{j+1}" for j in range(numb_signals)]
            )

            dataframe.index.name = "node"

            dataframe.to_csv(signals_directory / f"{graph_name}_{filter_name}.csv")


if __name__ == "__main__":

    law = np.random.default_rng(42)

    prepare_data(law=law)
