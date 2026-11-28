from pathlib import Path
import json
import numpy as np
import pandas as pd
import networkx as nx
from numpy.typing import NDArray

from src.artificial import build_artificial_graphs as bag
from src.artificial import build_artificial_signals as bas
from src.artificial import control_errors as ce
from src.artificial import filters as fil

project_root = Path(__file__).resolve().parents[2]

artificial_directory = project_root / "data" / "artificial"
artificial_directory.mkdir(parents=True, exist_ok=True)


def create_dataset_directory(raw: bool = False) -> Path:

    if raw:
        dataset_directory = artificial_directory / "raw"

    else:
        dataset_numbers = []

        for directory in artificial_directory.iterdir():
            if directory.is_dir() and directory.name.startswith("dataset_"):
                number = directory.name.removeprefix("dataset_")

                if number.isdigit():
                    dataset_numbers.append(int(number))

        dataset_number = max(dataset_numbers, default=0) + 1

        dataset_directory = artificial_directory / f"dataset_{dataset_number}"

    dataset_directory.mkdir(exist_ok=False)

    (dataset_directory / "weights").mkdir()
    (dataset_directory / "signals").mkdir()

    return dataset_directory


def save_weights_matrix(
    weights_matrix: NDArray[np.float64],
    file_name: str,
    dataset_directory: Path
) -> None:

    ce.error_file_name(file_name)

    m = weights_matrix.shape[0]
    labels = [f"S{i+1}" for i in range(m)]

    dataframe = pd.DataFrame(
        weights_matrix,
        index=labels,
        columns=labels
    )

    dataframe.to_csv(
        dataset_directory / "weights" / f"{file_name}.csv"
    )


def save_signals_matrix(
    signals_matrix: NDArray[np.float64],
    file_name: str,
    dataset_directory: Path
) -> None:

    ce.error_file_name(file_name)

    m, n = signals_matrix.shape

    row_labels = [f"S{i+1}" for i in range(m)]
    column_labels = [f"signal_{j+1}" for j in range(n)]

    dataframe = pd.DataFrame(
        signals_matrix,
        index=row_labels,
        columns=column_labels
    )

    dataframe.to_csv(
        dataset_directory / "signals" / f"{file_name}.csv"
    )


def save_dataset_parameters(
    parameters: dict,
    dataset_directory: Path
) -> None:

    with open(
        dataset_directory / "parameters.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(parameters, file, indent=4)


def prepare_data(parameters: dict, raw: bool = False) -> Path:

    graph_parameters = parameters["graphs"]
    filter_parameters = parameters["filters"]
    signal_parameters = parameters["signals_generation"]

    numb_nodes = graph_parameters["numb_nodes"]
    numb_signals = signal_parameters["numb_signals"]
    noise_level = signal_parameters["noise_level"]
    numb_repetitions = parameters["numb_repetitions"]

    graph_seed = graph_parameters["seed"]
    signal_seed = signal_parameters["seed"]

    filter_functions = {
        "tikhonov": fil.h_tikhonov(
            filter_parameters["tikhonov"]["a"]
        ),

        "generative": fil.h_generative(),

        "heat": fil.h_heat(
            filter_parameters["heat"]["a"]
        )
    }

    graph_seeds = np.random.SeedSequence(
        graph_seed,
        spawn_key=(0,)
    ).spawn(numb_repetitions)

    signal_seeds = np.random.SeedSequence(
        signal_seed,
        spawn_key=(1,)
    ).spawn(numb_repetitions)

    dataset_directory = create_dataset_directory(raw=raw)

    save_dataset_parameters(parameters, dataset_directory)

    for repetition in range(numb_repetitions):

        graph_laws = [
            np.random.default_rng(seed)
            for seed in graph_seeds[repetition].spawn(4)
        ]

        current_signal_seeds = signal_seeds[repetition].spawn(4)

        graph_1, _, weights_matrix_1 = bag.generate_graph_type_1(
            numb_nodes=numb_nodes,
            weight_threshold=graph_parameters["type_1"]["weight_threshold"],
            sigma=graph_parameters["type_1"]["sigma"],
            law=graph_laws[0]
        )

        graph_2, _, weights_matrix_2 = bag.generate_graph_type_2(
            numb_nodes=numb_nodes,
            alpha=graph_parameters["type_2"]["alpha"],
            sigma=graph_parameters["type_2"]["sigma"],
            law=graph_laws[1]
        )

        graph_3, weights_matrix_3 = bag.generate_graph_type_3(
            numb_nodes=numb_nodes,
            c=graph_parameters["type_3"]["c"],
            law=graph_laws[2]
        )

        graph_4, weights_matrix_4 = bag.generate_graph_type_4(
            numb_nodes=numb_nodes,
            numb_connections=graph_parameters["type_4"]["numb_connections"],
            law=graph_laws[3]
        )

        graphs = [
            (graph_1, weights_matrix_1),
            (graph_2, weights_matrix_2),
            (graph_3, weights_matrix_3),
            (graph_4, weights_matrix_4)
        ]

        for i, (graph, weights_matrix) in enumerate(graphs):

            graph_weights = nx.to_numpy_array(
                graph,
                nodelist=list(graph.nodes()),
                weight="weight"
            )

            ce.error_graph_weights_compatibility(
                weights_matrix,
                graph_weights
            )

            file_name = f"type_{i+1}_rep_{repetition+1}"

            save_weights_matrix(
                weights_matrix,
                file_name,
                dataset_directory
            )

            for filter_name, filter_function in filter_functions.items():

                law = np.random.default_rng(
                    current_signal_seeds[i]
                )

                signals_matrix = bas.smooth_signals(
                    weights_matrix=weights_matrix,
                    numb_signals=numb_signals,
                    filter_function=filter_function,
                    law=law,
                    noise_level=noise_level
                )

                save_signals_matrix(
                    signals_matrix,
                    f"{file_name}_{filter_name}",
                    dataset_directory
                )

    return dataset_directory
