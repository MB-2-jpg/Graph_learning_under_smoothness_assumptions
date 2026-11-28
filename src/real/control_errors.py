import numpy as np

VALID_DATASETS = ("USPS", "MNIST", "COIL20")


def error_dataset(dataset: str) -> None:
    if dataset not in VALID_DATASETS:
        raise ValueError(f"dataset must be one of {VALID_DATASETS} !")


def error_numb_nodes(numb_nodes: int) -> None:
    if not np.isfinite(numb_nodes) or numb_nodes < 3:
        raise ValueError("numb_nodes takes values on [3;+inf[")


def error_density(density: float) -> None:
    if not np.isfinite(density) or density <= 0:
        raise ValueError("density (average edges per node) takes values on R+* !")


def error_label_rate(label_rate: float) -> None:
    if not np.isfinite(label_rate) or label_rate <= 0 or label_rate >= 1:
        raise ValueError("label_rate takes values on ]0;1[ !")


def error_numb_neighbours(k: int) -> None:
    if not np.isfinite(k) or k < 1:
        raise ValueError("k (number of neighbours) takes values on [1;+inf[")


def error_model(model: str) -> None:
    if model not in ("log", "dong"):
        raise ValueError("model is 'log' (Algorithm 1) or 'dong' (Algorithm 2) !")


def error_raw_file_missing(path, dataset: str, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset}: missing raw file {path}.\n"
            f"Expected under data/reel_graphs/{dataset}/data_raw/. {hint}"
        )
