from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

import control_errors as ce
import interim_calcul as ic
import graph_learning as gl

#: section 6.2: round(2.6 * i^2) images of class i, i = 1..10 (10 is the digit 0)
USPS_CLASSES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 0)
USPS_SIZES: Tuple[int, ...] = tuple(round(2.6 * i ** 2) for i in range(1, 11))

#: section 6.2, MNIST "1" vs "2": the three proportions of figure 2
MNIST_PROPORTIONS: Tuple[Tuple[int, int], ...] = ((40, 160), (100, 100), (160, 40))

#: appendix C.1: 20 classes, sizes growing geometrically from 3 to 60
COIL20_SIZES: NDArray[np.float64] = 3.0 * (60.0 / 3.0) ** (np.arange(20) / 19.0)


def normalized_dist_vector(signals_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    dist_vector = gl.pairwise_sq_dists(signals_matrix)
    mean = float(dist_vector.mean())
    return dist_vector / mean if mean > 0 else dist_vector


def build_usps_sample(law: np.random.Generator
                      ) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    signals_matrix, labels = ic.load_dataset("USPS")
    sampled_signals, sampled_labels = ic.subsample_classes(
        signals_matrix, labels, USPS_CLASSES, USPS_SIZES, law)
    return sampled_signals, sampled_labels, normalized_dist_vector(sampled_signals)


def build_mnist_sample(numb_ones: int, numb_twos: int, law: np.random.Generator
                       ) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    ce.error_numb_nodes(numb_ones + numb_twos)
    signals_matrix, labels = ic.load_dataset("MNIST")
    signals_matrix = signals_matrix / 255.0
    sampled_signals, sampled_labels = ic.subsample_classes(
        signals_matrix, labels, (1, 2), (numb_ones, numb_twos), law)
    return sampled_signals, sampled_labels, normalized_dist_vector(sampled_signals)


def build_coil20_sample(law: np.random.Generator, sizes: Optional[Sequence[int]] = None
                        ) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    signals_matrix, labels = ic.load_dataset("COIL20")
    classes = np.unique(labels)
    if sizes is None:
        sizes = [int(round(s)) for s in COIL20_SIZES[:classes.size]]
    sampled_signals, sampled_labels = ic.subsample_classes(
        signals_matrix, labels, classes, sizes, law)
    return sampled_signals, sampled_labels, normalized_dist_vector(sampled_signals)


def subsample_nodes(signals_matrix: NDArray[np.float64], labels: NDArray[np.int64],
                    max_nodes: int, law: np.random.Generator
                    ) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    numb_nodes = signals_matrix.shape[0]
    if max_nodes is None or numb_nodes <= max_nodes:
        return signals_matrix, labels, normalized_dist_vector(signals_matrix)

    keep = []
    for class_label in np.unique(labels):
        pool = np.flatnonzero(labels == class_label)
        take = max(1, int(round(pool.size * max_nodes / numb_nodes)))
        keep.append(law.choice(pool, size=min(take, pool.size), replace=False))
    index = np.sort(np.concatenate(keep))
    reduced = signals_matrix[index]
    return reduced, labels[index], normalized_dist_vector(reduced)


def build_sample(dataset: str, law: np.random.Generator, **kwargs
                 ) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    ce.error_dataset(dataset)
    if dataset == "USPS":
        return build_usps_sample(law)
    if dataset == "MNIST":
        numb_ones, numb_twos = kwargs.pop("proportion", MNIST_PROPORTIONS[1])
        return build_mnist_sample(numb_ones, numb_twos, law)
    return build_coil20_sample(law, **kwargs)
