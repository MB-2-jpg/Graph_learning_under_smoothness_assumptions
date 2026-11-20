from typing import Dict, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import graph_learning as gl
import interim_calcul as ic

def _maybe_show() -> None:
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "ps", "svg", "template"):
        plt.show()
    else:
        plt.close("all")


#: one colour per method, used in every figure
COLORS: Dict[str, str] = {"Ours": "tab:blue", "Dong etal": "tab:orange",
                          "k-NN": "tab:green"}

#: metric key -> axis label, in the order of figure 1
METRIC_LABELS: Dict[str, str] = {
    "clustering_error": "clustering error",
    "lp_miss": "label propagation miss rate",
    "n_components": "number of components",
    "n_disconnected_nodes": "isolated nodes",
    "lp_unclass": "unclassifiable nodes",
}


def _mean_curve(records: Sequence[dict], model: str, metric: str
                ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    by_param: Dict[float, list] = {}
    for record in records:
        if record["model"] == model and metric in record:
            by_param.setdefault(record["param"], []).append(record)
    if not by_param:
        return np.empty(0), np.empty(0)
    densities = np.array([np.mean([r["density"] for r in rs]) for rs in by_param.values()])
    values = np.array([np.mean([r[metric] for r in rs]) for rs in by_param.values()])
    order = np.argsort(densities)
    return densities[order], values[order]


def display_density_curves(records: Sequence[dict], metrics: Sequence[str],
                           title: str, file_name: str, xmax: float = 22.0,
                           models: Sequence[str] = ("Ours", "Dong etal", "k-NN")
                           ) -> None:
    figure, axes = plt.subplots(1, len(metrics), figsize=(5.7 * len(metrics), 4.4))
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, metrics):
        for model in models:
            densities, values = _mean_curve(records, model, metric)
            if densities.size:
                keep = densities <= xmax
                ax.plot(densities[keep], values[keep], "o-", ms=3.5, lw=1.5,
                        color=COLORS.get(model), label=model)
        ax.set_xlabel("edges per node")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    destination = ic.save_figure(figure, file_name)
    print(f"  wrote {destination}")
    _maybe_show()


def pick_at_density(records: Sequence[dict], model: str, target_density: float
                    ) -> Optional[dict]:
    candidates = [r for r in records if r["model"] == model]
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs(r["density"] - target_density))


def _graph_panels(store: Dict[Tuple[str, float], NDArray[np.float64]],
                  records: Sequence[dict], coords: NDArray[np.float64],
                  node_values: NDArray[np.float64], target_density: float,
                  title: str, file_name: str, cmap: str, colorbar_label: str,
                  models: Sequence[str], per_model_values=None) -> None:
    figure, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 6.2))
    axes = np.atleast_1d(axes)
    scatter = None

    for ax, model in zip(axes, models):
        record = pick_at_density(records, model, target_density)
        if record is None or (model, record["param"]) not in store:
            ax.set_axis_off()
            continue
        weights_vector = store[(model, record["param"])]
        values = (node_values if per_model_values is None
                  else per_model_values(weights_vector))
        scatter = ic.display_graph_scatter_interim(
            ax, weights_vector, coords, values, cmap=cmap,
            title=f"{model} -- {record['density']:.1f} edges/node, "
                  f"{record['n_components']} component(s)")

    figure.suptitle(title, fontsize=12)
    if scatter is not None and colorbar_label:
        figure.colorbar(scatter, ax=axes, fraction=0.016, label=colorbar_label)
    destination = ic.save_figure(figure, file_name)
    print(f"  wrote {destination}")
    _maybe_show()


def display_graphs_by_label(store, records: Sequence[dict],
                            coords: NDArray[np.float64], labels: NDArray[np.int64],
                            target_density: float, title: str, file_name: str,
                            models: Sequence[str] = ("Ours", "Dong etal", "k-NN")
                            ) -> None:
    _graph_panels(store, records, coords, labels, target_density, title, file_name,
                  cmap="tab10", colorbar_label="", models=models)


def display_graphs_by_degree(store, records: Sequence[dict],
                             coords: NDArray[np.float64], numb_nodes: int,
                             target_density: float, title: str, file_name: str,
                             models: Sequence[str] = ("Ours", "Dong etal", "k-NN")
                             ) -> None:
    _graph_panels(store, records, coords, None, target_density, title, file_name,
                  cmap="viridis", colorbar_label="weighted degree $(Sw)_i$",
                  models=models,
                  per_model_values=lambda w: gl.degrees(w, numb_nodes))
