from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

__all__ = ["draw_graph", "draw_graph_xy", "draw_W", "plot_filters",
           "plot_signals", "circular_layout"]


def circular_layout(m: int) -> np.ndarray:
    theta = np.pi / 2 - 2 * np.pi * np.arange(m) / m
    return np.c_[np.cos(theta), np.sin(theta)]


def _edges(ax, w, P, cmap, rel_tol, lw_max=5.0, wmax=None):
    m = P.shape[0]
    i, j = np.triu_indices(m, k=1)
    wmax = (w.max() if w.max() > 0 else 1.0) if wmax is None else wmax
    keep = w > rel_tol * wmax
    segs = [[P[a], P[b]] for a, b, k in zip(i, j, keep) if k]
    vals = w[keep]
    if not segs:
        return None
    order = np.argsort(vals)                       # weak edges drawn first
    lc = LineCollection([segs[o] for o in order], cmap=cmap,
                        norm=plt.Normalize(0.0, wmax),
                        linewidths=0.6 + lw_max * vals[order] / wmax, zorder=1)
    lc.set_array(vals[order])
    ax.add_collection(lc)
    return lc


def draw_graph(ax, w, m: Optional[int] = None, cmap="plasma", title="",
               rel_tol=1e-6, labels=None):
    m = int((1 + np.sqrt(1 + 8 * w.size)) / 2) if m is None else m
    labels = (m <= 12) if labels is None else labels
    P = circular_layout(m)
    lc = _edges(ax, w, P, cmap, rel_tol)
    ax.scatter(*P.T, s=560 if labels else 120, c="white", edgecolors="k",
               linewidths=1.6, zorder=2)
    if labels:
        for k, (x, y) in enumerate(P):
            ax.text(x, y, rf"$x_{{{k+1}}}$", ha="center", va="center",
                    fontsize=12, zorder=3)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=10)
    return lc


def draw_graph_xy(ax, w, coords, cmap="plasma", title="", rel_tol=1e-6,
                  node_color=None, node_size=22, lw_max=3.0, wmax=None):
    coords = np.asarray(coords, float)
    lc = _edges(ax, w, coords, cmap, rel_tol, lw_max=lw_max, wmax=wmax)
    ax.scatter(*coords.T, s=node_size, c=("k" if node_color is None else node_color),
               zorder=2, cmap=None if node_color is None else "tab10")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=10)
    return lc


def draw_W(ax, W, cmap="plasma", title="learned $W$", labels=None):
    im = ax.imshow(W, cmap=cmap)
    ax.set_title(title, fontsize=10)
    if labels is not None:
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    return im


def plot_filters(ax, filters, lam=None, title="graph filters", ylim=(0, 0.4)):
    lam = np.linspace(0.02, 1, 400) if lam is None else lam
    for name, h in filters.items():
        ax.plot(lam, h(lam), lw=2, label=name)
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$h(\lambda)$")
    ax.set_title(title, fontsize=11)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)


def plot_signals(ax, X, k_max=6, n_max=100, title="", offset_scale=2.6):
    k_max = min(k_max, X.shape[0])
    off = offset_scale * np.std(X)
    for k in range(k_max):
        ax.plot(X[k, :n_max] - k * off, lw=0.9, color=plt.cm.tab10(k % 10))
        ax.text(-6, -k * off, rf"$x_{{{k+1}}}$", ha="right", va="center",
                fontsize=10, color=plt.cm.tab10(k % 10))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-14, n_max)
    for sp in ax.spines.values():
        sp.set_visible(False)
