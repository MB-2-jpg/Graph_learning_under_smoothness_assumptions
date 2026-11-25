import numpy as np

def error_alpha(alpha: float) -> None:
    if not np.isfinite(alpha) or alpha <= -1/5 or alpha == 0:
        raise ValueError("alpha takes values on ]-1/5;0[U]0;+inf[ !")


def error_sigma(sigma: float) -> None:
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("it takes values on R+* !")


def error_numb_nodes(numb_nodes: int) -> None:
    if not np.isfinite(numb_nodes) or numb_nodes < 3:
        raise ValueError("numb_nodes takes values on [3;+inf[")
