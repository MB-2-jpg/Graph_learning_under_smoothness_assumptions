import numpy as np
from numpy.typing import NDArray
from src.artificial import interim_calcul as ic


def algo_baseline(signals_matrix: NDArray[np.float64] , sigma: float, seuil=None) -> NDArray[np.float64]: 
    weigths_matrix = ic.gaussian_weights(signals_matrix, sigma)

    if seuil is not None:
        weigths_matrix[weigths_matrix<seuil] = 0 

    return weigths_matrix


def algo_Dong_Al(dist_vector: NDArray[np.float64], alpha: float, s: float, weight_init: NDArray[np.float64], c_init, gamma: float, epsilon: float) -> NDArray[np.float64]:
    m = 100
    S = np.zeros((m, (m-1)*m//2))
    c = 0
    for i in range(m):
        for j in range(i+1, m):
            S[i, c] = 1
            S[j, c] = 1
            c += 1
            
    w = weight_init
    c = c_init
    i_max = 100000
    for i in range(1, (i_max+1)):
        w_prec = w
        c_prec = c
        y = w - gamma*(2*alpha*(2*w_prec + S.T@(S@w_prec)) + 2*c_prec)
        y_ = c_prec + gamma * 2 * sum(w_prec)
        p = np.maximum(0, y-2*gamma*dist_vector)
        p_ = y_ - gamma*s
        q = p - gamma*(2*alpha*(2*p + S.T@(S@p)) + 2*p_)
        q_ = p_ + gamma * 2 * sum(p)
        w = w_prec - y + q
        c = c_prec - y_ + q_

        if( (np.linalg.norm(w - w_prec) / np.linalg.norm(w_prec) < epsilon) and (np.abs(c - c_prec) / np.abs(c_prec) < epsilon) ):
            break

    return w
        