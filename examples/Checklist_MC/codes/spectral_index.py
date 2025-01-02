# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 14:56:31 2024

@author: Thomas
"""

# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import quad


def spectral_index(spectr, pi, p, h_tilde):
    """This function computes the satisfaction spectral measure and its gradient, 
    utilizing a specified spectral function to compute the weights of the spectral measure.

    Parameters
    ----------
        spectr : function
        pi : array, shape (j_bar, n_bar)
        p : array,  shape (j_bar,)
        h_tilde : array, shape (n_bar,)

    Returns
    -------
        satis_spectr : scalar
        satis_spectr_grad : scalar
    """

    h_tilde = np.array(h_tilde).reshape(-1)
    p = np.array(p).reshape(-1)

    j_bar = pi.shape[0]

    # compute ex-ante performance scenarios
    y = pi@h_tilde
    # sort ex-ante performance scenarios, probabilities and P&Ls
    sort_j = np.argsort(y)
    y_sort = y[sort_j]
    p_sort = p[sort_j]
    pi_sort = pi[sort_j, :]
    # compute cumulative sums of the ordered probabilities
    u_sort = np.append(0, np.cumsum(p_sort))

    # compute weights of spectral measure
    w = np.zeros(j_bar)
    for j in range(j_bar):
        w[j], _ = quad(spectr, u_sort[j], u_sort[j + 1])  
    w = w/np.sum(w)

    # compute spectral measure
    satis_spectr = y_sort@w

    # compute gradient of spectral measure
    satis_spectr_grad = pi_sort.T@w

    return satis_spectr, satis_spectr_grad
