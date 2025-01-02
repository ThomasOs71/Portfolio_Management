#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import expm
from min_rel_entropy_sp import min_rel_entropy_sp


def fit_trans_matrix_credit(dates, n_oblig, n_cum, tau_hl=None):
    
    """This function estimates the transition matrix for credit risk, 
    utilizing observed data on the number of obligors and cumulative transitions over time, 
    and optionally incorporating a half-life parameter to adjust the decay rate of past transitions, 
    while minimizing relative entropy subject to probability and monotonicity constraints.

    Parameters
    ----------
        dates : array, shape(t_bar,)
        n_oblig : array, shape (t_bar, c_bar)
        n_cum : array, shape (t_bar, c_bar, c_bar)
        tau_hl : scalar, optional

    Returns
    -------
        p : array, shape (c_bar, c_bar)

    """
    
    t_bar = len(dates)
    c_bar = n_oblig[-1].shape[0]
    delta_t = np.zeros(t_bar - 1)

    num = np.zeros((c_bar, c_bar))
    den = np.zeros((c_bar, c_bar))
    g = np.zeros((c_bar, c_bar))
    
    # step 1: compute number of transitions at each time t
    m_num = np.zeros(n_cum.shape)
    m_num[0, :, :] = n_cum[0, :, :]
    m_num[1:, :, :] = np.diff(n_cum, axis=0)  # number of transitions at each time t≤t
    
    # step 2: esimate prior transition matrix
    for i in range(c_bar):
        for j in range(c_bar):
            if i != j:
                if tau_hl is None:
                    num[i, j] = n_cum[-1, i, j]
                    for t in range(1, t_bar):
                        den[i, j] = den[i, j] + n_oblig[t, i]*(np.busday_count(dates[t-1], dates[t]))/252
                    g[i, j] = num[i, j]/den[i, j]  # off-diagonal elements of g given tau_hl=None
                else:
                    for t in range(t_bar):
                        num[i, j] = num[i, j] + m_num[t, i, j]*np.exp(-(np.log(2)/tau_hl)*(np.busday_count(dates[t], dates[-1]))/252)
                    for t in range(1, t_bar):
                        den[i, j] = den[i, j] + n_oblig[t-1, i]*(np.exp(-(np.log(2)/tau_hl)*(np.busday_count(dates[t], dates[-1]))/252)
                                                                -np.exp(-(np.log(2)/tau_hl)*(np.busday_count(dates[t-1], dates[-1]))/252))
                    g[i, j] = (np.log(2)/tau_hl)*num[i, j]/den[i, j]  # off-diagonal elements of g given tau_hl

    for i in range(c_bar):
        g[i, i] = -np.sum(g[i, :])  # diagonal elements of g
    
    p_bar = expm(g)  # prior transition matrix
    
    # step 3: minimize relative entropy
    # probability constraint
    a_eq = np.ones((1, c_bar))  # 1×c_bar dimensional vector of ones
    b_eq = np.array([1])

    # initialize monotonicity constraint
    a_ineq = {}
    a_ineq[0] = np.diagflat(np.ones((1, c_bar-1)), 1) -np.diagflat(np.ones((1, c_bar)), 0)  # (c_bar-1)×c_bar upper triangular matrix
    a_ineq[0] = a_ineq[0][:-1]
    b_ineq = np.zeros((c_bar - 1))  # 1×(c_bar-1) dimensional vector of ones
    
    p = np.zeros((c_bar - 1, c_bar))
    for c in range(c_bar - 1):
        # minimize relative entropy
        p[c, :] = min_rel_entropy_sp(p_bar[c, :], a_ineq[c], b_ineq, a_eq, b_eq, False)
        # update monotonicity constraint
        a_temp = a_ineq.get(c).copy()
        a_temp[c, :] = -a_temp[c, :]
        a_ineq[c + 1] = a_temp.copy()
    
    p = np.r_[p, np.array([np.r_[np.zeros(c_bar - 1), 1]])]  # default constraint

    return p
