#!/usr/bin/env python3

import numpy as np
from scipy import stats


def simulate_markov_chain_multiv(x_tnow, p, m_bar, *, rho2=None, nu=None, j_bar=1000):
    """This function simulates a Markov chain with multivariate transitions, generating future scenarios 
       based on specified transition probabilities and copula scenarios

    Parameters
    ----------
        x_tnow : array, shape(d_bar, )
        p : array, shape(s_bar, s_bar)
        deltat_m : int
        rho2 : array, shape(d_bar, d_bar)
        nu : int
        j_bar : int

    Returns
    -------
        x_tnow_thor : array, shape(j_bar, m_bar + 1, d_bar)

    """

    d_bar = x_tnow.shape[0]

    # uncorrelated marginal transitions
    if rho2 is None:
        rho2 = np.eye(d_bar)
    # normal copula
    if nu is None:
        nu = 10**9  

    # copula scenarios and their grades
    x_tnow_thor = np.zeros((j_bar, m_bar + 1, d_bar))
    x_tnow_thor[:, 0, :] = x_tnow
    for m in np.arange(m_bar):
        # scenarios from a t copula
        u = stats.multivariate_t.rvs(np.zeros(d_bar), rho2, nu, j_bar)
        # grades
        eps = stats.t.cdf(u, nu)

        # projected path
        for j in np.arange(j_bar):
            for d in np.arange(d_bar):
                # thresholds for quantile
                f = np.r_[0, np.cumsum(p[int(x_tnow_thor[j, m, d]) - 1, :])]
                # state
                x_tnow_thor[j, m + 1, d] = np.sum(f <= eps[j, d])

    return x_tnow_thor
