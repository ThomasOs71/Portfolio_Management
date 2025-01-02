# -*- coding: utf-8 -*-

import numpy as np

from min_rel_entropy_sp import min_rel_entropy_sp


def twist_prob_mom_match(x, m_bar, s2_bar=None, p=None):
    """This function twists probabilities to match specified moments using relative entropy minimization

    Parameters
    ----------
        x : array, shape (j_bar, n_bar) if n_bar>1 or (j_bar,) for n_bar=1
        m_bar : array, shape (n_bar,)
        s2_bar : array, shape (n_bar, n_bar), optional
        p : array, shape (j_bar,), optional

    Returns
    -------
        p_bar : array, shape (j_bar,)
    """

    if np.ndim(m_bar) == 0:
        m_bar = np.reshape(m_bar, 1).copy()
    else:
        m_bar = np.array(m_bar).copy()
    if s2_bar is not None:
        if np.ndim(s2_bar) == 0:
            s2_bar = np.reshape(s2_bar, (1, 1))
        else:
            s2_bar = np.array(s2_bar).copy()

    if len(x.shape) == 1:
        x = x.reshape(-1, 1).copy()
    j_bar, n_bar = x.shape
    if p is None:
        p = np.ones(j_bar) / j_bar
    
    if n_bar + (n_bar*(n_bar + 1))/2 > j_bar:
        print('Error!')

    # step 1: compute the equality constraints
    z_eq = x.T.copy()
    mu_view_eq = m_bar.copy()
    if s2_bar is not None:
        s2_bar = np.array(s2_bar).copy()
        s2 = s2_bar + np.outer(m_bar, m_bar)
        x_t = x.T.copy()
        for n in range(n_bar):
            z_eq = np.vstack((z_eq, x_t[n:, :]*x_t[n]))
            mu_view_eq = np.r_[mu_view_eq, s2[n, n:]]

    # step 2: minimize the relative entropy
    p_bar = min_rel_entropy_sp(p, z_eq=z_eq, mu_view_eq=mu_view_eq)

    return np.squeeze(p_bar)
