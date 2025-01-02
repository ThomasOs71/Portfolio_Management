# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:12:29 2024

@author: Thomas
"""

# -*- coding: utf-8 -*-
import numpy as np
from crisp_fp import crisp_fp
from min_rel_entropy_sp import min_rel_entropy_sp


def conditional_fp(z, z_star, alpha, p_prior):
    """This function calculates conditional flexible probabilities 
    based on a given dataset and a reference set of values. 

    Parameters
    ----------
        z : array, shape (t_bar, )
        z_star : array, shape (k_bar, )
        alpha : scalar
        p_prior : array, shape (t_bar, )

    Returns
    -------
        p : array, shape (t_bar, k_bar) if k_bar>1 or (t_bar,) for k_bar=1

    """

    z_star = np.atleast_1d(z_star)

    t_bar = z.shape[0]
    k_bar = z_star.shape[0]

    # compute crisp probabilities
    p_crisp, _, _ = crisp_fp(z, z_star, alpha)
    p_crisp = p_crisp.T
    p_crisp[p_crisp == 0] = 10**-20
    
    if k_bar == 1:
        p_crisp = p_crisp/np.sum(p_crisp)
        p = np.zeros((k_bar, t_bar))
        # moments
        m_z = p_crisp@z
        s2_z = p_crisp@(z**2) - m_z**2
        # constraints
        a_ineq = np.atleast_2d(z**2)
        b_ineq = np.atleast_1d((m_z**2) + s2_z)
        a_eq = np.array([z])
        b_eq = np.array([m_z])
        # output
        p = min_rel_entropy_sp(p_prior, a_ineq, b_ineq, a_eq, b_eq)
    else:
        for k in range(k_bar):
            p_crisp[k, :] = p_crisp[k, :]/np.sum(p_crisp[k, :])

        # compute conditional flexible probabilities
        p = np.zeros((k_bar, t_bar))
        for k in range(k_bar):
            # moments
            m_z = p_crisp[k, :]@z
            s2_z = p_crisp[k, :]@(z**2) - m_z**2
            # constraints
            a_ineq = np.atleast_2d(z**2)
            b_ineq = np.atleast_1d((m_z**2) + s2_z)
            a_eq = np.array([z])
            b_eq = np.array([m_z])
            # output
            p[k, :] = min_rel_entropy_sp(p_prior, a_ineq, b_ineq, a_eq, b_eq)

    return np.squeeze(p.T)
