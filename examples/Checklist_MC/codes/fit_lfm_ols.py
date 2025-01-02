#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW


def fit_lfm_ols(x_t, z_t, p_t=None, fit_intercept=True):
    
    """This function performs Ordinary Least Squares with Flexible Probabilities (OLSFP) 
    to estimate the parameters of a Linear Factor Model (LFM).

    Parameters
    ----------
        x_t : array, shape (t_bar, n_bar) if n_bar>1 or (t_bar, ) for n_bar=1
        z_t : array, shape (t_bar, k_bar) if k_bar>1 or (t_bar, ) for k_bar=1
        p_t : array, optional, shape (t_bar,)
        fit_intercept : bool

    Returns
    -------
        alpha_hat_olsfp : array, shape (n_bar,)
        beta_hat_olsfp : array, shape (n_bar, k_bar) if k_bar>1 or (n_bar, ) for k_bar=1
        s2_eps_hat_olsfp : array, shape (n_bar, n_bar)
        eps_t : array, shape (t_bar, n_bar) if n_bar>1 or (t_bar, ) for n_bar=1

    """
    
    t_bar = x_t.shape[0]

    if len(z_t.shape) < 2:
        z_t = z_t.reshape((t_bar, 1)).copy()
        k_bar = 1
    else:
        k_bar = z_t.shape[1]

    if len(x_t.shape) < 2:
        x_t = x_t.reshape((t_bar, 1)).copy()
        n_bar = 1
    else:
        n_bar = x_t.shape[1]

    if p_t is None:
        p_t = np.ones(t_bar)/t_bar

    # step 1: compute HFP mean and covariance of (X,Z)'
    if fit_intercept is True:
        m_xz_hat_hfp = DescrStatsW(np.c_[x_t, z_t], weights=p_t).mean
        s2_xz_hat_hfp = DescrStatsW(np.c_[x_t, z_t], weights=p_t).cov
    else:
        m_xz_hat_hfp = np.zeros(n_bar + k_bar)
        s2_xz_hat_hfp = p_t*np.c_[x_t, z_t].T@np.c_[x_t, z_t]

    # step 2: compute OLSFP estimates
    s2_z_hat_hfp = s2_xz_hat_hfp[n_bar:, n_bar:]
    s_x_z_hat_hfp = s2_xz_hat_hfp[:n_bar, n_bar:]
    m_xz_hat_hfp = m_xz_hat_hfp.reshape(-1)
    m_z_hat_hfp = m_xz_hat_hfp[n_bar:].reshape(-1, 1)
    m_x_hat_hfp = m_xz_hat_hfp[:n_bar].reshape(-1, 1)

    beta_hat_olsfp = s_x_z_hat_hfp@np.linalg.inv(s2_z_hat_hfp)
    alpha_hat_olsfp = m_x_hat_hfp - beta_hat_olsfp@m_z_hat_hfp

    # step 3: compute residuals and OLSFP estimate of covariance of U
    eps_t = (x_t.T - alpha_hat_olsfp - beta_hat_olsfp@z_t.T).T
    s2_eps_hat_olsfp = DescrStatsW(eps_t, weights=p_t).cov

    return alpha_hat_olsfp[:, 0], np.squeeze(beta_hat_olsfp),\
        np.squeeze(s2_eps_hat_olsfp), np.squeeze(eps_t)
