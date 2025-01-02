#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate
from scipy.stats import norm


def implvol_delta2m_moneyness(sigma_delta, tau, delta_moneyness, y, tau_y, l_bar):
    """This function converts delta-moneyness into m-moneyness, constructs a grid for m-moneyness, 
    and calculates the implied volatility surface in m-moneyness space.

    Parameters
    ----------
        sigma_delta : array, shape (t_bar, k_bar, n_bar)
        tau : array, shape (k_bar,)
        delta_moneyness : array, shape (n_bar,)
        y : array, shape (t_bar, d_bar)
        tau_y : array, shape (d_bar,)
        l_bar : scalar

    Returns
    -------
       sigma_m : array, shape (t_bar, j_, l_bar)
       m_moneyness : array, shape (l_bar,)
    """
    
    # convert delta-moneyness into m-moneyness
    t_bar = sigma_delta.shape[0]
    k_bar = len(tau)
    n_bar = len(delta_moneyness)
    y_grid_t = np.zeros((t_bar, k_bar, n_bar))
    m_data = np.zeros((t_bar, k_bar, n_bar))
    tau_y = np.atleast_1d(tau_y)
    for t in range(t_bar):
        if tau_y.shape[0] == 1:
            y_grid_t[t, :, :] = np.tile(np.atleast_2d(y).T, (1, n_bar))
        else:
            y_grid_tmp = interpolate.interp1d(tau_y, y[t, :])
            y_grid_t[t, :, :] = np.tile(np.atleast_2d(y_grid_tmp(tau)).T, (1, n_bar))
        m_data[t, :, :] = norm.ppf(delta_moneyness)*sigma_delta[t, :, :] -\
                         ((y_grid_t[t, :, :] + sigma_delta[t, :, :]**2/2).T*np.sqrt(tau)).T

    # construct m_moneyness grid
    min_m = np.min(m_data)  # min m-moneyness
    max_m = np.max(m_data)  # max m-moneyness
    # equally-spaced grid between minimal and maximal m-moneyness
    m_moneyness = min_m + (max_m - min_m)*np.arange(l_bar)/(l_bar - 1)

    # implied volatility surface in m-moneyness
    sigma_m = np.zeros((t_bar, k_bar, l_bar))
    for t in range(t_bar):
        for k in range(k_bar):
            poly_coef = np.polyfit(m_data[t, k, :], sigma_delta[t, k, :], 2)
            polyf = np.poly1d(poly_coef)
            sigma_m[t, k, :] = polyf(m_moneyness.flatten())

    return sigma_m, np.squeeze(m_moneyness)
