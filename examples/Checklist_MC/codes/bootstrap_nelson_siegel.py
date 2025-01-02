#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import newton
from collections import defaultdict
from fit_nelson_siegel_bonds import fit_nelson_siegel_bonds


def bootstrap_nelson_siegel(v_bond, dates, c, tend, freq_paym=0.5, facev=1.0):
    """This function implements the Nelson-Siegel bootstrap method to derive the parameters of 
    the Nelson-Siegel yield curve model at various time points. It computes the time to maturity, 
    coupon payments, and yields for each bond, fits the Nelson-Siegel model using bond values and parameters, 
    and returns the Nelson-Siegel parameters along with associated yield curve information.

    Parameters
    ----------
        v_bond : array, shape (t_bar, n_bar)
        dates : array, shape (t_bar, )
        c : array, shape (n_bar,)
        tend : array, shape (n,)
        freq_paym : scalar
        tau : array, optional, shape (l_bar)
        tau_ref : array, optional, shape (m_bar)
        y_ref : array, optional, shape (t_bar, m_bar)

    Returns
    ----------
        theta : array, shape (t_bar, 4)
        y_tau : array, shape (t_bar, l_bar)
        y_ref_tau : array, shape (t_bar, l_bar)
        s_tau : array, shape (t_bar, l_bar)
    """

    n_bar = v_bond.shape[1]
    t_bar = len(dates)

    theta = np.zeros((t_bar, 4))
    for t in range(t_bar):
        tau_real = np.zeros(n_bar)
        c_k = defaultdict(dict)
        y = defaultdict(dict)
        for n in range(n_bar):
            # compute time to maturity 
            tau_real[n] = np.busday_count(dates[t], tend[n])/252  # time from dates[t] to the maturity (in years)

            # compute the number of coupon payments, time to coupon payments, and coupons
            y[n] = np.flip(np.arange(tau_real[n], 0.0001, -freq_paym), 0)  # time to coupon payments
            k_n = len(y[n])  # number of coupon payments from dates[t] to maturity
            c_k[n] = c[n]*np.ones(k_n)*freq_paym  # every bond has 1/freq_paym payments per year
            c_k[n][-1] = c_k[n][-1] + 1  # include notional

        # fit NS model
        if t == 0:
            theta[t] = fit_nelson_siegel_bonds(v_bond[t, :], c_k, y, facev=facev)
        else:
            theta[t] = fit_nelson_siegel_bonds(v_bond[t, :], c_k, y, theta_0=theta[t - 1], facev=facev)

    return theta
