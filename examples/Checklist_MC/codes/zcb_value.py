# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate


def zcb_value(t_hor, x_thor, tau, t_end, rd='y', facev=1, eta=0.013):
    """This function calculates the value of zero-coupon bonds (ZCBs) at a specified horizon date 
    handling different types of yield curves (yields, shadow rates, or Nelson-Siegel parameters) 
    and transforming shadow rates into yields when necessary.

    Parameters
    ----------
        t_hor : date
        x_thor : array, shape(j_bar, d_bar) if d_bar>1 or (t_bar,) for d_bar=1
        tau : array, shape(d_bar,)
        t_end : array of dates, shape(k_bar,)
        rd : string, optional
        facev : array, optional, shape(k_bar,)
        eta : scalar, optional

    Returns
    -------
        v : array, shape(j_bar, k_bar) if k_bar>1 or (t_bar,) for k_bar=1
    """

    j_bar = x_thor.shape[0]
    k_bar = t_end.shape[0]

    if isinstance(facev, int):
        facev = facev*np.ones(k_bar)

    # compute time to maturiy at the horizon for each zcb
    tau_star = np.array([np.busday_count(t_hor, t_end[i])/252 for i in range(k_bar)])
    tau_star = np.where(tau_star < 0, 0, tau_star)

    # compute yield for each time to maturity
    if rd == 'y':
        # risk drivers are yields
        interp = interpolate.interp1d(tau.flatten(), x_thor, axis=1, fill_value='extrapolate')
        x_star = interp(tau_star)
    elif rd == 'sr':
        # risk drivers are shadow rates
        interp = interpolate.interp1d(tau.flatten(), x_thor, axis=1, fill_value='extrapolate')
        # transform shadow rates to yields
        x_star = np.zeros((interp(tau_star).shape))
        x_star[interp(tau_star) >= eta] = interp(tau_star)[interp(tau_star) >= eta]
        x_star[interp(tau_star) < eta] = eta*np.exp(interp(tau_star)[interp(tau_star) < eta]/eta - 1)
    elif rd == 'ns':
        # risk drivers are NS parameters
        x_star = np.zeros((j_bar, k_bar))
        idx_nonzero = (tau_star > 0)
        for j in range(j_bar):
            x_star[j, idx_nonzero] = x_thor[j][0] - x_thor[j][1]*((1 - np.exp(-x_thor[j][3]*tau_star[idx_nonzero]))/
         (x_thor[j][3]*tau_star[idx_nonzero])) + x_thor[j][2]*((1 - np.exp(-x_thor[j][3]*tau_star[idx_nonzero]))/
         (x_thor[j][3]*tau_star[idx_nonzero]) - np.exp(-x_thor[j][3]*tau_star[idx_nonzero]))

    # compute value of each zero coupon-bond
    v = facev*np.exp(-tau_star*x_star)

    return np.squeeze(v)
