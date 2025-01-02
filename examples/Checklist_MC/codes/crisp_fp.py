# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats
from bisect import bisect_right
from bisect import bisect_left


# define an auxiliary function for computing the smooth quantiles
def quantile_smooth(c_bar, x, p=None, h=None):
    """
    Parameters
    ----------
        c_bar : scalar, array, shape(k_bar,)
        x : array, shape (j_bar,)
        p : array, shape (j_bar,), optional
        h: scalar, optional

    Returns
    -------
        q : array, shape (k_bar,)
    """

    c_bar = np.atleast_1d(c_bar)
    j_bar = x.shape[0]
    k_bar = c_bar.shape[0]

    # sorted scenarios-probabilities
    if p is None:
        # equal probabilities as default value
        p = np.ones(j_bar)/j_bar
    sort_x = np.argsort(x)
    x_sort = pd.Series(x).iloc[sort_x]
    p_sort = p[sort_x]

    # cumulative sums of sorted probabilities
    u_sort = np.zeros(j_bar + 1)
    for j in range(1, j_bar + 1):
        u_sort[j] = np.sum(p_sort[:j])

    # kernel smoothing
    if h is None:
        h = 0.25*(j_bar**(-0.2))
    q = np.zeros(k_bar)
    for k in range(k_bar):
        w = np.diff(stats.norm.cdf(u_sort, c_bar[k], h))
        w = w/np.sum(w)
        q[k] = x_sort@w

    return np.squeeze(q)


def crisp_fp(z, z_star, alpha):
    """This function computes crisp probabilities based on a dataset and a set of target values, 
    ensuring that the probabilities adhere to specified alpha thresholds.

    Parameters
    ----------
        z : array, shape (t_bar, )
        z_star : array, shape (k_bar, )
        alpha : scalar

    Returns
    -------
        p : array, shape (t_bar,k_bar) if k_bar>1 or (t_bar,) for k_bar==1
        z_lb : array, shape (k_bar,)
        z_ub : array, shape (k_bar,)
    """

    z_star = np.atleast_1d(z_star)
    t_bar = z.shape[0]
    k_bar = z_star.shape[0]
    
    # sorted scenarios-probabilities
    p = np.ones(t_bar)/t_bar  # equal probabilities as default value
    sort_z = np.argsort(z)
    z_sort = pd.Series(z).iloc[sort_z]
   
    # sumulative sums of sorted probabilities
    u_sort = np.cumsum(p[:t_bar + 1])
    u_sort = np.zeros(t_bar + 1)
    for j in range(1, t_bar + 1):
        u_sort[j] = np.sum(p[:j])

    # compute cdf of the risk factor at target values
    cdf_z_star = np.zeros(k_bar)
    z_0 = z_sort.iloc[0] - (z_sort.iloc[1] - z_sort.iloc[0])*u_sort[1]/(u_sort[2] - u_sort[1])
    z_sort = np.append(z_0, z_sort)
    cindx = [0]*k_bar
    for k in range(k_bar):
        cindx[k] = bisect_right(z_sort, z_star[k])
    for k in range(k_bar):
        if cindx[k] == 0:
            cdf_z_star[k] = 0
        elif cindx[k] == t_bar + 1:
            cdf_z_star[k] = 1
        else:
            cdf_z_star[k] = u_sort[cindx[k]-1] + (u_sort[cindx[k]] - u_sort[cindx[k]-1])*\
                           (z_star[k] - z_sort[cindx[k]-1])/(z_sort[cindx[k]] - z_sort[cindx[k]-1])
    cdf_z_star = np.squeeze(cdf_z_star)
    cdf_z_star = np.atleast_1d(cdf_z_star)

    # compute crisp probabilities
    z_lb = np.zeros(k_bar)
    z_ub = np.zeros(k_bar)
    p = np.zeros((k_bar, t_bar))
    pp = np.zeros((k_bar, t_bar))
    for k in range(k_bar):
        # compute range
        if z_star[k] <= quantile_smooth(alpha/2, z):
            z_lb[k] = np.min(z)
            z_ub[k] = quantile_smooth(alpha, z)
        elif z_star[k] >= quantile_smooth(1 - alpha/2, z):
            z_lb[k] = quantile_smooth(1 - alpha, z)
            z_ub[k] = np.max(z)
        else:
            z_lb[k] = quantile_smooth(cdf_z_star[k] - alpha/2, z)
            z_ub[k] = quantile_smooth(cdf_z_star[k] + alpha/2, z)

        # crisp probabilities
        pp[k, (z <= z_ub[k])&(z >= z_lb[k])] = 1
        p[k, :] = pp[k, :]/np.sum(pp[k, :])

    return np.squeeze(p.T), np.squeeze(z_lb), np.squeeze(z_ub)
