#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from bisect import bisect_right
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter

from schweizer_wolff import schweizer_wolff


def invariance_test_copula(eps, lag_bar, k_bar=None):
    """
       This function assesses copula invariance by conducting a Schweizer-Wolff dependence test and plots the results,
       including a 3D histogram of bivariate data and a bar plot illustrating dependence over various lags.

    Parameters
    ----------
        eps : array, shape (t_bar,)
        lag_bar : scalar
        k_bar: int

    Returns
    -------
        sw: array, shape(lag_bar,)

    """

    t_bar = eps.shape[0]

    # Schweizer-Wolff dependence for lags
    sw = np.zeros(lag_bar)
    for l in range(lag_bar):
        sw[l] = schweizer_wolff(np.column_stack((eps[(l + 1):], eps[: -(l + 1)])))

    # grades scenarios
    x_lag = eps[:-lag_bar]
    y_lag = eps[lag_bar:]
    p = np.ones(np.column_stack((x_lag, y_lag)).shape[0])/np.column_stack((x_lag, y_lag)).shape[0]  # equal probabilities
    x_grid, ind_sort = np.sort(np.column_stack((x_lag, y_lag)), axis=0), np.argsort(np.column_stack((x_lag, y_lag)), axis=0)  # sorted scenarios
    # marginal cdf's
    cdf_x = np.zeros(np.column_stack((x_lag, y_lag)).shape)
    for n in range(np.column_stack((x_lag, y_lag)).shape[1]):
        x_bar = x_grid[:, n]
        x_bar = np.atleast_1d(x_bar)
        x = np.column_stack((x_lag, y_lag))[:, n]

        # sorted scenarios-probabilities
        sort_x = np.argsort(x)
        x_sort = pd.Series(x).iloc[sort_x]
        p_sort = p[sort_x]

        # cumulative sums of sorted probabilities
        u_sort = np.zeros(x.shape[0] + 1)
        for j in range(1, x.shape[0] + 1):
            u_sort[j] = np.sum(p_sort[:j])

        # output cdf
        cindx = [0]*x_bar.shape[0]
        for k in range(x_bar.shape[0]):
            cindx[k] = bisect_right(x_sort, x_bar[k])
        cdf_x[:, n] = u_sort[cindx]
        
    # copula scenarios
    u = np.zeros(np.column_stack((x_lag, y_lag)).shape)
    for n in range(np.column_stack((x_lag, y_lag)).shape[1]):
        u[ind_sort[:, n], n] = cdf_x[:, n]
    u[u >= 1] = 1 - np.spacing(1)
    u[u <= 0] = np.spacing(1)  # clear spurious outputs
    
    # normalized histogram
    if k_bar is None:
        k_bar = np.floor(np.sqrt(7*np.log(t_bar)))
    k_bar = int(k_bar)
    
    p = np.ones(u.shape[0])/u.shape[0]  # uniform probabilities
    min_x_1 = np.min(u[:, 0])
    min_x_2 = np.min(u[:, 1])

    # bin width
    h_1 = (np.max(u[:, 0]) - min_x_1)/k_bar
    h_2 = (np.max(u[:, 1]) - min_x_2)/k_bar

    # bin centroids
    xi_1 = np.zeros(k_bar)
    xi_2 = np.zeros(k_bar)
    for k in range(k_bar):
        xi_1[k] = min_x_1 + (k + 1 - 0.5)*h_1
        xi_2[k] = min_x_2 + (k + 1 - 0.5)*h_2

    # normalized histogram heights
    f = np.zeros((k_bar, k_bar))
    for k_1 in range(k_bar):
            for k_2 in range(k_bar):
                # take edge cases into account
                if k_1 > 0 and k_2 > 0:
                    ind = ((u[:, 0] > xi_1[k_1] - h_1/2)&(u[:, 0] <= xi_1[k_1] + h_1/2) &
                           (u[:, 1] > xi_2[k_2] - h_2/2)&(u[:, 1] <= xi_2[k_2] + h_2/2))
                elif k_1 > 0 and k_2 == 0:
                    ind = ((u[:, 0] > xi_1[k_1] - h_1/2)&(u[:, 0] <= xi_1[k_1] + h_1/2) &
                           (u[:, 1] >= min_x_2)&(u[:, 1] <= xi_2[k_2] + h_2/2))
                elif k_1 == 0 and k_2 > 0:
                    ind = ((u[:, 0] >= min_x_1)&(u[:, 0] <= xi_1[k_1] + h_1/2) &
                           (u[:, 1] > xi_2[k_2] - h_2/2) & (u[:, 1] <= xi_2[k_2] + h_2/2))
                else:
                    ind = ((u[:, 0] >= min_x_1)&(u[:, 0] <= xi_1[k_1] + h_1/2) &
                           (u[:, 1] >= min_x_2)&(u[:, 1] <= xi_2[k_2] + h_2/2))

                f[k_1, k_2] = np.sum(p[ind])/(h_1*h_2)

    ############################################################## plots ##############################################################
    # 2D histogram
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xpos, ypos = np.meshgrid(xi_1 - (xi_1[1] - xi_1[0])/2, xi_2 - (xi_2[1] - xi_2[0])/2)  # adjust bin centers to left edges
    ax.bar3d(xpos.flatten('F'), ypos.flatten('F'), np.zeros_like(xpos.flatten('F')), xi_1[1] - xi_1[0], xi_2[1] - xi_2[0], f.flatten())
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f')); ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f')); ax.invert_xaxis()
    plt.xlabel('Grade obs.'); plt.ylabel('Grade lagged obs.')
    
    # dependence plot
    fig = plt.figure()
    plt.bar(range(1, lag_bar + 1), sw, 0.5, facecolor='#969696', edgecolor='#212529')
    plt.bar(range(1, lag_bar + 1)[lag_bar - 1], sw[lag_bar - 1], 0.5, facecolor='#f56502', edgecolor='#212529')
    plt.xlabel('Lag'); plt.ylabel('Dependence'); plt.ylim([0, 1]); plt.xticks(np.arange(1, lag_bar + 1))

    return sw
