#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from bisect import bisect_right

def schweizer_wolff(x, p=None):
    """This function computes the Schweizer-Wolff measure for a given set of scenarios
    and their associated probabilities.

    Parameters
    ----------
        x : array, shape (j_bar, 2)
        p : array, shape (j_bar, )

    Returns
    -------
        sw : scalar
    """

    j_bar = x.shape[0]  # number of scenarios
    n_bar = x.shape[1]
    
    if p is None:
        p = np.ones(j_bar)/j_bar

    # grades scenarios
    x_grid, ind_sort = np.sort(x, axis=0), np.argsort(x, axis=0)  # sorted scenarios

    cdf_x1 = lambda x: np.sum(p*(x>=x_grid[:,0])) 
    cdf_x2 = lambda x: np.sum(p*(x>=x_grid[:,1])) 

    # copula scenarios
    u = np.zeros((j_bar, 2))
    for j in np.arange(j_bar):
        u[j, 0] = cdf_x1(x[j,0])
        u[j, 1] = cdf_x2(x[j,1])
    
    # joint scenario-probability cdf of grades
    cdf_u = np.array([np.sum(p*(u[:, 0] <= i/j_bar)*(u[:, 1] <= k/j_bar))
            for i in range(j_bar+1) for k in range(j_bar+1)]).reshape((j_bar+1, j_bar+1))

    # approximate Schweizer-Wolff measure
    sw_abs = np.array([np.abs(cdf_u[i, k] - i*k/j_bar**2)
             for i in range(j_bar+1) for k in range(j_bar+1)]).reshape((j_bar+1, j_bar+1))
    sw = 12*np.sum(sw_abs)/j_bar**2
    return sw
