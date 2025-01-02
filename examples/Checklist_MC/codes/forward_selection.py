# -*- coding: utf-8 -*-
import numpy as np


def forward_selection(optim, n_bar, k_bar=None):
    
    """This function performs forward selection by iteratively building sets of selections, 
    optimizing over constraint sets, and conducting light-touch searches, returning the selected elements, 
    corresponding function values, and a list of selection sets at each step.

    Parameters
    ----------
        optim : function
        n_bar : int
        k_bar : int

    Returns
    -------
        x_fwd : list
        f_x_fwd : array, shape(k_bar,)
        s_k_fwd_list : list
    """

    if k_bar:
        k_bar = min(k_bar, n_bar)
    else:
        k_bar = n_bar
    i_1 = np.arange(1, n_bar+1)
    x_fwd = []
    f_x_fwd = np.zeros(k_bar)

    # step 0: initialize
    s_k_fwd = []
    s_k_fwd_list = []
    for k in range(k_bar):
        # step 1: build k-element set of selections
        s_prev_fwd = s_k_fwd
        i_k_fwd = []
        x_k = []
        f_x_k = []
        for n in np.setdiff1d(i_1, s_prev_fwd):
            i_k_fwd.append(np.union1d(s_prev_fwd, n).astype(int))
            
            # step 2: optimize over constraint set
            all = optim(i_k_fwd[-1])
            x_k.append(all[0])
            f_x_k.append(all[1])

        # step 3: perform light-touch search
        opt_indices = np.argmin(f_x_k)
        x_fwd.append(x_k[opt_indices])
        f_x_fwd[k] = f_x_k[opt_indices]
        s_k_fwd = i_k_fwd[opt_indices]
        s_k_fwd_list.append(s_k_fwd)

    return x_fwd, f_x_fwd, s_k_fwd_list
