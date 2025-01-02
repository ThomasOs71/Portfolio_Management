# -*- coding: utf-8 -*-
import numpy as np


def backward_selection(optim, n_bar, k_bar=None):
    
    """This function conducts backward selection, optimizing a given function over a constraint set,
    iteratively reducing the set of selections, and returning the selected elements, 
    corresponding function values, and a list of the selection sets at each step.

    Parameters
    ----------
        optim : function
        n_bar : intg
        k_bar : int

    Returns
    -------
        x_bwd : list
        f_x_bwd : array, shape(k_bar,)
        s_k_bwd_list : list
    """

    if k_bar:
        k_bar = min(k_bar, n_bar)
    else:
        k_bar = n_bar
    x_bwd = []
    f_x_bwd = []

    # step 0: initialize
    s_k_bwd = np.arange(1, n_bar + 1)
    s_k_bwd_list = []
    i_k_bwd = [s_k_bwd]

    for k in range(n_bar, 0, -1):
        x_k = []
        f_x_k = []
        for s_k_i in i_k_bwd:
            # step 1: optimize over constraint set
            all_ = optim(s_k_i)
            x_k.append(all_[0])
            f_x_k.append(all_[1])

        # step 2: perform light-touch search
        opt_indices = np.argmin(f_x_k)
        s_k_bwd = i_k_bwd[opt_indices]
        if k <= k_bar:
            x_bwd.insert(0, x_k[opt_indices])
            f_x_bwd.insert(0, f_x_k[opt_indices])
            s_k_bwd_list.insert(0, s_k_bwd)

        # step 3: build (k-1)-element set of selections
        i_k_bwd = []
        for n in s_k_bwd:
            i_k_bwd.append(np.setdiff1d(s_k_bwd, n).astype(int))

    return x_bwd, np.array(f_x_bwd), s_k_bwd_list
