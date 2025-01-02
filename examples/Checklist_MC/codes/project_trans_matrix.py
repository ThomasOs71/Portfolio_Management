# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import expm, logm
from cvxopt import matrix
from cvxopt.solvers import qp, options

from min_rel_entropy_sp import min_rel_entropy_sp

def project_trans_matrix(p, delta_t, credit=False):
    """This function computes the projected transition matrix, considering optional credit constraints

    Parameters
    ----------
        p : array, shape (c_bar, c_bar)
        delta_t : scalar
        credit : boolean (optional)

    Returns
    -------
        p_delta_t : array, shape (c_bar, c_bar)

    """

    # Step 1: Compute log-matrix
    l = logm(p)    # log-matrix of p1

    # Step 2: Compute generator
    c_bar = len(p)
    p_a = matrix(np.eye(c_bar*c_bar))
    q = matrix(-l.reshape((c_bar*c_bar, 1)))
    G = matrix(0.0, (c_bar*c_bar, c_bar*c_bar))
    G[::c_bar*c_bar + 1] = np.append([0], np.tile(np.append(-np.ones(c_bar), [0]), c_bar - 1))
    h = matrix(0.0, (c_bar*c_bar, 1))
    a = matrix(np.repeat(np.diagflat(np.ones(c_bar)), c_bar, axis=1))
    b = matrix(0.0, (c_bar, 1))
    options['show_progress'] = False
    g = qp(p_a, q, G, h, a, b)['x']  # generator matrix

    # Step 3: Compute projected transition matrix
    g = np.array(g).reshape((c_bar, c_bar)) 
    p_delta_t = expm(delta_t*g)  # projected transition matrix

    if credit is True:
        p = p_delta_t
        # probability constraint
        a_eq = np.ones((1, c_bar))  # 1×c_bar dimensional vector of ones
        b_eq = np.array([1])
        # initialize monotonicity constraint
        a_ineq = {}
        a_ineq[0] = np.diagflat(np.ones((1, c_bar-1)), 1) - np.diagflat(np.ones((1, c_bar)), 0)  # (c_bar-1)×c_bar upper triangular matrix
        a_ineq[0] = a_ineq[0][:-1]
        b_ineq = np.zeros((c_bar-1))  # 1×(c_bar-1) dimensional vector of ones
        p_delta_t = np.zeros((c_bar, c_bar))
        for c in range(c_bar - 1):
            p_delta_t[c, :] = min_rel_entropy_sp(p[c, :], a_ineq[c], b_ineq, a_eq, b_eq, False)  # minimize relative entropy
            a_temp = a_ineq.get(c).copy()  # update monotonicity constraint
            a_temp[c, :] = -a_temp[c, :]
            a_ineq[c+1] = a_temp.copy()
        p_delta_t[-1, :] = np.zeros((1, p.shape[1]))  # default constraint
        p_delta_t[-1, -1] = 1
        
    return p_delta_t
