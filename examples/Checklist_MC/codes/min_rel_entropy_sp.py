# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.sparse import eye
from statsmodels.stats.weightstats import DescrStatsW


def min_rel_entropy_sp(p_pri, z_ineq=None, mu_view_ineq=None, z_eq=None, mu_view_eq=None, normalize=True):
    """This function minimizes the relative entropy subject to inequality and equality constraints 
       and returns the updated probabilities

    Note
    ----
        The constraints :math:`p_j \geq 0` and :math:`\sum p_j = 1` are set automatically.

    Parameters
    ----------
        p_pri : array, shape(j_bar,)
        z_ineq : array, shape(l_bar, j_bar), optional
        mu_view_ineq : array, shape(l_bar,), optional
        z_eq : array, shape(m_bar, j_bar), optional
        mu_view_eq : array, shape(m_bar,), optional
        normalize : bool, optional

    Returns
    -------
        p_bar : array, shape(j_bar,)
    """

    # if there is no constraint, then just return p_pri
    if z_ineq is None and z_eq is None:
        return p_pri
    # no inequality constraints
    elif z_ineq is None:
        z = z_eq
        mu_view = mu_view_eq
        l_bar = 0
        m_bar = len(mu_view_eq)
    # no equality constraints
    elif z_eq is None:
        z = z_ineq
        mu_view = mu_view_ineq
        l_bar = len(mu_view_ineq)
        m_bar = 0
    else:
        # concatenate constraints
        z = np.concatenate((z_ineq, z_eq), axis=0)
        mu_view = np.concatenate((mu_view_ineq, mu_view_eq), axis=0)
        l_bar = len(mu_view_ineq)
        m_bar = len(mu_view_eq)
        
    # normalize constraints
    if normalize is True:
        m_z = DescrStatsW(z.T).mean 
        s2_z = DescrStatsW(z.T).cov 
        s_z = np.sqrt(np.diag(s2_z))
        z = ((z.T - m_z)/s_z).T
        mu_view = (mu_view - m_z)/s_z

    # pdf of a discrete exponential family
    def exp_family(theta):
        x = theta@z + np.log(p_pri)
        phi = logsumexp(x)
        p = np.exp(x - phi)
        p[p < 1e-32] = 1e-32
        p = p/np.sum(p)
        return p

    # minus dual Lagrangian
    def lagrangian(theta):
        x = theta@z + np.log(p_pri)
        phi = logsumexp(x)  # stable computation of log sum exp
        return phi - theta@mu_view

    def gradient(theta):
        return z@exp_family(theta) - mu_view

    def hessian(theta):
        p = exp_family(theta)
        z_bar = z.T - z@p
        return (z_bar.T*p)@z_bar

    # compute optimal lagrange multipliers and posterior probabilities
    k_bar = l_bar + m_bar  # dimension of lagrange dual problem
    theta0 = np.zeros(k_bar)  # intial value
    # if no constraints, then perform newton conjugate gradient
    # trust-region algorithm
    if l_bar == 0:
        options = {'gtol': 1e-10}
        res = minimize(lagrangian, theta0, method='trust-ncg', jac=gradient, hess=hessian, options=options)
    # otherwise perform sequential least squares programming
    else:
        options = {'ftol': 1e-10, 'disp': False, 'maxiter': 1000}
        alpha = -eye(l_bar, k_bar)
        constraints = {'type': 'ineq', 'fun': lambda theta: alpha@theta}
        res = minimize(lagrangian, theta0, method='SLSQP', jac=gradient, constraints=constraints, options=options)

    return np.squeeze(exp_family(res['x']))
