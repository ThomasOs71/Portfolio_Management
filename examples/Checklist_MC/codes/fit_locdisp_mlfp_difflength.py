# -*- coding: utf-8 -*-
import numpy as np
from fit_locdisp_mlfp import fit_locdisp_mlfp
from fit_lfm_mlfp import fit_lfm_mlfp


def fit_locdisp_mlfp_difflength(eps, p=None, nu=4., *, threshold=10**(-5), maxiter=10**5):
    
    """This function estimates location and dispersion parameters for multivariate series with different lengths, 
    reshuffling the series and maximizing conditional log-likelihoods iteratively while handling missing values.

    Parameters
    ----------
        eps : array, shape (t_bar,)
        p : array, shape (t_bar,), optional
        nu : float, optional
        threshold: float, optional
        maxiter : float

    Returns
    -------
        mu : float
        sigma2 : float

    Note: We suppose missing values, if any, are at beginning 
    (farthest observations in past could be missing).
    """

    if isinstance(threshold, float):
        threshold = [threshold, threshold]

    t_bar, i_bar = eps.shape

    if p is None:
        p = np.ones(t_bar)/t_bar

    # step 0: initialize

    # reshuffle series in a nested pattern, such that series with
    # longer history comes first and one with shorter history comes last
    l_bar = np.zeros(i_bar)
    for i in range(i_bar):
        l_bar[i] = min(np.where(~np.isnan(eps[:, i]))[0])

    index = np.argsort(l_bar)
    l_sort = l_bar[index]
    eps_sort = eps[:, index]
    idx = np.argsort(index)

    c = 0
    eps_nested = []
    eps_nested.append(eps_sort[:, 0])
    t = []
    t.append(int(l_sort[0]))
    for j in range(1, i_bar):
        if l_sort[j] == l_sort[j-1]:
            eps_nested[c] = np.column_stack((eps_nested[c], eps_sort[:, j]))
        else:
            c = c+1
            eps_nested.append(eps_sort[:, j])
            t.append(int(l_sort[j]))

    mu, sigma2 = fit_locdisp_mlfp(eps_nested[0], p=p, nu=nu, threshold=threshold[0], maxiter=maxiter)
    ii_bar = eps_nested[0].shape[1] if np.ndim(eps_nested[0]) > 1 else 1
    mu, sigma2 = mu.reshape((ii_bar, 1)), sigma2.reshape((ii_bar, ii_bar))

    # step 1: maximize conditional log-likelihoods
    for c in range(1, len(eps_nested)):
        data = eps_nested[c][t[c]:]
        e = np.zeros((t_bar - t[c], mu.shape[0]))
        sza = 1
        for j in range(c):
            if eps_nested[j].ndim == 2:
                szb = eps_nested[j].shape[1]
                e[:, sza-1:sza-1+szb] = eps_nested[j][t[c]:t_bar, :]
            else:
                szb = 1
                e[:, sza-1:sza-1+szb] = np.atleast_2d(eps_nested[j][t[c]:t_bar]).T
            sza = sza + szb
        # a) probabilities
        p_k = p[t[c]:t_bar]/np.sum(p[t[c]:t_bar])
        # b) degrees of freedom
        nu_c = nu + e.shape[1]
        # c) loadings
        alpha, beta, s2, _ = fit_lfm_mlfp(data, e, p=p_k, nu=nu_c, tol=threshold[1], maxiter=maxiter)
        if np.ndim(data) < 2:
            n_bar = 1
        else:
            n_bar = data.shape[1]
        if np.ndim(e) < 2:
            k_bar = 1
        else:
            k_bar = e.shape[1]
        alpha, beta, s2 = alpha.reshape((n_bar, 1)), beta.reshape((n_bar, k_bar)), s2.reshape((n_bar, n_bar))
        # d) location/scatter
        # Mahalanobis distance
        if e[-1, :].reshape(1, -1).shape[1] > 1:
            mah = np.squeeze(np.sqrt(np.sum((e[-1, :].reshape(1, -1) - mu.reshape(-1)).T*\
                                            np.linalg.solve(sigma2, (e[-1, :].reshape(1, -1) - mu.reshape(-1)).T), axis=0)))
        else:
            mah = np.squeeze(np.sqrt((e[-1, :].reshape(1, -1) - mu.reshape(-1))/sigma2*(e[-1, :].reshape(1, -1) - mu.reshape(-1))))
        gamma = (nu_c/(nu + mah**2))*s2 + beta@sigma2@beta.T
        sigma2 = np.r_[np.r_['-1', sigma2, sigma2@beta.T], np.r_['-1', beta@sigma2, gamma]]
        sigma2 = (sigma2 + sigma2.T)/2
        mu = np.r_[mu, alpha + beta@mu]

    # step 2: reshuffling output
    mu = mu[idx]
    sigma2 = sigma2[np.ix_(idx, idx)]

    return np.squeeze(mu), np.squeeze(sigma2)
