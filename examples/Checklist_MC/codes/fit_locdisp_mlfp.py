# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from scipy import special


def fit_locdisp_mlfp(eps, *, p=None, nu=1000, threshold=1e-3, maxiter=1000, print_iter=False):
    
    """This function estimates the location and dispersion parameters of
    a multivariate time series using the maximum likelihood method.

    Parameters
    ----------
        eps : array, shape (t_bar, i_bar)
        p : array, shape (t_bar,), optional
        nu: float, optional
        threshold : float, optional
        maxiter : int, optional
        print_iter : bool

    Returns
    -------
        mu : array, shape (i_bar,)
        sigma2 : array, shape (i_bar, i_bar)
    """

    if len(eps.shape) == 1:
        eps = eps.reshape(-1, 1)

    t_bar, i_bar = eps.shape

    if p is None:
        p = np.ones(t_bar)/t_bar

    # step 0: set initial values using method of moments
    mu = p@eps
    sigma2 = ((eps - mu).T*p)@(eps - mu)

    if nu > 2.:
        # if nu <=2, then the covariance is not defined
        sigma2 = sigma2*(nu - 2.)/nu

    for i in range(maxiter):
        # step 1: update the weights
        if nu >= 1e3 and np.linalg.det(sigma2) < 1e-13:
            w = np.ones(t_bar)
        else:
            if eps.shape[1] > 1:
                w = (nu + i_bar)/(nu + np.squeeze(np.sum((eps - mu).T*np.linalg.solve(sigma2, (eps - mu).T), axis=0)))
            else:
                w = (nu + i_bar)/(nu + np.squeeze((eps - mu)/sigma2*(eps - mu)))
        q = w*p
        
        # step 2: update location and dispersion parameters
        mu_old, sigma2_old = mu, sigma2
        mu = q@eps
        sigma2 = ((eps - mu).T*q)@(eps - mu)
        mu = mu/np.sum(q)

        # step 3: check convergence
        er = max(np.linalg.norm(mu - mu_old, ord=np.inf)/np.linalg.norm(mu_old, ord=np.inf),
                  np.linalg.norm(sigma2 - sigma2_old, ord=np.inf)/np.linalg.norm(sigma2_old, ord=np.inf))

        if print_iter is True:
            if np.shape(sigma2) == 0:
                # univaraite student t
                lf = stats.t.logpdf(eps, nu, mu, sigma2)
            else:
                # multivariate student t
                n_bar = sigma2.shape[0]
                d2 = np.sum((eps - mu).T*np.linalg.solve(sigma2, (eps - mu).T), axis=0)
                lf = -((nu + n_bar)/2.)*np.log(1. + d2/nu) + special.gammaln((nu + n_bar)/2.) -\
                     special.gammaln(nu/2.) - (n_bar/2.)*np.log(nu*np.pi) - 0.5*np.linalg.slogdet(sigma2)[1]
            print('Iter: %i; Loglikelihood: %.5f; Error: %.5f'%(i, p@lf, er))

        if er <= threshold:
            break
            
    return np.squeeze(mu), np.squeeze(sigma2)
