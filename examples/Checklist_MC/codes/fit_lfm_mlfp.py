# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from scipy import special
from sklearn.linear_model import Lasso
from fit_lfm_ols import fit_lfm_ols


def fit_lfm_mlfp(x, z, p=None, nu=4, tol=1e-3, fit_intercept=True, maxiter=500, 
                 print_iter=False, rescale=False, shrink=False, lam=0.):
    """This function fits a Linear Factor Model (LFM) using Maximum Likelihood with Flexible Probabilities (MLFP). 
    It estimates the factor loadings, intercepts, residual covariances, and residuals based on the input time series data.

    Parameters
    ----------
        x : array, shape (t_bar, n_bar) if n_bar>1 or (t_bar, ) for n_bar=1
        z : array, shape (t_bar, k_bar) if k_bar>1 or (t_bar, ) for k_bar=1
        p : array, optional, shape (t_bar,)
        nu : scalar, optional
        tol : float, optional
        fit_intercept: bool, optional
        maxiter : scalar, optional
        print_iter : bool, optional
        rescale : bool, optional
        shrink : bool, optional
        lam : float, optional

    Returns
    -------
       alpha : array, shape (n_bar,)
       beta : array, shape (n_bar, k_bar) if k_bar>1 or (n_bar, ) for k_bar=1
       sigma2 : array, shape (n_bar, n_bar)
       eps : shape (t_bar, n_bar) if n_bar>1 or (t_bar, ) for n_bar=1
    """

    if np.ndim(x) < 2:
        x = x.reshape(-1, 1).copy()
    t_bar, n_bar = x.shape
    if np.ndim(z) < 2:
        z = z.reshape(-1, 1).copy()
    t_bar, n_bar = x.shape
    k_bar = z.shape[1]

    if p is None:
        p = np.ones(t_bar)/t_bar

    # rescale the variables
    if rescale is True:
        sigma2_x = np.cov(x.T, aweights=p)
        sigma_x = np.sqrt(np.diag(sigma2_x))
        x = x.copy()/sigma_x

        sigma2_z = np.cov(z.T, aweights=p)
        sigma_z = np.sqrt(np.diag(sigma2_z))
        z = z.copy()/sigma_z

    # Step 0: Set initial values using method of moments
    if shrink:
        if lam == 0:
            alpha, beta, sigma2, eps = fit_lfm_ols(x, z, p, fit_intercept)
        else:
            if fit_intercept is True:
                m_x = p@x
                m_z = p@z
            else:
                m_x = np.zeros(n_bar,)
                m_z = np.zeros(k_bar,)
            x_p = ((x - m_x).T*np.sqrt(p)).T
            z_p = ((z - m_z).T*np.sqrt(p)).T
            clf = Lasso(alpha=lam/(2.*t_bar), fit_intercept=False)
            clf.fit(z_p, x_p)
            beta = clf.coef_
            if k_bar == 1:
                alpha = m_x - beta*m_z
                eps = x - alpha - z*beta
            else:
                alpha = m_x - beta@m_z
                eps = x - alpha - z@np.atleast_2d(beta).T
            sigma2 = np.cov(eps.T, aweights=p)
    else:
        alpha, beta, sigma2, eps = fit_lfm_ols(x, z, p, fit_intercept=fit_intercept)
    alpha, beta, sigma2, eps = alpha.reshape((n_bar, 1)), beta.reshape((n_bar, k_bar)),\
                               sigma2.reshape((n_bar, n_bar)), eps.reshape((t_bar, n_bar))
    if nu > 2.:
        # if nu <=2, then the covariance is not defined
        sigma2 = (nu - 2.)/nu*sigma2

    mu_eps = np.zeros(n_bar)
    for i in range(maxiter):
        # step 1: update the weights and historical flexible probabilities
        if nu >= 1e3 and np.linalg.det(sigma2) < 1e-13:
            w = np.ones(t_bar)
        else:
            w = (nu + n_bar)/(nu + np.sum((eps - mu_eps).T*np.linalg.solve(sigma2, (eps - mu_eps).T), axis=0))
        q = w*p
        q = q/np.sum(q)

        # step 2: update shift parameters, factor loadings and covariance
        alpha_old, beta_old, sigma2_old = alpha, beta, sigma2
        if shrink:
            if lam == 0:
                alpha, beta, sigma2, eps = fit_lfm_ols(x, z, q, fit_intercept)
            else:
                if fit_intercept is True:
                    m_x = q@x
                    m_z = q@z
                else:
                    m_x = np.zeros(n_bar,)
                    m_z = np.zeros(k_bar,)
                x_q = ((x - m_x).T * np.sqrt(q)).T
                z_q = ((z - m_z).T * np.sqrt(q)).T
                clf = Lasso(alpha=lam/(2.*t_bar), fit_intercept=False)
                clf.fit(z_q, x_q)
                beta = clf.coef_
                if k_bar == 1:
                    alpha = m_x - beta*m_z
                    eps = x - alpha - z*beta
                else:
                    alpha = m_x - beta@m_z
                    eps = x - alpha - z@np.atleast_2d(beta).T
                sigma2 = np.cov(eps.T, aweights=q)
        else:
            alpha, beta, sigma2, eps = fit_lfm_ols(x, z, q, fit_intercept=fit_intercept)
        alpha, beta, sigma2, eps = alpha.reshape((n_bar, 1)), beta.reshape((n_bar, k_bar)),\
                                   sigma2.reshape((n_bar, n_bar)), eps.reshape((t_bar, n_bar))
        sigma2 = (w@q)*sigma2

        # step 3: check convergence
        beta_tilde_old = np.column_stack((alpha_old, beta_old))
        beta_tilde = np.column_stack((alpha, beta))
        errors = [np.linalg.norm(beta_tilde - beta_tilde_old, ord=np.inf)/np.linalg.norm(beta_tilde_old, ord=np.inf),
                  np.linalg.norm(sigma2 - sigma2_old, ord=np.inf)/np.linalg.norm(sigma2_old, ord=np.inf)]
        # print the loglikelihood and the error
        if print_iter is True:
            if np.shape(sigma2) == 0:
                # univaraite student t
                lf = stats.t.logpdf(eps, nu, mu_eps, sigma2)
            else:
                # multivariate student t
                n_bar = sigma2.shape[0]
                d2 = np.sum((eps - mu_eps).T*np.linalg.solve(sigma2, (eps - mu_eps).T), axis=0)
                lf = -((nu + n_bar)/2.)*np.log(1. + d2/nu) + special.gammaln((nu + n_bar)/2.) -\
                     special.gammaln(nu/2.) - (n_bar/2.)*np.log(nu*np.pi) - 0.5*np.linalg.slogdet(sigma2)[1]
            print('Iter: %i; Loglikelihood: %.5f; Errors: %.3e' %(i, q@lf, max(errors)))
        if max(errors) < tol:
            break
    if rescale is True:
        alpha = alpha*sigma_x
        beta = ((beta/sigma_z).T*sigma_x).T
        sigma2 = (sigma2.T*sigma_x).T*sigma_x

    return np.squeeze(alpha), np.squeeze(beta), np.squeeze(sigma2), np.squeeze(eps)
