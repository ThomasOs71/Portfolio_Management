# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:45:11 2024

@author: Thomas

Verfahren -> Minimum Relative Entropy
Verfahren #2 -> Kinlaw

"""


# %% Packages (External)
from numpy import arange, array, zeros, eye, ones, sqrt, r_,\
                  linspace, median, sum, where, diag, block
from numpy.random import lognormal
from numpy.linalg import eig
from scipy.stats import chi2
from cvxopt.solvers import options, socp
from cvxopt import matrix
from matplotlib.pyplot import show, fill_between

import numpy as np
from scipy import stats
from scipy import special

# %% Support Function

### Sigma2 <-> Covariance: Student t
def cov2sig2_t(cov: "np.array(n,n)", 
               nu: int) -> "np.array (n,n)":
    '''
    Transforms a Covariance into a Sigma2 Dispersion for a Student t with Nu Dof.
    '''
    sigma2 = cov*(nu - 2.)/nu
    return sigma2
    
def sig2tocov_t(sigma2: "np.array (n,n)", 
                nu: int) -> "np.array (n,n)" :
    '''
    Transforms a Sigma2 Dispersion for a Student t with NU Dof into a Covariance.
    '''
    cov = sigma2*nu/(nu - 2.)
    return cov

### Sample Mean / Dispersion
def loc_disp_fp(eps: "np.array(t,n)",
                p: "np.array(t,1)") -> "np.array(n,), np.array(n,n)":
    if p.ndim == 2:
        p = p.squeeze()
    mu = p@eps
    sigma2 = ((eps - mu).T*p)@(eps - mu)
    return mu, sigma2

### Mahalanobis
def mahab_distance():
    return None


# %% Robust Optimzation - Student t

### Kernroutine
def fit_locdisp_mlfp(eps: "np.array((t_bar, i_bar)", 
                     *, 
                     p: "np.array(t_bar,1)" = None, 
                     nu: "int" = 1000, 
                     threshold: float = 1e-3, 
                     maxiter: int = 1000, 
                     print_iter: bool = False) -> dict:
    
    """
    This function estimates the location and Dispersion parameters of
    a multivariate time series using the maximum likelihood method assuming a Student t Distribution.

    Parameters
    ----------
        eps : array, shape (t_bar, i_bar)
        p : array, shape (t_bar, 1), optional
        nu: float, optional
        threshold : float, optional
        maxiter : int, optional
        print_iter : bool


    Returns
    -------
        mu : array, shape (i_bar, 1)
        sigma2 : array, shape (i_bar, i_bar)
        nu: int
        loglf: float, loglikelihood
    """
    ## Preliminary Data Check
    # Check Dimension of EPS -> t x n
    if len(eps.shape) == 1:
        eps = eps.reshape(-1, 1)

    t_bar, i_bar = eps.shape

    # Check p
    if p is None:
        p = np.ones(t_bar)/t_bar
    else:
        p = np.squeeze(p)

    ## step 0: set initial values using method of moments
    mu, sigma2 = loc_disp_fp(eps,p)
    
    '''
    # mu = p@eps #Previous Code
    # sigma2 = ((eps - mu).T*p)@(eps - mu) #Previous Code
    '''
    if nu > 2.:
        # if nu <=2, then the covariance is not defined
        '''
        # sigma2 = sigma2*(nu - 2.)/nu #Previous Code
        '''
        sigma2 = sig2tocov_t(sigma2, nu)

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

        if np.shape(sigma2) == 0:
            # univaraite student t
            lf = stats.t.logpdf(eps, nu, mu, sigma2)
        else:
            # multivariate student t
            n_bar = sigma2.shape[0]
            d2 = np.sum((eps - mu).T*np.linalg.solve(sigma2, (eps - mu).T), axis=0)
            lf = -((nu + n_bar)/2.)*np.log(1. + d2/nu) + special.gammaln((nu + n_bar)/2.) -\
                     special.gammaln(nu/2.) - (n_bar/2.)*np.log(nu*np.pi) - 0.5*np.linalg.slogdet(sigma2)[1]
        if print_iter is True:
            print('Iter: %i; Loglikelihood: %.5f; Error: %.5f'%(i, p@lf, er))

        if er <= threshold:
            break
    
    # Sumup Results:
    result_dict = {"mu": mu.reshape(-1,1),
                   "sigma2":sigma2,
                   "nu":int(nu),
                   "loglf":float(p@lf)}
            
    return result_dict


### Optimierung für mehrere DoF
def fit_locdisp_mlfp_nu(eps: "np.array(t_bar,i_bar)", 
                        *, 
                        p: "np.array(t_bar, 1)" = None, 
                        nu_min: int = 4,
                        nu_max: int = 100, 
                        threshold: float = 1e-3, 
                        maxiter: int = 1000, 
                        print_iter: bool = False) -> dict:
    
    """
    Ziel:
        This function estimates the location and Dispersion parameters of
        a multivariate time series using the maximum likelihood method assuming a Student t Distribution.
        Multiple Estimations are performed assuming different DoFs.

    Parameters
        eps : array, shape (t_bar, i_bar)
        p : array, shape (t_bar,), optional
        nu: float, optional
        threshold : float, optional
        maxiter : int, optional
        print_iter : bool


    Returns
    -------
        dict: Keys are the DoF
              Each Entry contains an dict with:
                  mu : array, shape (i_bar,1)
                  sigma2 : array, shape (i_bar, i_bar)
                  nu: int
                  loglf: float, loglikelihood
    """
    
    ### Define Grid of Nu
    nu_grid = arange(nu_min,nu_max)
    
    
    dict_results = {}

    for nu_ in nu_grid:
        dict_results[int(nu_)] = fit_locdisp_mlfp(eps, 
                                        p = p, 
                                        nu = nu_, 
                                        threshold = 1e-3, 
                                        maxiter = 1000, 
                                        print_iter = False)     
    return dict_results




def fit_locdisp_mlfp_nu_best(dict_results):
    '''
    Ziel:
        Find best Approximation of SPD by Multivariate Studend t.
        Previous Estimation is performed by fit_locdisp_mlfp_nu
    Parameter:
        dict_results: dict, contains results from fit_locdisp_mlfp_nu
    Return:
        dict: Contains parameter of best estimate:
            mu : array, shape (i_bar,1)
            sigma2 : array, shape (i_bar, i_bar)
            nu: int
            loglf: float, loglikelihood
    '''
    
    min_loglf = -100000
    best_nu = None
    
    for i in dict_results.keys():
        if dict_results[i]["loglf"] > min_loglf:
            best_nu = int(dict_results[i]["nu"])
            min_loglf = dict_results[i]["loglf"]

    return dict_results[best_nu]




# %% Robust Optimization - with missing data

### Randomly Missing Data


### Differences in Time Length

def fit_locdisp_mlfp_difflength(eps,
                                p,
                                nu):
    
    t_bar = eps.shape[0]
    
    
    # organize invariants into nested time series based on minimum index of first non-NaN value 
    i_bar = eps.shape[1]  # number of invariants
    l_bar = np.array([min(np.where(~np.isnan(eps[:, i]))[0]) for i in range(i_bar)])  # minimum indices of first non-NaN values of invariants
    # sort indices and invariants
    index = np.argsort(l_bar)
    l_sort, eps_sort = l_bar[index], eps[:, index] 
    
    # reshuffle invariants in a nested pattern
    c = 0
    eps_nested = []
    eps_nested.append(eps_sort[:, 0])
    t = []
    t.append(int(l_sort[0]))
    for j in range(1, i_bar):
        if l_sort[j] == l_sort[j - 1]:
            eps_nested[c] = np.column_stack((eps_nested[c], eps_sort[:, j]))
        else:
            c = c + 1
            eps_nested.append(eps_sort[:, j])
            t.append(int(l_sort[j]))
        
        
    
    # set initial parameters using HFP estimate
    mu_hat_mlfp, sigma2_hat_mlfp = fit_locdisp_mlfp(eps_nested[0], p=p, nu=nu, threshold=10**(-5), maxiter=10**5)


    ########## input (you can change it) ##########
    n_bar = 10  # number of stocks
    ###############################################
    
    v_tnow = lognormal(4, 0.05, n_bar)  # current values of stocks
    e_pi = 0.5*arange(1, n_bar + 1)  # mean vector of stock P&L's
    cv_pi = diag(e_pi)@(0.2*ones((n_bar, n_bar)) + 0.8*eye(n_bar))@diag(e_pi)  # covariance matrix of stock P&L's
    
    t = diag(diag(cv_pi))  # robustness matrix
    t[t >= median(diag(t))] = 10**-5*t[t >= median(diag(t))]  # high penalty for low-variance P&L's

    c_bar = len(eps_nested)  # number of nested buckets

    # maximize conditional log-likelihoods
    for c in range(1, c_bar):
        data = eps_nested[c][t[c]:]
        e = np.zeros((t_bar - t[c], i_bar))
        sza = 1
        for j in range(c):
            if eps_nested[j].ndim == 2:
                szb = eps_nested[j].shape[1]
                e[:, sza-1:sza-1+szb] = eps_nested[j][t[c]:t_bar, :]
            else:
                szb = 1
                e[:, sza-1:sza-1+szb] = np.atleast_2d(eps_nested[j][t[c]:t_bar]).T
            sza = sza + szb
        # probabilities
        p_k = p[t[c]:t_bar]/np.sum(p[t[c]:t_bar])
        # degrees of freedom
        nu_c = nu + e.shape[1]
        # loadings
        alpha, beta, s2, _ = fit_lfm_mlfp(data, e, p=p_k, nu=nu_c, tol=10**(-5), maxiter=10**5)
        if np.ndim(data) < 2:
            n_bar = 1
        else:
            n_bar = data.shape[1]
        if np.ndim(e) < 2:
            k_bar = 1
        else:
            k_bar = e.shape[1]
        alpha, beta, s2 = alpha.reshape((n_bar, 1)), beta.reshape((n_bar, k_bar)), s2.reshape((n_bar, n_bar))
        # update location and dispersion
        mah = np.sqrt(np.sum((e[-1, :].reshape(1, -1)-mu_hat_mlfp.reshape(-1)).T*\
              np.linalg.solve(sigma2_hat_mlfp, (e[-1, :].reshape(1, -1)-mu_hat_mlfp.reshape(-1)).T), axis=0))
        gamma = (nu_c/(nu + mah**2))*s2 + beta@sigma2_hat_mlfp@beta.T
        sigma2_hat_mlfp = np.r_[np.r_['-1', sigma2_hat_mlfp, sigma2_hat_mlfp@beta.T], np.r_['-1', beta@sigma2_hat_mlfp, gamma]]
        sigma2_hat_mlfp = (sigma2_hat_mlfp + sigma2_hat_mlfp.T)/2
        mu_hat_mlfp = np.r_[mu_hat_mlfp, alpha + beta@mu_hat_mlfp]
        
    # reshuffle output
    mu_hat_mlfp = mu_hat_mlfp[index]
    sigma2_hat_mlfp = sigma2_hat_mlfp[np.ix_(index, index)]
    return 

# %% Mean Shrinkage
### James Stein


# %% Covariance Shrinkage

### Denoising



### Spectrum (LW)

### Glasso

### Factor Analysis    


# %% Combined Approach
'''
https://www.arpm.co/lab/combining-estimation-techniques.html

James stein -> Forcasts von andren häusern?
'''




# %% Evaluation - Estimation / Decision Theory
'''
https://www.arpm.co/lab/estimation-assessment.html#x166-84700027.1
'''












