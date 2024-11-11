# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 23:04:51 2024

@author: Thomas

# Aim of the Package
    # Modules to generate Flexible Probability Distributions


Offene Punkte:
    1) Allgemeine Formulierung von MRE
    2) Conditional_FP -> Prüfen
    3) Partial Sample Regression
    4) Evaluationsfunctionen über die Sensitivitäten der Flex. Prop
    
Quellen:
https://www.arpm.co/lab/quasi-bayesian-flexible-probabilities.html
https://www.arpm.co/lab/views-processing.html
https://www.arpm.co/lab/inference-mc-variational.html
"""

# %% Packages
import numpy as np
from bisect import bisect_right
from bisect import bisect_left
from scipy import stats
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.sparse import eye
from statsmodels.stats.weightstats import DescrStatsW



# %% Support Functions
def min_rel_entropy_sp(p_pri, 
                       z_ineq=None, 
                       mu_view_ineq=None, 
                       z_eq=None, 
                       mu_view_eq=None, 
                       normalize=True):
    
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


def quantile_smooth(c_bar, 
                    x, 
                    p = None, 
                    h = None):
    """
    Ziel:
        Auxiliary function for computing the smooth quantiles
        
    Parameters
        c_bar : scalar, array, shape(k_bar,)
        x : array, shape (j_bar,)
        p : array, shape (j_bar,), optional
        h: scalar, optional

    Returns
        q : array, shape (k_bar,)
    """

    c_bar = np.atleast_1d(c_bar)
    j_bar = x.shape[0]
    k_bar = c_bar.shape[0]

    # sorted scenarios-probabilities
    if p is None:
        # equal probabilities as default value
        p = np.ones(j_bar)/j_bar
    sort_x = np.argsort(x)
    x_sort = pd.Series(x).iloc[sort_x]
    p_sort = p[sort_x]

    # cumulative sums of sorted probabilities
    u_sort = np.zeros(j_bar + 1)
    for j in range(1, j_bar + 1):
        u_sort[j] = np.sum(p_sort[:j])

    # kernel smoothing
    if h is None:
        h = 0.25*(j_bar**(-0.2))
    q = np.zeros(k_bar)
    for k in range(k_bar):
        w = np.diff(stats.norm.cdf(u_sort, c_bar[k], h))
        w = w/np.sum(w)
        q[k] = x_sort@w

    return np.squeeze(q)


def smoothing():
    return None 
    
def scoring():
    return None


# %% Basic Functions for Flexible Probs

def mean_sp():
    return None


def cov_sp():
    return None




# %% Functions for Determining Flexible Probabilities


### Time Conditioning
def time_conditioning_exponentional_fp(t_bar:"int",
                                       tau_hl:"int",
                                       t_star:"int" = None) -> 'np.array(t_bar,1)':
    '''
    Ziel:
        Erstellte Gewichte für einen exponentionellen Filter für eine Zeitreihe mit Länge t_star
        Die Erstellung basiert ausschließlich auf der Zeit relativ zum aktuellsten Zeitpunkt.
        
    Parameter:
        t_bar: Länge des Filters / Länge der Zeitreihe, auf die der Filter angewendet wird.
        tau_hl: Half-Life-Parameter für Determinierung der Gewichte
        t_star: Referenz-Zeitpunkt des Filters (Punkt mit höchster Wsk)
        
    Returns:
        p_exp_w: Numpy-Array (t_bar,1) mit (normalisierten) exponentiellen Gewichten
    '''
    # 1. Check: t_star > t_bar
    if t_star is not None:
        if t_star > t_bar:
            t_star = t_bar
            print("Debug: t_star > t_bar. t_star auf t_bar gesetzt")
    
    # 2. Check: Falls t_star nicht definiert, setze t_star = t_bar
    if t_star is None:
        t_star = t_bar
    
    # Calculation of Exponentional Filter
    p_exp = np.exp(-(np.log(2)/tau_hl)*abs(t_star - np.arange(1, t_bar+1)))
    
    # Reweighting of Exponentional Filter
    p_exp_w = p_exp / np.sum(p_exp)
    
    # Reshape
    p_exp_w = p_exp_w.reshape(-1,1)
    return p_exp_w

def time_conditioning_linear_fp(t_bar:"int")-> 'np.array(t_bar,1)':
    
    '''
    Ziel:
        Erstellte Gewichte für einen linearen Filter für eine Zeitreihe mit Länge t_star
        Die Erstellung basiert ausschließlich auf der Zeit relativ zum aktuellsten Zeitpunkt.
        
    Parameter:
        t_bar: Länge des Filters / Länge der Zeitreihe, auf die der Filter angewendet wird.
        tau_hl: Half-Life-Parameter für Determinierung der Gewichte
        
    Returns:
        p_exp_w: Numpy-Array (t_bar,1) mit (normalisierten) exponentiellen Gewichten
    '''
    
    # Calculation of Linear Filter
    p_lin = t_bar - np.arange(0,t_bar)[::-1]
    
    # Reweighting of Linear Filter
    p_lin_w = p_lin / np.sum(p_lin)
    
    # Reshape
    p_lin_w = p_lin_w.reshape(-1,1)
    return p_lin_w


### State Conditioning
'''
geht state_conditioning_crisp_fp für multivariate?
'''

def state_conditioning_crisp_fp(z: "np.array(t_bar,1)", 
                                z_star: "np.array(k_bar,)", 
                                alpha: "float") -> "tuple(np.array(t_bar,1), np.array(k_bar,), np.array(k_bar,)":
    
    """
    Ziel:
        This function computes crisp probabilities based on a dataset and a set of target values, 
        ensuring that the probabilities adhere to specified alpha thresholds.
        
        Observations within the "Confidence-Band" (CDF(z_star) +/- (alpha/2) have identical probabilities.
        Observations outside the "Confidence-Band" have a probability of zero.

    Parameters
        z : array, shape (t_bar, 1)
        z_star : array, shape (k_bar, )
        alpha : scalar

    Returns:
        p : array, shape (t_bar,k_bar) if k_bar>1 or (t_bar,) for k_bar==1
        z_lb : array, shape (k_bar,)  Lower Bound of original Array Z
        z_ub : array, shape (k_bar,)  Upper Bound of original Array Z
    """
    # Transform z from (array,1) to (array,)
    if z.ndim > 1:
        z = z.squeeze()

    # Get Basic Information about z and z_star
    z_star = np.atleast_1d(z_star)
    t_bar = z.shape[0]
    k_bar = z_star.shape[0]
    
    # sorted scenarios-probabilities
    p = np.ones(t_bar)/t_bar  # equal probabilities as default value
    sort_z = np.argsort(z)
    z_sort = pd.Series(z).iloc[sort_z]
   
    # sumulative sums of sorted probabilities
    u_sort = np.cumsum(p[:t_bar + 1])
    u_sort = np.zeros(t_bar + 1)
    for j in range(1, t_bar + 1):
        u_sort[j] = np.sum(p[:j])

    # compute cdf of the risk factor at target values
    cdf_z_star = np.zeros(k_bar)
    z_0 = z_sort.iloc[0] - (z_sort.iloc[1] - z_sort.iloc[0])*u_sort[1]/(u_sort[2] - u_sort[1])
    z_sort = np.append(z_0, z_sort)
    cindx = [0]*k_bar
    for k in range(k_bar):
        cindx[k] = bisect_right(z_sort, z_star[k])
    for k in range(k_bar):
        if cindx[k] == 0:
            cdf_z_star[k] = 0
        elif cindx[k] == t_bar + 1:
            cdf_z_star[k] = 1
        else:
            cdf_z_star[k] = u_sort[cindx[k]-1] + (u_sort[cindx[k]] - u_sort[cindx[k]-1])*\
                           (z_star[k] - z_sort[cindx[k]-1])/(z_sort[cindx[k]] - z_sort[cindx[k]-1])
    cdf_z_star = np.squeeze(cdf_z_star)
    cdf_z_star = np.atleast_1d(cdf_z_star)

    # compute crisp probabilities
    z_lb = np.zeros(k_bar)
    z_ub = np.zeros(k_bar)
    p = np.zeros((k_bar, t_bar))
    pp = np.zeros((k_bar, t_bar))
    for k in range(k_bar):
        # compute range
        if z_star[k] <= quantile_smooth(alpha/2, z):
            z_lb[k] = np.min(z)
            z_ub[k] = quantile_smooth(alpha, z)
        elif z_star[k] >= quantile_smooth(1 - alpha/2, z):
            z_lb[k] = quantile_smooth(1 - alpha, z)
            z_ub[k] = np.max(z)
        else:
            z_lb[k] = quantile_smooth(cdf_z_star[k] - alpha/2, z)
            z_ub[k] = quantile_smooth(cdf_z_star[k] + alpha/2, z)

        # crisp probabilities
        pp[k, (z <= z_ub[k])&(z >= z_lb[k])] = 1
        p[k, :] = pp[k, :]/np.sum(pp[k, :])

    return p.T, np.squeeze(z_lb), np.squeeze(z_ub)
    

def state_conditioning_smooth_kernel_fp(z: "np.array(t_bar,n_bar)", 
                                        z_star: "np.array(k_bar,1)", 
                                        h: "np.array(n_bar,n_bar)",
                                        gamma: "int" = 2) -> "np.array(t_bar,1)":
    """
    Ziel:
        This Function calculates Smooth Kernel Probabilities.
        The closer values in z are to z_star, the higher the corresponding probability
        
        The Choice of the Kernel is based on:
            1) gamma: gamma = 1 -> Exp. Kernel, gamma = 2 -> Gaussian Kernel
            2) Number of Variables
            3) Z-Star - Critical Values under Evaluation
            4) h: Bandwith-Matrix
        ! For the Multivariate Case n_bar > 1, only the Gaussian Kernel is used !
        

    Parameters
        z : array, shape (t_bar, n_bar) - Variables determining the Probabilities
        z_star : array, shape (k_bar, 1) - Criticial Values 
        h: array, shape (n_bar, n_bar) - Bandwith Matrix
        gamma : integer, [0,1] - Determines the Kernel (1-> Exp, 2 -> Gaussian)

    Returns:
        p : array, shape (t_bar,1)
        
    Notes:
        The Variables in z should be stationary.
        
        !!! Bitte nochmal prüfen !!!
    """                                     

    ## Get Basic Stats
    n_bar = z.shape[1]
    
    ## Checks
    # Vergleich z und z_star für n_bar
    assert z.shape[1] == z_star.shape[1], "Anzahl der Variablen zwischen z und z_star nicht identisch."
    assert h.shape[0] == h.shape[1], "Falsche Dimension in Bandwith Matrix."
    assert z.shape[1] == h.shape[1], "Anzahl der Variablen zwischen z und Bandwith Matrix nicht identisch."    
    assert (h.T == h).all(), "Bandwith Matrix ist nicht symmetrisch."
                                             
    ## Univariate Case
    if n_bar == 1:
        p_smooth = np.exp(-(abs(z - z_star)/h)**gamma)  # smooth kernel probabilities
        p_smooth = p_smooth/sum(p_smooth)  # rescaled probabilities
    
    
    ## Multivariate Case
    if n_bar > 1:
        print("Multivariate Z: gamma set to 2")
        p_smooth = np.exp(-np.diag((z -z_star)@np.linalg.inv(h)@(z -z_star).T))
        p_smooth = p_smooth / sum(p_smooth)
        '''
        Ergebnis identisch zu Schleifen-Formulierung
        '''
    
    return p_smooth.reshape(-1,1)
    

### Partial Sample Regressoin
def partial_sample_regression():
    return None



### General Conditioning
# !!! To Do !!! -> Funktioniert noch nicht!
def conditional_fp(z: "np.array(t_bar, 1)", 
                   z_star: "np.array(k_bar,1)", 
                   alpha: "float", 
                   p_prior: "np.array(t_bar, 1)") -> "np.array(t_bar,1)":
        
    """
    Ziel:
        This function calculates conditional flexible probabilities 
        based on a given dataset and a reference set of values. 
        Kombiniert Crisp Probabilities mit einem Prior (z.B. Exponentional)

    Parameters
        z : array, shape (t_bar, 1 )
        z_star : array, shape (k_bar, 1 )
        alpha : scalar
        p_prior : array, shape (t_bar, 1 )

    Returns
        p : array, shape (t_bar, k_bar) if k_bar > 1 or (t_bar, 1) for k_bar=1

    Notes:

    """
    # Change Dimension of z
    if z.ndim == 1:
        z = z.reshape(-1,1)
    
    z_star = np.atleast_1d(z_star)

    t_bar = z.shape[0]
    k_bar = z_star.shape[0]

    # compute crisp probabilities
    p_crisp, _, _ = state_conditioning_crisp_fp(z, z_star, alpha)
    p_crisp = p_crisp.T
    p_crisp[p_crisp == 0] = 10**-20
    
    if k_bar == 1:
        p_crisp = p_crisp/np.sum(p_crisp)
        p = np.zeros((k_bar, t_bar))
        # moments
        m_z = p_crisp@z
        s2_z = p_crisp@(z**2) - m_z**2
        # constraints
        a_ineq = np.atleast_2d(z**2)
        b_ineq = np.atleast_1d((m_z**2) + s2_z)
        a_eq = np.array([z])
        b_eq = np.array([m_z])
        # output
        p = min_rel_entropy_sp(p_prior, a_ineq, b_ineq, a_eq, b_eq)
    else:
        for k in range(k_bar):
            p_crisp[k, :] = p_crisp[k, :]/np.sum(p_crisp[k, :])

        # compute conditional flexible probabilities
        p = np.zeros((k_bar, t_bar))
        for k in range(k_bar):
            # moments
            m_z = p_crisp[k, :]@z
            s2_z = p_crisp[k, :]@(z**2) - m_z**2
            # constraints
            a_ineq = np.atleast_2d(z**2)
            b_ineq = np.atleast_1d((m_z**2) + s2_z)
            a_eq = np.array([z])
            b_eq = np.array([m_z])
            # output
            p[k, :] = min_rel_entropy_sp(p_prior, a_ineq, b_ineq, a_eq, b_eq)

    return p.T




# %% Evaluation

def dirichlet():
    return None


def effective_number_of_scenarios(p_array:'np.array(t,1)',
                                  absolut:"bool" = True) -> "float":
    '''
    Ziel:
        Berechnung der ENS für die Bestimmung der statistichen Signifikanz der Flex. Prob Weights
        
    Parameter:
        p_array: Array mit Dimension (t,1), für das die ENS berechnet werden
        absolut: Wenn "True" wird die absolute ENS ausgegeben, die auf der Länge des Arrays basiert
                 Wenn "False" wird die relative ENS ausgeben, welcher für die Länge des Array korrigiert
                 -> absolut = False -> ENS / len(p_array)
        
    Returns:
        ens: Effective Number of Scenarios: Type Float; Falls absolut = False -> [0,1]
        
    '''
    # Korrektur um Null-Wahrscheinlichkeiten
    p_array_adj = p_array[p_array > 0]  # Korrektur für Nullen im Array, wg. Log-Function  
    
    # Berechnung des ENS
    ens = np.exp(-p_array_adj@np.log(p_array_adj))  # effective number of scenarios
    
    # Relative ENS
    if absolut is False:
       ens = ens / p_array.shape[0]         
    return ens



    