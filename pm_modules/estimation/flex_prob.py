# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 23:04:51 2024

@author: Thomas

# Aim of the Package
    # Modules to generate Flexible Probability Distributions


"""

# %% Packages
import numpy as np
from bisect import bisect_right
from bisect import bisect_left
from scipy import stats
import pandas as pd

# %% Support Functions

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

def state_conditioning_crisp_fp(z, 
                                z_star, 
                                alpha):
    """
    Ziel:
        This function computes crisp probabilities based on a dataset and a set of target values, 
        ensuring that the probabilities adhere to specified alpha thresholds.

    Parameters
        z : array, shape (t_bar, )
        z_star : array, shape (k_bar, )
        alpha : scalar

    Returns:
        p : array, shape (t_bar,k_bar) if k_bar>1 or (t_bar,) for k_bar==1
        z_lb : array, shape (k_bar,)
        z_ub : array, shape (k_bar,)
    """

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

    return np.squeeze(p.T), np.squeeze(z_lb), np.squeeze(z_ub)
    

def state_conditioning_smooth_kernel_fp():
    
    ### Univariate
    ########## inputs (you can change them) ##########
    z_star = 1  # target value
    h = 0.2  # kernel bandwidth
    gamma = 2  # kernel type parameter
    ##################################################
    
    p_smooth = exp(-(abs(vix_score - z_star)/h)**gamma)  # smooth kernel probabilities
    p_smooth = p_smooth/sum(p_smooth)  # rescaled probabilities
    
    ### Multivaraite
    # Das muss dann mit Mahalanobis bauen
    
    
    
    return None






### General Conditioning
def minium_relative_entropy_fp():

    return None



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




# %% TESTING

import matplotlib.pyplot as plt



t = 100
a = time_conditioning_exponentional_fp(t, 100)

plt.plot(a)

b = time_conditioning_linear_fp(100)


plt.plot(b)

np.sum(a)
effective_number_of_scenarios(b,absolut=True)






# %% Entwicklung

    