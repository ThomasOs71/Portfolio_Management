#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
import warnings


def fit_nelson_siegel_bonds(v_bond, c, upsilon, *, facev=1, theta_0=None):
    
    """This function fits the parameters of the Nelson-Siegel yield curve model to a set of bond values and parameters. 
    It iterates through each bond, computes the Nelson-Siegel yield curve, calculates the bond value, and computes 
    a minimization function to adjust the parameters such that the model closely matches the observed bond values.

    Parameters
    ----------
        v_bond : array, shape (n_bar,)
        c : dict, length (n_bar)
        upsilon : dict, length (n_bar)
        facev : scalar
        theta_0: array, shape (4,)

    Returns
    -------
        theta : array, shape (4,)

    """
    
    n_bar = len(v_bond)

    def fit_nelsonsiegel_bonds_target(theta):                      
        v_bond_theta = np.zeros(n_bar)
        output = 0.0
        for n in range(n_bar):
            # step 1: compute Nelson-Siegel yield curve
            y_ns_theta = theta[0] - theta[1]*((1 - np.exp(-theta[3] * upsilon[n]))/(theta[3] * upsilon[n])) + theta[2]*\
                   ((1 - np.exp(-theta[3] * upsilon[n]))/(theta[3] * upsilon[n]) - np.exp(-theta[3] * upsilon[n]))
             
            # step 2: compute coupon bond value        
            v_zcb = np.exp(-upsilon[n]*y_ns_theta)  # zero-coupon bond value
            v_bond_theta[n] = facev*(c[n]@v_zcb)  # bond value
            
            # step 3: compute minimization function 
            if n==0:
                h_tilde = (upsilon[n+1][-1]-upsilon[n][-1])/2
            elif n==n_bar-1:
                h_tilde = (upsilon[n][-1]-upsilon[n-1][-1])/2
            else:
                h_tilde = (upsilon[n+1][-1]-upsilon[n-1][-1])/2 
            output += h_tilde * np.abs(v_bond_theta[n] - v_bond[n])
        return output
    
    if theta_0 is None:
        theta_0 = 0.1*np.ones(4)
        
    # step 4: fit Nelson-Siegel parameters
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = minimize(fit_nelsonsiegel_bonds_target, theta_0,
                       bounds=((None, None), (None, None), (None, None), (0, None)))
    theta = res.x

    return theta
