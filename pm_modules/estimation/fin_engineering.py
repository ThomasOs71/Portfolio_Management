# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:44:30 2024

@author: Thomas

# Financial Engineering
"""

# %% Packages
import numpy as np
from scipy.optimize import minimize

import numpy as np
import pandas as pd
from bisect import bisect_right
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from schweizer_wolff import schweizer_wolff

# %% Step 1: Valuation
'''
Brauchen wohl Funktionen für Illiquid Assets
'''



# %% Step 2: Risk Driver
'''
Risk Driver determine the P&L of Instruments which can be used in Econometric Analysis.
'''

### Equity
def risk_driver_equity(total_return_index: "np.array"):
    '''
    Aim: Generates the Risk Driver for Equities
        
    Parameters
    ----------
    total_return_index : "np.array"
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return np.log(total_return_index)

### Fixed Income
def risk_driver_fe_log_ytm(ytm,
                             ):
    '''
    Aim : 
    ------
    Generates 
    Generates Risk Driver of Fixed Income as Log-YtM

    Quote ARPM:
    When the investment horizon is rather long, in the order of months, 
    or when yields are low, such as 1% or less, it becomes more likely for rates
    to approach zero; thus the constraint that yields are positive disrupts the homogeneity of their evolution. 
    In such situations, we cannot use yields as risk drivers for bonds.

    Parameters:
    ------
    ytm: np.array(t,n) für YTMs der Yield Curve

    Returns
    -------
    ytm_log: np.array(t,n) für LOG YTMs der Yield Curve

    Notes:
    -------
    Keep the Parameterization of the Risk Driver for FE in Mind for later Repricing!

    '''
    return np.log(ytm)


def risk_driver_fe_shadow_rates(ytm,
                                eta = 0.013,
                                lower_shift = 0):
    '''
    Aim: 
    Generates the Risk Driver of (Risk-Free) Fixed Income as Inverse-Call Transformation.
    
    Shadow rates follow a homogeneous, random-walk-like process in both high-rates and low-rates environments.

    Parameters
    ----------
    ytm : np.array(t,n) für YTMs der Yield Curve
    lower_shift: TBD

    Returns
    -------
    ytm_shadow_rates: np.array(t,n) für Shadow Rates der YTM der Yield Curve (per Inverse Call)

    Notes:
    -------
    - Keep the Parameterization of the Risk Driver for FE in Mind for later Repricing!
    - https://www.tandfonline.com/doi/abs/10.2469/faj.v72.n3.7
    '''
    assert lower_shift <= 0, "Lower Shift muss kleiner 0 sein"
    
    # Use Shift in Case of Negative YTM
    y_t = ytm - lower_shift  # c_eta^(-1) <- c_eta^(-1)(ytm_t - y_lower_shift)
    
    # Inverse Call-Transformation
    c_inv_eta = np.zeros((y_t.shape))
    c_inv_eta[y_t >= eta] = y_t[y_t >= eta]
    c_inv_eta[y_t < eta] = eta*(np.log(y_t[y_t < eta]/eta)+1)
    
    return c_inv_eta


def risk_driver_fe_nelson_siegel(ytm,
                                 tau_grid,
                                 ):
    '''
    

    Yields
    ------
    TYPE
        DESCRIPTION.

    tau_select = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20])  # times to maturity

    '''
    # Parameter
    t_bar = len(ytm)
    
    ########## input (you can change it) ########## 
    theta_init = 0.5*np.ones(4)  # initial parameters 
    ###############################################

    def minimization_target(theta, t):
        # Nelson-Siegel yield curve
        y_ns_theta = theta[0] - theta[1]*((1 - np.exp(-theta[3]*tau_grid))/(theta[3]*tau_grid)) + theta[2]*\
                     ((1 - np.exp(-theta[3]*tau_grid))/(theta[3]*tau_grid) - np.exp(-theta[3]*tau_grid))  
        # minimization
        output = 0.0
        for n in range (len(tau_grid)):
            if n == 0:
                h = tau_grid[n + 1] - tau_grid[n]
            elif n == len(tau_grid) - 1:
                h = tau_grid[n] - tau_grid[n - 1]
            else:
                h = (tau_grid[n + 1] - tau_grid[n - 1])/2
            output += h*abs(ytm[t][n] - y_ns_theta[n])
        return output 

    # compute Nelson-Siegel yield curve
    theta = np.zeros((t_bar, 4))
    y_ns = np.zeros((t_bar, len(tau_grid)))
    for t in range(t_bar):
        # Nelson-Siegel model parameters
        if t==0:
            theta[t] = minimize(minimization_target, theta_init, args=(0)).x
        else:
            theta[t] = minimize(minimization_target, theta[t - 1], args=(t)).x
        # Nelson-Siegel yield curve    
        y_ns[t, :] = theta[t, 0] - theta[t, 1]*((1 - np.exp(-theta[t, 3]*tau_grid))/(theta[t, 3]*tau_grid)) + theta[t, 2]*\
                     ((1 - np.exp(-theta[t, 3]*tau_grid))/(theta[t, 3]*tau_grid) - np.exp(-theta[t, 3]*tau_grid))  
                     
    return y_ns



def risk_driver_fe_nelson_siegel_2_yc():
    return None




def risk_driver_fe_credit_risk(ytm_credit,
                               ytm_credit_grid,
                               ytm_riskfree,
                               ytm_riskfree_grid,
                               log_modelling = True,
                               yc_method = "Shadow_Rates",
                               eta = 0.013,
                               lower_shift = 0):
    '''
    S_t(tau) = Y_t(tau) - Y_t^(ref)(tau)
    '''
    ### Berechnung von Y_t^(ref)(tau)
    if yc_method == "Shadow_Rates": # -> Macht das überhaupt hier sinn?
        pass
    elif yc_method == "Nelson_Siegel":
        pass
    elif yc_method == "Log_YTM":
        pass
    else:
        raise ValueError("Eine bekannte Function für yc_method.")
    '''    
    ### Berechnung von Credit über S_t(tau) = Y_t(tau) - Y_t^(ref)(tau)
    1. relevante ytm_riskfree determinieren
    2. Spread errechnen als difference
    3. log nehmen?
    spread_t = ytm_credit 
    
    if log_modelling is True:
        return np.log(spread_t), np
    
    # Berechnung der Credit Spreads aus den Daten
    '''
    return None



### Fixed Income (ETF)
'''
Was machen wir bei FIxed Income ETF?
'''

### Currency
def risk_driver_fx(total_return_index: "np.array"):
    '''
    Aim: Generates the Risk Driver for Currencies

    Parameters
    ----------
    total_return_index : "np.array"
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return np.log(total_return_index)



### Commodities (ETFs / None Futures)
'''
Was machen wir bei Commodity ETF
'''


### Risk Driver Testing Routine

def risk_driver_missing_values():
    return None


def risk_driver_outlier():
    return None






# %% Step 3-I: Quest for Invariance
'''
Use the Risk Drivers to:
    1) Find a suitable Econometric Model which is driven by
    2) multivariate iid process
-> Both can then be used to forecast the Evaluation of the Risk Drivers
    
'''

### Step 3-Ia: Univariate Quest for Invariance
'''
Multivariate Models for Invariants extraction
'''
def random_walk_univariate():
    # 1. IID Test
    # 2. Estimation of Invariant Distribution
    # hier muss definitiv das testing rein... und auch die Bestimmung über die Verteilung der Invariants
    return None

def garch_univariate():
    # 1. IID Test
    # 2. Estimation of Invariant Distribution
    # hier muss definitiv das testing rein... und auch die Bestimmung über die Verteilung der Invariants
    return None

















### Step 3-Ia: Univariate Quest for Invariance
'''
Univariate Models for Invariants extraction
'''








### Step 3-II: Testing Invariance? (Simulation Testing?)
'''
Esting der Modells? -> Ist das nötig?
'''


# %% Step 4: Repricing
'''
Generating Instrument Prices from the Simulation
'''

### Equities
def repricing_equities_full():
    return None


def repricing_equities_taylor_approx():
    return None




### Fixed Income



### Currencies








