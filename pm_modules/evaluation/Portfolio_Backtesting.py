# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:28:51 2024

Backtesting Portfolio Strategies


@author: Felix
"""

import numpy as np


#%% Backtesting

def port_bh(r, w,  v_0 = 100.0):
    
    '''
    Buy-and-Hold Portfolio Backtesting for Multiple Periods
     
    Parameters
    ----------
    
    r:      np.array([], ndim=2), 
            discrete return matrix (T x K)
    w:      np.array([], ndim=1), 
            target weight matrix (T x K)
    v_0:    float, 
            initial portfolio value    
    
    Returns
    -------
    
    out:    dict, 
            portfolio evolvement incl. weighting history
    
    Notes
    -----
    Backtesting assumes that trading fractional shares are possible.
    
    '''
    # Dimension
    T = r.shape[0]
    K = r.shape[1]
    
    # Portfolio Value
    port = np.zeros(T+1)# T+1 dimension
    bop_value = np.zeros((T, K))
    #eop_value = np.zeros((T, K))
    #  Weighting
    bop_w = np.zeros((T, K))
    #eop_w = np.zeros((T, K))
    
    port[0] = v_0
    bop_w[0, :] = w
    bop_value[0, :] = v_0 * w
    
    eop_value = bop_value[0, :] * np.cumprod(1 + r, axis=0)  
    port[1:] = np.sum(eop_value, axis=1)
    eop_w = eop_value /  port[1:].reshape(-1, 1)
    
    bop_w[1:] = eop_w[:-1]
    bop_value[1:] = eop_value[:-1]
    
    out = {}
    out["Portfolio"] = port
    out["Return"] = port[1:] / port[:-1] - 1
    out["BOP"] = {"Value": bop_value, "Weight": bop_w}
    out["EOP"] = {"Value": eop_value, "Weight": eop_w}
    
    
    return out


def port_multi_period(r, w, v_0 = 100.0, variable=0.0, fix=0.0, tol=0.0, rebalance_int=None):
    
    '''
    Multi-Period Portfolio Backtesting 
    (assumes that trading fractional shares are possible)
    
    Parameters
    ----------
    
    r:              np.array([], ndim=2), 
                    discrete return matrix (T x K)
    w:              np.array([], ndim=2), 
                    target weight matrix (T x K)
    v_0:            float, 
                    initial portfolio value
    variable:       float, 
                    proportional transaction costs on absolute turnover in percentage
    fix:            float,     
                    fix transaction costs in money units
    tol:            float,
                    turnover tolerance to initiate transaction
    rebalance_int:  bool,
                    rebalancing interval if given
    
    Returns
    -------
    
    out:            dict, 
                    portfolio evolvement incl. trading and weighting history
    
    Notes
    -----
    Backtesting assumes that trading fractional shares are possible.
    '''
    
    # Dimension
    T = r.shape[0]
    K = r.shape[1]
    
    if w.shape != r.shape:
        raise SystemExit('Shape of w must be equal to shape of r')
    
    # Portfolio Value
    port = np.zeros(T+1)# T+1 dimension
    bop_value = np.zeros((T, K))
    eop_value = np.zeros((T, K))
    #  Weighting
    bop_w = np.zeros((T, K))
    eop_w = np.zeros((T, K))
    # Costs
    transaction = np.zeros((T, K))
    turnover = np.zeros((T, K))
    cost = np.zeros((T, K))
    
    port[0] = v_0
    bop_w[0, :] = w[0,:]
    
    if rebalance_int is None:
        
        # Block 1: No fixed rebalancing interval
        rebalance_int = np.zeros(T, dtype=bool)
        
        for t in range(0, T): 
            
            if t > 0:
                
                turnover[t,:] = abs(eop_w[t-1, :] - w[t,:]) 
                rebalance_int[t] = np.sum(turnover[t]) > tol
                
                # Check for Transaction
                if rebalance_int[t]:
                     # trade
                     transaction[t,:] = (eop_w[t-1, :] - w[t,:]) * port[t]
                     bop_w[t, :]  = w[t,:]
                     var_costs = variable * turnover[t,:] * port[t] # variable costs
                     cost[t,:]  = var_costs + (var_costs > 0) * fix # variable  + fix costs
                else:
                     # no trade
                     bop_w[t, :] =  eop_w[t-1, :]
                     # if no trade, override turnover with zero
                     turnover[t,:] = 0.0
            
            # Portfolio Evolvement    
            bop_value[t,:] = (port[t] - np.sum(cost[t, :])) * bop_w[t, :] 
            eop_value[t,:] = bop_value[t,:] * (1 + r[t,:])  
             
            port[t+1] = np.sum( eop_value[t,:] )
            eop_w[t, :] = eop_value[t,:] /  port[t+1].reshape(-1, 1)
            
    else:
        # Block 2: Fixed rebalancing interval is given
        
         for t in range(0, T):
            
            if t > 0:
                
                turnover[t,:] = abs(eop_w[t-1, :] - w[t,:]) 
                
                # Check for Transaction
                if rebalance_int[t]:
                     # trade
                     transaction[t,:] = (eop_w[t-1, :] - w[t,:]) * port[t]
                     bop_w[t, :]  = w[t,:]
                     var_costs = variable * turnover[t,:] * port[t] # variable costs
                     cost[t,:]  =  var_costs + (var_costs > 0) * fix # variable  + fix costs
                else:
                     # no trade
                     bop_w[t, :] =  eop_w[t-1, :]
                     # if no trade, override turnover with zero
                     turnover[t,:] = 0.0
            
            # Portfolio Evolvement    
            bop_value[t,:] =  (port[t] - np.sum(cost[t, :])) * bop_w[t, :] 
            eop_value[t,:] = bop_value[t,:] * (1 + r[t,:])  
             
            port[t+1] = np.sum( eop_value[t,:] )
            eop_w[t, :] = eop_value[t,:] /  port[t+1].reshape(-1, 1)
            
    
    # Summary Output        
    out = {}
    out["Portfolio"] = port
    out["Return"] = port[1:] / port[:-1] - 1
    out["Costs"] = cost
    out["Turnover"] = turnover
    out["Transaction"] = transaction
    out["Rebalancing"] = rebalance_int
    out["BOP"] = {"Value": bop_value, "Weight": bop_w}
    out["EOP"] = {"Value": eop_value, "Weight": eop_w}
    
    
    return out


#%% Examples
'''

import pandas as pd
import matplotlib.pyplot as plt

asset_labels = ["Stock", "Bond", "Alternatives", "Cash"]
n_assets = len(asset_labels)
# Observation per Year
freq = 252
# Number of Years to Simulate
years = 1
# Length of Simulation
T = freq * years

mu = np.array([0.07, 0.03, 0.04, 0.01], ndmin = 2) / freq
sigma = np.array([0.18, 0.05, 0.20, 0.005], ndmin = 2) / (freq**0.5)
phi = np.array([ [1, 0.15, 0.35, 0], 
                 [0.15, 1, -0.15, 0.05],
                 [0.35, -0.15, 1, -0.05],
                 [0, 0.05, -0.05, 1] ])

print(pd.DataFrame(mu*freq, columns = asset_labels))
print(pd.DataFrame(sigma*(freq**0.5), columns = asset_labels) )
print(pd.DataFrame(phi, columns = asset_labels, index=asset_labels))

# Covariance Matrix: Cov(i,j) = rho_ij * sigma_i * sigma_j
Sigma = phi * np.outer(sigma, sigma)

        
np.random.seed(100)
x = np.random.multivariate_normal(mu.flatten(), Sigma, size=T)
p = np.cumprod(1+x, axis=0)
pd.DataFrame(np.corrcoef(x.T))


plt.plot(p )
plt.grid()
plt.title("Simulated Paths")
plt.legend(asset_labels)
plt.show()

# Weights
w_array = np.tile([0.6, 0.3, 0.05, 0.05], (252, 1))

# Buy& Hold

# Test: Buy and Hold Portfolio
test = port_bh(r=x, w=w_array[0])
plt.plot(test["Portfolio"])
plt.plot(test["Return"])
plt.plot(test["EOP"]["Weight"])
plt.plot(test["BOP"]["Weight"])

# Test 1:  rebalance_int trading with turnover tolerance
test0 = port_multi_period( r = x, w = w_array,
                          v_0 = 1000000.0, variable=0.0025, fix=10, tol=0.01,
                          rebalance_int=None)       

plt.plot(test0["Portfolio"])
plt.plot(test0["Return"])
plt.plot(test0["BOP"]["Weight"])
plt.plot(test0["EOP"]["Weight"])
plt.plot(np.sum(test0["Turnover"], axis=1))
plt.plot(np.sum(test0["Costs"], axis=1))


# Test 2:  rebalance_int trading with given rebalancing indicator
test1 = port_multi_period( r = x, w = w_array,
                          v_0 = 1000000.0, variable=0.0025, fix=10, tol=0.01,
                          rebalance_int = np.ones(252, dtype=bool))   

plt.plot(test1["Portfolio"])
plt.plot(test1["Return"])
plt.plot(test1["BOP"]["Weight"])
plt.plot(test1["EOP"]["Weight"])
plt.plot(np.sum(test1["Turnover"], axis=1))
plt.plot(np.sum(test1["Costs"], axis=1))

'''
