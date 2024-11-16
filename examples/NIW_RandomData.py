# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:00:31 2024

@author: Thomas

Ziel:
    1) Codes -> Sollte okay sein
    2) EInbau von Views....über Sample möglich
        3) Wann führt eine Veränderung des Erwartungswertes nicht zu einer Veränderung der Weights?
            Eigentlich nur dann, wenn die Correlation identisch bleibt und die 

"""
# %% Packages
### External
import numpy as np
import scipy.stats as ss

### Internal
from pm_modules.estimation.Inference import *
from pm_modules.estimation.joint_dist import *

# %% Testing

# DoF 
v_true = 8

# Location
mu_true = np.array([0.08,0.1,0.04,0.03])

# Dispersion
vola_true = np.array([0.18,0.20,0.06,0.06])
corr_true = np.array([[1,0.6,0.2,0.1],
                     [0.6,1,0.2,0.2],
                     [0.2,0.2,1,0.6],
                     [0.1,0.2,0.6,1]])

cov_true = np.diag(vola_true)@corr_true@np.diag(vola_true)
assert (corr_true.T == corr_true.T).all()  # Check Symmetrie
assert (cov_true.T == cov_true.T).all()  # Check Symmetrie

sigma2_true = (v_true-2)/(v_true) * cov_true  # sigma2_t = (v-2)/v * cov

# %% Testung NIW
### Parameter
w = np.ones(4) / len(mu_true)  # Strat. Weights
t_pri = 100
v_pri = 100
t_bar = 200
lam = 0.5

# Prior Parameter
sigma2_pri = cov_true
mu_pri = bl_implied_equilibirum_returns(sigma2_pri, w, 2*(1 - lam)/lam)

prior = dict({"mu_pri": mu_pri,
              "sigma2_pri": sigma2_pri})


# Sample Parameter
mu_sample = mu_true.reshape(-1,1)
sigma2_sample = cov_true

sample = dict({"mu_sample":mu_sample,
               "sigma2_sample": sigma2_sample})

# Posterior Parameters
posterior = niw_posterior(mu_pri,
                          sigma2_pri,
                          mu_sample,
                          sigma2_sample,
                          t_pri,
                          v_pri,
                          t_bar
                          )

pred_pos = niw_predictive_posterior(posterior["mu_pos"], 
                             posterior["sigma2_pos"], 
                             posterior["v_pos"], 
                             posterior["t_pos"])

'''

allo_bayes = niw_bayesian_allocation(prior,
                                     sample,
                                     t_pri,
                                     v_pri,
                                     t_bar,
                                     0.1)
'''

### Test - Änderung der Exp. Return bei gleicher Vola
# Sample Parameter
mu_sample = mu_true.reshape(-1,1)
sigma2_sample = cov_true

mu_sample_adjust = mu_pri.copy() # Eine Abänderung
mu_sample_adjust[0] = mu_pri[0]*1.25

sample = dict({"mu_sample":mu_sample_adjust,
               "sigma2_sample": sigma2_sample})

# Posterior Parameters
posterior = niw_posterior(mu_pri,
                          sigma2_pri,
                          mu_sample_adjust,
                          sigma2_sample,
                          t_pri,
                          v_pri,
                          t_bar
                          )

pred_pos = niw_predictive_posterior(posterior["mu_pos"], 
                             posterior["sigma2_pos"], 
                             posterior["v_pos"], 
                             posterior["t_pos"])

'''

allo_bayes = niw_bayesian_allocation(prior,
                                     sample,
                                     t_pri,
                                     v_pri,
                                     t_bar,
                                     0.1)

'''
# %% Testing
###
r_cov = pred_pos["r_sigma2"] * (pred_pos["r_nu"])/((pred_pos["r_nu"]) - 2)
r_exp = pred_pos["r_mu"]

###
def weights_mv(cov_):
    c1 = np.ones(4)@np.linalg.inv(cov_)@np.ones(4)
    return (1/c1) * np.linalg.inv(cov_)@np.ones((4,1))
    
def weights_msn(cov_,mu_):
    c2 = np.ones(4)@np.linalg.inv(cov_)@mu_
    return (1/c2) * np.linalg.inv(cov_)@mu_

# Distribution under Prior
weights_msn(cov_true, mu_pri)
'''
Should be Equal to w_str
'''

weights_mv(r_cov)
weights_msn(r_cov, r_exp)










