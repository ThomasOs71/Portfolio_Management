# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:00:31 2024

@author: Thomas

Ziel:
    1) Testing der Black-Litterman Routinen -> CHeck
    2) Sensitivitätsanalyse bei Schätzung der Daten -> Check
        1) Über Normal Distribution / Sample -> Check
        2) Über Student t -> Check
    3) Optimierung 

"""
# %% Packages
### External
import numpy as np
import scipy.stats as ss

### Internal
from pm_modules.estimation.Inference import *
from pm_modules.estimation.joint_dist import *


# %% Distribution Assumptions
'''
Student t Distribution
'''


# DoF 
v_true = 4

# Location
mu_true = np.array([0.08,0.1,0.04,0.03])

# Dispersion
vola_true = np.array([0.2,0.25,0.04,0.04])
corr_true = np.array([[1,0.6,0.2,0.1],
                     [0.6,1,0.2,0.2],
                     [0.2,0.2,1,0.6],
                     [0.1,0.2,0.6,1]])

cov_true = np.diag(vola_true)@corr_true@np.diag(vola_true)
assert (corr_true.T == corr_true.T).all()  # Check Symmetrie
assert (cov_true.T == cov_true.T).all()  # Check Symmetrie

sigma2_true = (v_true-2)/(v_true) * cov_true  # sigma2_t = (v-2)/v * cov


# %% Black Litterman - True Distribution
# Settings
lam = 5
tau = 50

weights = np.array([0.25,0.25,0.25,0.25])

v = np.array([[1,0,0,0],
              [0,1,0,0]])

eta = np.array([1,-1]).reshape(-1,1)
c = np.array([0.25,0.25]).reshape(-1,1)

### True Parameters
## Implied Returns
mu_r_impl_true = bl_implied_equilibirum_returns(cov_true, 
                                                weights,
                                                lam = lam)
'''
Wie setzen wir Lambda -> Roncalli
'''
## Prior
# Prior Mean Distribution
prior_true = bl_prior_distribution(mu_r_impl_true,
                                 cov_true,
                                 tau = tau)

# Prior Predictive Distribution
pri_pred_true = bl_prior_predictive_performance_distribution(mu_r_impl_true,
                                                             cov_true,
                                                             tau = tau)

## Views
# View Distribution
views_true = bl_views(v,
                      eta,
                      c,
                      prior_true,
                      pri_pred_true)

# Check
bl_check_views(v, 
               views_true[1], 
               prior_true)

# Explanation
bl_view_explanation(v,
                    views_true[0],
                    views_true[1])
'''
Uncertainty is super gering...liegt das am Tau?
'''

## Posterior
# Posterior Mean Distribution
posterior_true = bl_posterior_distribution(prior_true, 
                                           views_true[0], 
                                           views_true[1], 
                                           v, 
                                           cov_true, 
                                           tau)

## Predictive Distribution
pos_pred_true = bl_posterior_predictive_performance_distribution(posterior_true, 
                                                                 cov_true)
pos_mu_true = pos_pred_true[0]
pos_cov_true = pos_pred_true[1]


# %% Estimation - Theory Estimation of Parameters
#########
iterations = 1000
##########

## Save Array
cov_sample_array = np.zeros([iterations, len(mu_true), len(mu_true)])
pos_mean_array = np.zeros([len(mu_true),iterations])
pos_cov_array = np.zeros([iterations, len(mu_true), len(mu_true)])

cov_sample_array_t = np.zeros([iterations, len(mu_true), len(mu_true)])
pos_mean_array_t = np.zeros([len(mu_true),iterations])
pos_cov_array_t = np.zeros([iterations, len(mu_true), len(mu_true)])


### Start Iteration
for iter_ in range(0,iterations):
    ## Generate Random Draws
    data = ss.multivariate_t.rvs(mu_true, 
                             sigma2_true, 
                             v_true, 
                             200)
    
    ## Estimation based on Historical Estimation
    # Estimate Parameter
    cov_sample = np.cov(data.T)
    # Save Parameter
    cov_sample_array[iter_,:,:] = cov_sample
    # BL Estimation
    pos_sample = bl_original_wrapper(weights, lam, cov_sample, tau, v, eta, c)    
    # Save Posterior
    pos_mean_array[:,iter_] = pos_sample[0].squeeze()
    pos_cov_array[iter_,:,:] = pos_sample[1]

    ## Estimation based on Student t
    # Estimate Parameter
    ml_t = fit_locdisp_mlfp_nu(data)
    ml_t_best = fit_locdisp_mlfp_nu_best(ml_t)
    # Change to Covariance
    cov_t = sig2tocov_t(ml_t_best["sigma2"], ml_t_best["nu"])
    # Save Parameter
    cov_sample_array_t[iter_,:,:] = cov_t
    # BL Estimation
    pos_sample_t = bl_original_wrapper(weights, lam, cov_t, tau, v, eta, c)    
    # Save Posterior
    pos_mean_array_t[:,iter_] = pos_sample_t[0].squeeze()
    pos_cov_array_t[iter_,:,:] = pos_sample_t[1]
    
    
np.std(pos_mean_array,axis=1) 
np.std(pos_mean_array_t,axis=1) 


pos_mean_array.mean(axis=1)
pos_mean_array.mean(axis=1)


### Evaluation
## Cov Parameter
# Frobenius Distance [Ist das ein gutes Measure? wie ist das zu interpretieren]
cov_frob = np.linalg.norm(cov_sample_array - cov_true,axis=0).sum()
cov_frob_t = np.linalg.norm(cov_sample_array_t - cov_true,axis=0).sum()

## BL Parameter

# Cov
bl_pos_mu_frob = np.linalg.norm(pos_mean_array - pos_mu_true,axis=1).sum()
bl_pos_mu_frob_t = np.linalg.norm(pos_mean_array_t - pos_mu_true,axis=1).sum()

bl_pos_cov_frob = np.linalg.norm(pos_cov_array - pos_cov_true,axis=0).sum()
bl_pos_cov_frob_t = np.linalg.norm(pos_cov_array_t - pos_cov_true,axis=0).sum()



# %% Portfolio Optimization
### True Portfolio

def weights_mv(cov_):
    c1 = np.ones(4)@np.linalg.inv(cov_)@np.ones(4)
    return (1/c1) * np.linalg.inv(cov_)@np.ones((4,1))
    
def weights_msn(cov_,mu_):
    c2 = np.ones(4)@np.linalg.inv(cov_)@mu_
    return (1/c2) * np.linalg.inv(cov_)@mu_
        
### Portfolios - True
w_mv_true = weights_mv(pos_cov_true)
w_msn_true = weights_msn(pos_cov_true,pos_mu_true)

### Portfolio Simulation
# Speicher
w_mv_sample = np.zeros([len(mu_true),iterations])
w_msn_sample = np.zeros([len(mu_true),iterations])

w_mv_t = np.zeros([len(mu_true),iterations])
w_msn_t = np.zeros([len(mu_true),iterations])

# Generation of Portfolio Weights
for iter_ in range(0,iterations):
    ### Historical
    w_mv_sample[:,iter_] = weights_mv(pos_cov_array[iter_,:,:]).T
    w_msn_sample[:,iter_] = weights_msn(pos_cov_array[iter_,:,:],
                                        pos_mean_array[:,iter_]).T
    ### Student t
    w_mv_t[:,iter_] = weights_mv(pos_cov_array_t[iter_,:,:]).T
    w_msn_t[:,iter_] = weights_msn(pos_cov_array_t[iter_,:,:],
                                        pos_mean_array_t[:,iter_]).T

# Distance
w_mv_sample_frob = np.linalg.norm(w_mv_sample - w_mv_true,axis=1)
w_mv_t_frob = np.linalg.norm(w_mv_t - w_mv_true,axis=1)

w_msn_sample_frob = np.linalg.norm(w_msn_sample - w_msn_true,axis=1)
w_msn_t_frob = np.linalg.norm(w_msn_t - w_msn_true,axis=1)

### Graphs
## To Do
w_msn_sample_frob

# Standardizierung duech Beobachtung 
(w_msn_t - w_msn_true).mean(axis=1)
a = np.sqrt(np.diag(np.cov(w_msn_t - w_msn_true)))
(w_msn_sample - w_msn_true).mean(axis=1)
