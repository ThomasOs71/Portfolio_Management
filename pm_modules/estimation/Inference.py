# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 19:02:05 2024

@author: Thomas

Offene Punkte:
    1) BL -> Individual Confidence Levels -> NivhtSymmetrische Sigma^2 View -> Einfluss auf sonst was?
        2) Lösung für Lambda
        3) Tau anders machen? -> siehe 4
        4) Statt 1/tau -> Vorgefertige Covariance of Prior Mean über Estimation Theory bzw. MC? 
           im Code sollte das einfach machbar sein über Conditioning: https://www.arpm.co/lab/bl-post-dib.html#x1259-310400070.3
           5) COde nochmal umändern um die Möglichkeit einer eigenen Cov_Prior zu geben! -> s. Auch Roncalli

    2) NIW -> View in Sigma_Sample / Mu_Sample einfügen
            -> Das geht noch allgemeiner indem die Varianz von Mu_Prior und Sigma_Prior separat baubar sind
            -> Dsa sollte noch mit qualitativen Views verbessert werden, sonst ergeben sich extremgewichte
            -> Am besten über Entropy... oder hier irgendwie mit BL Ansätzen um Views aufzunehmen
            -> Optimize over Parameter gamma to find optimal Gamma for the usecase!
    
"""
from numpy import arange, array, diag, exp, eye, log, sqrt, sum, zeros
from numpy.linalg import eig, solve, inv
from pandas import read_csv

import numpy as np
from IPython.display import display, Math

from pm_modules.support import *

# %% Entropy Pooling
'''
Generalization des Bayesian Updating
'''


def entropy_pooling():
    return None


def entropy_pooling_interface():
    return None


def entropy_pooling_sequential():
    return None    
    

def relative_entropy(p):
    p = np.zeros((10,1))
    p[1,0] = 5

    a = p == 0
    p[a] = 2
    
    # wie gehe ich mit den nullern um? EInfach rauskicken?
    
    return None


def effective_number_of_scenarios(p_):
    return np.exp(-relative_entopy(p_))


def entropy_pooling_agg_unc_additive():
    return None


def entropy_pooling_agg_unc_multiplicative():
    return None


def entropy_pooling_check_posterior_moments()
    return None


# %% Bayesian Approaches
'''
Bayesian Approaches
1) Normal Inverse Wishart
2) Black-Litterman
'''
### Normal-Inverse Wishart
def niw_posterior(mu_pri: "np.array(n,1)",
                  sigma2_pri: "np.array(n,n)",
                  mu_sample: "np.array(n,1)",
                  sigma2_sample: "np.array(n,n)",
                  t_pri: int,
                  v_pri: int,
                  t_bar: int) -> dict:
    '''
    Ziel

    Input:
        mu_pri : Mean_Prior"np.array(n,1)"
        
        sigma2_pri : "np.array(n,n)"
            DESCRIPTION.
            mu_sample : "np.array(n,1)"
            DESCRIPTION.
            sigma2_sample : "np.array(n,n)"
            DESCRIPTION.
    t_pri : int
            DESCRIPTION.
            v_pri : int
                DESCRIPTION.
                t_bar : int
                DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    '''
    # Confidence Prior
    g_mu = (t_pri)/(t_pri + t_bar)  # Confidence Mu Prior 
    g_sigma2 = (v_pri)/(v_pri + t_bar)  # Confidence Sigma2 Prior              
    
    # Mu_Posterior                 
    mu_pos = g_mu * mu_pri + (1 - g_mu) * mu_sample
    
    # Sigma_Posterior
    sigma2_pos = g_sigma2 * sigma2_pri + (1 - g_sigma2) * sigma2_sample  \
        + (1-g_mu) * (1- g_sigma2) * (mu_pri - mu_sample)@(mu_pri - mu_sample).T 
        
    # Confidence Posterior
    t_pos = t_pri + t_bar
    v_pos = v_pri + t_bar

    return dict({"mu_pos": mu_pos, 
                "sigma2_pos":sigma2_pos,
                 "t_pos":t_pos,
                 "v_pos":v_pos})
    
def niw_predictive_posterior(mu_pos,
                             sigma2_pos,
                             v_pos,
                             t_pos):
    '''
    
    '''
    # Mu der Predictive Posterior Distribution für r_t
    r_mu_pred_pos = mu_pos
    # Sigma_2 der Predictive Posterior Distribution für r_t
    r_sigma2_pred_pos = ((t_pos + 1) / t_pos) * sigma2_pos
    '''
    Achtung: Das hier ist ungleich Covariance, weil Pred. Pos. Distribution Student t ist
    '''
    r_nu_pred_pos = v_pos
    
    return dict({"r_mu": r_mu_pred_pos, 
                "r_sigma2":r_sigma2_pred_pos,
                "r_nu":r_nu_pred_pos})


def niw_bayesian_allocation(prior,
                            sample,
                            t_pri,
                            v_pri,
                            t_bar,
                            lam):
    '''
    Aktuell ohne Weight Einschränkung
    '''
    # Extract
    mu_pri = prior["mu_pri"]
    sigma2_pri = prior["sigma2_pri"]
    mu_sample = sample["mu_sample"]
    sigma2_sample = sample["sigma2_sample"]
    
    # Calculate Weights (Determine amount of Shrinkage towards Prior)
    g_mu = (t_pri)/(t_pri + t_bar)
    g_sigma2 = (v_pri)/(v_pri + t_bar)
    k = lam/(2*(1 - lam)) * (t_bar - 2 * ( 1 - g_sigma2))/(t_bar + (1-g_mu))

    kappa = lam/(2*(1 - lam))*(t_bar - 2*(1 - g_sigma2))/(t_bar + (1 - g_mu))
    # Define Parameter (not fully equal to Parameter of Pred. Posterior Dist)
    mu = g_mu * mu_pri + (1 - g_mu) * mu_sample
    
    # Sigma_Posterior
    sigma2 = g_sigma2 * sigma2_pri + (1 - g_sigma2) * sigma2_sample  \
        + (1-g_mu) * (1- g_sigma2) * (mu_pri - mu_sample)@(mu_pri - mu_sample).T  
    
    w_bayes = k * np.linalg.inv(sigma2) @ mu
    
    return w_bayes
    








### Black Litterman
'''
Tests für k = 1 und k > 1
'''

def bl_implied_equilibirum_returns(cov_hat_r: "np.array(n,n)",
                                   weights: "np.array(n,1)",
                                   lam: int = 1,
                                   cov_bm: "np.array(n,1)" = None):
    '''
    Aim:
        Calculates the Implied Equilibirum Returns
    
    Input:
        cov: Covariance, np.array(n,n)
        weights: Weights of Instruments in Base Allocation (Strategic ALlocation)
        lam: Risk Aversion Parameter
        cov_bm: Covariance Returns der Instrumente
        
    Return:
        
    
    Notes:
        - Implied Returns sind die aus dem Risiko abgeleiteten "Erw" Renditen einer
          vorgegebenen "strategischen" Allokation
        - Über die Covariance der BM erhält das eine "Excess" Exp Return Interpretation (? Korrekt ?)
        - Covariance gerne mit Exp. Decay (s. Original Papier)
        - Annahmen: Basiert auf MV Optimisation (s. Herleitung)
    '''
    if cov_bm is None:
        ret_eq = lam*cov_hat_r@weights.squeeze()
    
    if cov_bm is not None:
        ret_eq = lam * (cov_hat_r@weights.squeeze() - cov_bm)
    
    return ret_eq.reshape(len(ret_eq),1)


def bl_prior_distribution(mu_r_equil: "np.array(n,1)",
                          cov_hat_r: "np.array(n,n)",
                          tau: "int") -> dict:
    '''
    Aim:
        Returns the Parameter of the Prior Distribution (of the Mean) for BL
        
    Input:
        mu_r_equil: np.array, shape(n,1), (Equilibrium) Exp. Returns
        cov_hat_r: Est. Covariance Matrix of Returns
        tau: Confidence in Prior
        
    Return:
        tuple: (Expected Value for Prior Mean, Uncertainty on the Prior Mean)
    '''
    
    ### Parameters of Prior Expected Value Distribution M
    mu_m_pri = mu_r_equil  # Expected Value for Prior Mean
    cov_m_pri = (1/tau)*cov_hat_r  # Uncertainty on the Prior Mean    

    return (mu_m_pri.reshape(len(mu_r_equil),1), cov_m_pri)



def bl_prior_predictive_performance_distribution(mu_r_equil: "np.array(n,1)",
                                                 cov_hat_r: "np.array(n,n)",
                                                 tau: "int"):
    '''
    Ziel:
        Prior Predicitve Performance Distribution Simulation
        
    Input:
        mu_r_equil: Equilibirum Returns
        
    Return
        prior_pred_dist: tuple, (Expected Return for Prior Predictive Model, Covariance of Returns for Prior Predictive Model)
        
    Notes:
         Auch bekannt als "Equilibirum Performance Model"
        - Die Prior Predictive Performance Distribution hat eine höhere Covariance zur Berücksichtigung der Unsicherheit
          in der Schätzung des Exp. Returns
   
    '''    
    ### Parameters of Prior Expected Value Distribution M
    mu_prior = bl_prior_distribution(mu_r_equil, cov_hat_r, tau)
    mu_m_pri = mu_prior[0]
    cov_m_pri = mu_prior[1]
    
    ### Parameters of Prior Performance Model for Returns     
    mu_pri_pred_r = mu_r_equil  # Expected Return of Prior Returns
    cov_pri_pred_r = cov_hat_r + cov_m_pri  # Covariance of Prior Returns

    return (mu_pri_pred_r, cov_pri_pred_r)    



def bl_views(v: "np.array(k,n)",
             eta: "np.array(k,1)",
             c: "np.array(k,1)",
             prior_mean_dist: tuple,
             prior_pred_dist: tuple) -> tuple :
    '''
    Ziel:
        - Erstellung der aktiven Views und der zugehörigen Unsicherheit
        - Qualitative Views relativ zum Prior Mean Return. 
          Darüber Gewichtung mit Standard Deviation
        
    Input
        - v: Matrix mit Views [Views in Rows], np.array(k,n)
        - eta: Qualitative Views mit Integer [-3,3], np.array(k,1)
        - c: Confidence in jeweiligen View, float [0,1], np.array(k,1)
        - prior_mean_dist: tuple (prior_mean, prior_uncertainty)
        - prior_pred_dist: tuple(prior_pred_r_mean, prior_pred_r_cov)
        
    Return
        - tuple: Active Views, Uncertainty in active View
        - mu_m_pri ist in BL identisch zu den equilibrium returns (mu_r_equil),
          aus Gründen der Klarheit so benamt.
    '''
    ### Reset Dimensions
    if eta.ndim > 1:
        eta = eta.squeeze()
    
    ### Extract prior_mean_dist
    mu_m_pri = prior_mean_dist[0]
    cov_m_pri = prior_mean_dist[1]
    
    ### Extract prior_pred_dist
    cov_pri_pred_r = prior_pred_dist[1]
    
    ### Aktive Views
    views = v @ mu_m_pri + (eta * sqrt(diag(v @ cov_pri_pred_r @ v.T))).reshape(len(eta),1)  
       
    
    '''
    Das hier wird riesig..., aber das sollte richtig sein!
    '''
    # cv_pri_pred_r = sigma2_hat_r + sigma2_m_pri [Beide QUellen von Unsicherheit: Allgemeine + Schätzunsicherheit]

    ### Uncertainty in View     
    sigma2_view = ((1 - c)/c) * (v @ cov_m_pri @ v.T)  # covariance matrix
    '''
    Bei verschiedenen Confidences...Symmetrie der Covariance weg! [Schlimm? Lässt sich das umgehen? Effekte für Interpretation Eff_Rank]
    '''
    return (views.reshape(-1,1)), np.atleast_2d(sigma2_view)



def bl_check_views(v:"np.array(k,n)",
                   sigma2_view: "np.array(k,k)",
                   prior_mean_dist: "tuple(mean,cov)") -> None:
    '''
    Ziel:
        Umfasst die Prüfung von:
            1) Anzahl der Views 
            2) Effektice Rang Testing
    -> Both only generate Warnings
    
    Input:
        v: Pick-Matrix for Views, np.array(k,n)
        sigma2_view: Uncertainty Regarding Views
        prior_mean_dist: Parameters of the Prior (Mean) Distribution
                         [0] -> Mean, [1] -> Covariance
        
    Returns:
        None
    
    Note:
        - Effective Rank prüft die Statistische Unabhängigkeit der Views und den vollen Rank der Views.
          Bei Ablehnung ist das View-Set zu überprüfen.
    '''
    
    ### Check Number of Views vs. Number of Variables    
    number_views = v.shape[0]
    number_variables = v.shape[1]
    if number_views > number_variables:
        print(f"{number_views} > {number_variables}: Die Anzahl der Views übersteigt die Anzahl der Variablen")
    
    ### Effective Rank of Views
    # Function
    def eff_rank(s2):
        lam, _ = eig(s2)
        w = lam/sum(lam)

        return np.exp(-w@np.log(w))  # Entropy of (relative) Eigenvalues
    
    # Unpack prior_mean_dist
    cov_m_pri = prior_mean_dist[1]
    
    # (Unconditional) Covariance of View Model
    cv_i = v@cov_m_pri@v.T + sigma2_view
    
    # Erstellung der Correlation / Entfernen der Volatilitäten
    cr_i = np.diag(1 / np.sqrt(np.diag(cv_i)))@cv_i@np.diag(1 / np.sqrt(np.diag(cv_i)))
    
    # Effective Rank Calculation
    eff_rank_views = float(eff_rank(cr_i))

    # Krit-Value
    alpha = 0.5 * number_views
    
    # Decision
    if eff_rank_views > alpha:
        print(f"Effective Rank {np.round(eff_rank_views,2)} > {np.round(alpha,2)}: Views in Ordnung")
    else:
        print(f"Effective Rank {np.round(eff_rank_views,2)} < {np.round(alpha,2)}: Views zu extrem / nicht ausreichend unabhängig voneinander")

    return None



def bl_view_explanation(v: "np.array(k,n)",
                        views: "np.array(k,1)",
                        sigma2_view: "np.array(k,k)",
                        name_variables = None) -> list:
    '''
    Beschreibung, was die Views aussagen
    '''
    # Variables
    number_view = v.shape[0]
    number_variables = v.shape[1]

    # Give Default Name, wenn name_variables not defined
    if name_variables is None:
        name_variables = np.arange(1,number_variables+1)

    # Volatility / Uncertainty of Views     
    uncertainty_view = np.sqrt(np.diag(sigma2_view))
    
    ### Speicher    
    view_explanation = []
    ### Absolute View
    # Check for Absolute VIew
    index_abs_view = np.sum(v,axis=1) == 1
    
    abs_views = v[index_abs_view]
    
    # Absolute Views
    for i,k in enumerate(abs_views):
        if np.sum(k) == 1:
            variable_sel = np.argmax(k)
            view_explanation.append(f'The Expected Return of Instrument {name_variables[variable_sel]} is {np.round(views[i],2)} % with Uncertainty {np.round(uncertainty_view[i],2)} %')
        if np.sum(k) == 0:
            variable_out = np.argmax(k)
            variable_under = np.argmin(k)
            view_explanation.append(f'The Expected Outperformance of Instrument {name_variables[variable_out]} vs. {name_variables[variable_under]} is {np.round(views[i],2)} % with Uncertainty {np.round(uncertainty_view[i],2)} %')

    return view_explanation



def bl_posterior_distribution(prior_mean_dist: "tuple(mean,cov)",
                              views: "np.array(k,1)",
                              sigma2_view: "np.array(k,k)",
                              v: "np.array(k,n)",
                              cov_hat_r: "np.array(n,n)",
                              tau: int):
    '''
    Ziel:
        Berechnung der Parameter für die Verteilung des Posterior (Means)
        
    Input:
        prior_mean_dist: Parameter der Prior Mean Distribution, tuple(mean,cov)
        views: Active Views, np.array(k,1)
        sigma2_view: Uncertainty in Views, np.array(k,k)
        v: View Pick Matrix, np.array(k,n)
        cov_hat_r: Est. Return Covariance, np.array(n,n)
        tau: Confidence in Prior
    
    Return:
        pos_mean_dist: tuple, Parameter der Posterior (Mean) Distribution, tuple(mean,cov)
        
    '''
    # Extract Parameter for Prior Mean Distribution
    mu_m_pri = prior_mean_dist[0]

    # posterior expectation
    mu_m_pos = mu_m_pri + (1/tau)*cov_hat_r@v.T@solve((1/tau)*v@cov_hat_r@v.T + sigma2_view,views - (v@mu_m_pri).reshape(len(v),1))

    # posterior covarinace
    cov_m_pos = (1/tau)*cov_hat_r - (1/tau)*cov_hat_r@v.T@solve((1/tau)*v@cov_hat_r@v.T + sigma2_view,(1/tau)*v@cov_hat_r)

    return (mu_m_pos, cov_m_pos)



def bl_posterior_predictive_performance_distribution(pos_mean_dist: "tuple(mean,cov)",
                                                     cov_hat_r: "np.array(n,n)") -> tuple:
    '''
    Aim:
        Berechnung der Parameter der Posterior Predictive Return / Performance Distribution
        
    Input:
        pos_mean_dist: tuple(mean,cov) enthält die Parameter der Posterior (Mean) Distribution
        cov_hat_r: Est. Covariance Matrix of Returns
    
    Return:
        pos_pred_dist: tuple, Parameter der Posterior Predictive Return Distribution 
        
    '''                                                     
    e_pos_pred = pos_mean_dist[0]  # posterior predictive expectation
    # print('e_pos_pred =', e_pos_pred.round(5))

    cv_pos_pred = pos_mean_dist[1] + cov_hat_r  # posterior predictive covariance
    # print('cv_pos_pred \n', cv_pos_pred.round(5))

    return (e_pos_pred, cv_pos_pred)



def bl_view_agressiveness(pos_pred_dist,
                          pri_pred_dist,
                          v) -> "np.array":
    
    ### Extract
    mu_pos_pred = pos_pred_dist[0]
    mu_pri_pred = pri_pred_dist[0]
    cv_pri_pred = pri_pred_dist[1]   
    
    gradient_d_kl = v@(mu_pos_pred - mu_pri_pred)@inv(v@cv_pri_pred@v.T)  # gradient of relative entropy
    return



def bl_implied_equilibirum_returns_mre():
    return None



def bl_original_wrapper(weights_,
                        lam_,
                        cov_,
                        tau_,
                        v_,
                        eta_,
                        c_):
    
    
    # Implied Returns
    
    mu_r_equil_ = bl_implied_equilibirum_returns(cov_,  
                                                 weights_,
                                                 lam = lam_)
    
    # Prior
    pri_mu_ = bl_prior_distribution(mu_r_equil_,
                                   cov_,
                                   tau_)
    # Prior Pred
    pri_pred_ = bl_prior_predictive_performance_distribution(mu_r_equil_,
                                                            cov_,
                                                            tau_)
    
    # Views
    # View Distribution
    views_ = bl_views(v_,
                      eta_,
                      c_,
                      pri_mu_,
                      pri_pred_)

    # Check
    '''
    bl_check_views(v_, 
                   views_[1], 
                   pri_mu_)
    '''
    
    ## Posterior
    # Posterior Mean Distribution
    pos_mu_ = bl_posterior_distribution(pri_mu_, 
                                               views_[0], 
                                               views_[1], 
                                               v_, 
                                               cov_, 
                                               tau_)

    ## Predictive Distribution
    pos_pred_ = bl_posterior_predictive_performance_distribution(pos_mu_, 
                                                                 cov_)
    
    return pos_pred_
    










'''
Test Daten


### Data
cov = np.array([[1,0.2,0.1],
                [0.2,1,0.1],
                [0.1,0.1,1]])

w = np.array([0.4,0.2,0.4]).reshape(-1,1)
mu_r = implied_equilibirum_returns(cov, w)
tau = 50

prior_mean_dist = bl_prior_distribution(mu_r,cov,50)
prior_pred_dist = bl_prior_predictive_performance_distribution(mu_r, cov, 50)

v = np.array([[1,0,0],
              [0,0,1]])

eta = np.array([1,-1])

c = np.array([0.4,0.6]).reshape(-1,1)








c = np.array([0.3,0.5,0.8]).reshape(-1,1)

d = (1 -c)/c



'''








'''
how to set the risk aversion parameter?
Roncalli...per Sharpe Ratio... aber woher nehmen wir die?
'''

# %% Entopy Pooling

### Analytical






















# %% Partial Sample Regression
