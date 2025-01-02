# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:17:06 2024

@author: Thomas


Step 7b: Ex-ante risk attribution
This case study performs Step 7b of the Checklist, namely Ex-ante risk attribution. 

The purpose of this step is:
    to decompose the satisfaction Satis(Y) , or equivalently the risk, 
    associated with the ex-ante performance, into additive contributions [Satis{betaZ}]_k
    from the factors.

that determine the ex-ante performance attribution.

Here we attribute:
    1) the (negative) variance and 
    2) the sub-quantile (cVaR) of our portfolio 
    
    to the pool of relevant factors selected in s_checklist_historical_step07a, 
    by means of the Euler decomposition for homogeneous measures.
"""

# %% Prepare environment
### Packages
from numpy import isclose, append, array, argsort, average, cov, cumsum, flip, min, sum, where, zeros
from pandas import read_csv
from seaborn import barplot
### Load data

'''
From database db_forecast_historical created in script s_checklist_historical_step03b,
we upload:
    1) the scenario probabilities
'''
db_forecast_historical = read_csv('C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step4/db_forecast_historical.csv', low_memory=False)
p_j = array(db_forecast_historical.p_scenario.dropna())  # probabilities
j_bar = p_j.shape[0]  # number of Montecarlo scenarios

'''
From database db_exante_perf_historical created in script s_checklist_historical_step05b,
we upload:
    1) the ex-ante portfolio return scenarios 
'''
db_exante_perf_historical = read_csv('C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step6\db_exante_perf_historical.csv')
r_w = db_exante_perf_historical.r_w.values  # ex-ante portfolio return scenarios

'''
From database db_quantile_and_satis_historical created in script s_checklist_historical_step06,
we upload:
    1) the sub-quantile satisfaction  
    2) the performance variance and
    3) the confidence level for the sub-quantile  
'''
db_quantile_and_satis_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step7/db_quantile_and_satis_historical.csv')
sub_q_r = db_quantile_and_satis_historical['sub_q_r'][0]  # sub-quantile
neg_v_r = db_quantile_and_satis_historical['neg_v_r'][0]  # negative variance
c = db_quantile_and_satis_historical['c'][0]  # confidence level for sub-quantile

'''
From database db_performance_attribution created in script s_checklist_historical_step07a 
we upload:
    1) the selected risk factors, 
    2) the corresponding exposures 
    3) and the risk factor scenarios 
'''

db_performance_attribution = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step7b/db_performance_attribution.csv')
k_bar = int(db_performance_attribution['k_bar'][0])  # number of exogenous risk factors
risk_factors = db_performance_attribution['risk_factors'].dropna()  # labels of risk factors
beta = db_performance_attribution['beta'].dropna()  # exposures
z = db_performance_attribution['z'].values.reshape(-1, k_bar + 1)  # risk factor scenarios

# %% 1. Marginal contributions to the sub-quantile satisfaction measure 
'''
We attribute the sub-quantile (cVaR) q_{-R}(0.05) computed in 
s_checklist_historical_step06 to the k = 3 selected factors and 
the shifted residual Z0 of the performance attribution model. 

Since the sub-quantile (cVaR) is positive homogeneous, 
we rely on the Euler decomposition.
'''
# TODO / ANmerkung: Über Z0 muss die Summe der Constribution = dem Measure sein
#                   Es bleibt nix übrig

'''
We obtain the factor contributions [q_{-R}(0.05)]_k according to the SPD 
implementation
 
Step 1: Sorting of Datapoints
    First, we sort the performance scenarios in increasing order 
    to obtain {r_w_sort}^J, and induce the same sorting on the probabilities 
    to obtain {p_sort}^J.
'''
r_w_sort = argsort(r_w, axis=0)  # sorting order of ex-ante performance scenarios
p_sort = p_j[r_w_sort]  # sorted probabilities
z_sort = z[r_w_sort, :]  # sorted risk factor scenarios

'''
Step 2: Then we calculate the weights w^{j}
'''

# Determine relevant observations below Alpha
j_c = min(where(cumsum(p_sort) >= 1 - c))  

# calculate weights corresponding to CVaR(alpha)
w = zeros((j_bar))
for j in range(j_c):
    w[j] = 1/(1 - c)*p_sort[j]
w[j_c] = 1 - sum(w)

'''
Using these weights, we compute the factor contributions [q_{-R}(0.05)]_k and
convert these into percentage contributions.
'''

sub_q_contrib = beta*(w.T@z_sort)  # factor contributions
pc_sub_q_contrib = sub_q_contrib/sum(sub_q_r)  # percentage contributions

assert isclose(sub_q_r,sum(sub_q_contrib))

###################################### plot ######################################
print('\n'.join(f"{factor}: {val*100:.2f}%"
      for factor, val in zip(append('Residual', risk_factors), pc_sub_q_contrib)))

# sub-quantile satisfaction measure
heights = append(append(sub_q_contrib[0], sub_q_contrib[1:]), sub_q_r)
lbls = append('Residual', append(risk_factors, 'Total'))
barplot(x=heights, y=range(0, k_bar+2), hue=lbls, orient='y').set(yticks=[]);

# %% 2. Marginal contributions to the variance satisfaction measure 
'''
We attribute the variance satisfaction measure -V{R} computed in 
s_checklist_historical_step06 to the k = 3 selected factors and 
the shifted residual Z0 of the performance attribution model. 

Since the variance satisfaction measure is positive homogeneous, 
we rely on the Euler decomposition.

Step 1: Compute the covariance by means of the scenario-probability recipe.
'''
cv_z = cov(z.T, aweights=p_j, bias=True)  # covariance

'''
Step 2: Compute the factor contributions [-V{R}]_k, and 
        convert these into percentage contributions.
'''

v_r_contrib = -beta*(cv_z@beta.T)  # calculate contributions
pc_v_r_contrib = v_r_contrib/neg_v_r  # percentage contributions

##################################### plot #####################################
print('\n'.join(f"{factor}: {val*100:.2f}%"
      for factor, val in zip(append('Residual', risk_factors), pc_v_r_contrib)))
# negative variance satisfaction measure
heights = append(append(v_r_contrib[0], v_r_contrib[1:]), neg_v_r)
barplot(x=heights, y=range(0, k_bar+2), hue=lbls, orient='y').set(yticks=[])
