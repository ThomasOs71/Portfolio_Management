# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:14:18 2024

@author: Thomas

s_checklist_montecarlo_step07
This case study performs Step 7 of the "Checklist", namely Ex-ante attribution, to 

    decompose the ex-ante performance and 
    the corresponding satisfaction/risk 
    
    into contributions from exogenous risk factors.

"""
# %% Packages
from numpy import append, array, argsort, busday_count, c_, cumsum, \
                  datetime64, flip, min, sum, sqrt, where, zeros
from sklearn.linear_model import Lasso
from pandas import read_csv
from matplotlib.pyplot import barh, show, xlabel, ylabel

# %% Load data
### Market Risk Drivers (Montecarlo checklist - Step 2)
# import db_riskdrivers_series database
db_riskdrivers_series = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_riskdrivers_series.csv', index_col=0)
x = db_riskdrivers_series.values  
riskdriver_names = array(db_riskdrivers_series.columns)

### Additional variables (Montecarlo checklist - Step 2)
# import db_riskdrivers_tools database 
db_riskdrivers_tools = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_riskdrivers_tools.csv')
d_bar = db_riskdrivers_series.shape[1]  
t_now = datetime64(db_riskdrivers_tools.t_now[0], 'D')  # current time 

### Number of scenarios and future investment horizon (Montecarlo checklist - Step 3)
# import db_riskdrivers_tools database 
db_projection_tools = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_projection_tools.csv')
j_bar = int(db_projection_tools['j_bar'][0])  # number of scenarios
t_hor = datetime64(db_projection_tools['t_hor'][0], 'D')   # investment horizon
m_bar = busday_count(t_now, t_hor)  # number of daily monitoring times 

### Risk drivers forecast (Montecarlo checklist - Step 3)
# import db_projection_riskdrivers database
db_projection_riskdrivers = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_projection_riskdrivers.csv')
x_fcst = db_projection_riskdrivers.values.reshape(j_bar, m_bar + 1, d_bar)

# Scenario Probabilities Forecast (Montecarlo checklist - Step 3)
# import db_scenario_probs database
db_scenprob = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_scenario_probs.csv')
p_scenario = db_scenprob['p_scenario'].values

# Portfolio Ex-Ante Performance (Montecarlo checklist - Step 5)
# import db_exante_perf database
db_exante_perf = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_exante_perf.csv')
y_h = db_exante_perf.values.squeeze()

# Satisfaction Measures (Montecarlo checklist - Step 6)
# import db_quantile_and_satis database
db_quantile_and_satis = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_quantile_and_satis.csv')
c_subquantile = db_quantile_and_satis['c_subquantile'][0]  # confidence level for the sub-quantile
subq_pi = db_quantile_and_satis['subq_pi'][0]  # sub-quantile  
neg_var_pi = db_quantile_and_satis['neg_var_pi'][0]  # negarive variance 

# %% 1. Ex-ante attribution: performance
'''
Perform Top-Down Attribution
    -> Risk Drivers are factors
    -> Perform Lasso Regression to include Factor Selection
'''
########## input (you can change it) ##########
lam = 250000  # parameter for lasso minimization
###############################################
    
z = x_fcst[:, -1, :] - x[-1, :]  # risk driver increments [Diff zwischen Final Period und Now]

### Estimate exposures, intercept and residuals

# Standardize inputs
m_y = p_scenario@y_h
m_z = p_scenario@z

y_p = ((y_h - m_y).T*sqrt(p_scenario)).T
z_p = ((z - m_z).T*sqrt(p_scenario)).T

# Lasso Regression
clf = Lasso(alpha=lam/(2.*y_h.shape[0]), fit_intercept=False)  # lasso minimization
clf.fit(z_p, y_p)  # fit lasso

# Results
beta = clf.coef_  # exposures
alpha = m_y - beta@m_z  # intercept
u = y_h - alpha - z@beta  # residuals

# selected data for relevant risk factors only
ind_relevant_risk_factors = where(beta != 0)[0]
beta = beta[ind_relevant_risk_factors]
z = z[:, ind_relevant_risk_factors]
k_bar = beta.shape[0]  # number of relevant risk factors
f_uz = (c_[u, z], p_scenario)  # joint distribution of residual and risk factors
risk_factors = riskdriver_names[ind_relevant_risk_factors]
print('Number of relevant risk factors: ' + str(k_bar) + " von 104 Insgesamt")

# %% 2. Ex-ante attribution: risk (Read more) 
'''

'''
# Aggregation of Shift and residuals to 0-th factors
z_0 = alpha + u  # 0-th factor
beta_0 = 1  # exposure to the residual

# update
beta_new = append(beta_0, beta)  # exposures
k_new = beta_new.shape[0]  # number of relevant risk factors
z_new = c_[z_0, z]  # risk factors

# sort the scenarios of the risk factors and probabilities
# according to order induced by ex-ante performance scenarios
sort_y = argsort(y_h, axis=0)
p_sort = p_scenario[sort_y]
z_new_sort = z_new[sort_y, :]

# marginal contributions to the sub-quantile satisfaction measure
j_c = min(where(cumsum(p_sort) >= 1 - c_subquantile)) 
w = zeros((j_bar))  # weights
for j in range(j_c):
    w[j] = 1/(1 - c_subquantile)*p_sort[j]
w[j_c] = 1 - sum(w)
subq_contrib = beta_new*(w.T@z_new_sort)  # contributions
pc_subq_contrib = subq_contrib/sum(subq_pi)  #  percentage contributions
print('Percentage contributions to sub-quantile satisfaction measure')
print('-'*61)
for k in range(1, k_bar + 1):
    print('{:31}'.format(risk_factors[k - 1])+':',
          '{: 7.2%}'.format(pc_subq_contrib[k]))
print('{:31}'.format('residual')+':','{: 7.2%}'.format(pc_subq_contrib[0]))
print('')

# marginal contributions to the variance satisfaction measure
e_z = p_scenario@z_new  # expectation 
cv_z_new = ((z_new - e_z).T*p_scenario)@(z_new - e_z)  # covariance 
var_contrib = -beta_new*(cv_z_new@beta_new.T)  # calculate contributions
pc_var_contrib = var_contrib/neg_var_pi  # percentage contributions
print('Percentage contributions to variance satisfaction measure')
print('-'*57)
for k in range(1, k_bar + 1):
    print('{:31}'.format(risk_factors[k - 1])+':', '{: 7.2%}'.format(pc_var_contrib[k]))
print('{:31}'.format('residual')+':','{: 7.2%}'.format(pc_var_contrib[0]))


#################################### plots ####################################
# sub-quantile satisfaction measure
heights = flip(append(subq_pi, append(subq_contrib[1:], subq_contrib[0])))
heights_r = heights*1e-6
lbls = flip(append('total', append(risk_factors, 'residual')))
colors = ['C5'] + ['C0']*k_bar + ['C2']
barh(range(k_new + 1), heights_r, tick_label=lbls, color=colors)
ylabel('Risk driver increments'); xlabel('Sub-quantile (million USD)'); show();

# negative variance satisfaction measure
heights = flip(append(neg_var_pi, append(var_contrib[1:], var_contrib[0])))
colors = ['C5'] + ['C0']*k_bar + ['C2']
barh(range(k_new + 1), heights, tick_label=lbls, color=colors)
xlabel('-Variance'); ylabel('Risk driver increments'); show();
