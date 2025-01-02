# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:16:58 2024

@author: Thomas


Step 7a: Ex-ante performance attribution
This case study performs Step 7a of the Checklist, namely:
        Ex-ante performance attribution. 
        
The purpose of this step is to compute 
    1) the exposures beta of the ex-ante performance Y to a set of risk factors based on a LFM,
    2) and to compute the joint distribution of the factors and the residuals f_{U,Z}
    3) and the Residuals U
    
starting from the joint distribution of the ex-ante performance and the factors. 

Bottom-up and top-down methods to perform the ex-ante performance attribution 
are discussed extensively within Ex-ante performance attribution.

Here we use the top-down factors on demand approach to linearly attribute 
the portfolio ex-ante R return to the linear returns of the pool of the 
3 most relevant instruments rel(Z())

    rel(Z) = (R_{rel(1)},
              R_{rel(2)},
              R_{rel(r)})

where rel(n) denotes the n-th jointly most relevant factor. 
We apply lasso regression to select the relevant pool of factors.

The joint distribution of the ex-ante portfolio return and the factors f_{y,z} 
is readily available in its scenario-probability representation.
"""

# %% Prepare environment
### Packages
from numpy import array, average, append, c_, where, sqrt
from pandas import DataFrame, read_csv, Series
from sklearn.linear_model import Lasso

### Load data
'''
From database db_forecast_historical created in script s_checklist_historical_step03b,
we upload:
    1) the scenario probabilities 
'''

db_forecast_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step4\db_forecast_historical.csv', low_memory=False)
p_j = array(db_forecast_historical.p_scenario.dropna())  # probabilities

'''
From database db_exante_perf_historical created in script s_checklist_historical_step05b,
we upload:
    1) the ex-ante portfolio return scenarios 
    2) and the instrument return scenarios.
 '''
 
db_exante_perf_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step6\db_exante_perf_historical.csv')
r_w = db_exante_perf_historical.r_w.values  # ex-ante portfolio return scenarios
r = db_exante_perf_historical.iloc[:, 1:].values  # instrument return scenarios

# %% 1. Extract factors 
'''
We set the risk factor scenarios as the instrument return scenarios:
    
    z^(j) = r^(j)

The probabilities, inherited from the scenario-probability distribution of the 
instrument return scenarios, are all equal p^(j) = 1 / J
'''

z = r  # instrument returns

# %% 2. Compute exposures using lasso regression 
'''
We estimate the top-down exposures via lasso regression.

Step 1: Standardization
    We first standardize each scenario for both the performance and factors 
    by subtracting the mean and scaling by the square root of the probability.
'''

m_y = average(r_w, weights=p_j)  # mean performance
m_z = average(z, weights=p_j, axis=0)  # factor means
y_p = sqrt(p_j)*(r_w - m_y)  # standardized performance
z_p = sqrt(p_j).reshape(-1, 1)*(z - m_z)  # standardized factors

'''
Step 2: Lasso Regression
    We then perform lasso regression with lambda = 3*10^(-3)

    We then:
        1) extract the exposures beta and 
        2) compute the shift term, based on the zero-center constraint on the residual, 
        3) and the residual scenarios 
'''

######### input (you can change it) #########
lam = 3e-3  # parameter for lasso regression
#############################################

clf = Lasso(alpha=lam/(2.*r_w.shape[0]), fit_intercept=False)  # lasso regression
clf.fit(z_p, y_p)  # fit lasso
beta = clf.coef_  # exposures
alpha = m_y - beta@m_z  # shift
u = r_w - alpha - z@beta  # residuals


'''
Step 3: Faktor Selektion
    We select the k = 3 most relevant factors (rel (Z)) by selecting those for 
    which beta != 0, as described in Lasso Regression
'''

# selected data for relevant risk factors only
ind_relevant_risk_factors = where(beta != 0)[0]
beta = beta[ind_relevant_risk_factors]
rel_z = z[:, ind_relevant_risk_factors]
k_bar = beta.shape[0]  # number of relevant risk factors
risk_factors = db_exante_perf_historical.iloc[:, 1:].columns[ind_relevant_risk_factors]
[print(f'{risk_factors[k]} returns beta_{k+1} = {beta[k].round(4)}') for k in range(k_bar)];

# %% 3. Compute joint distribution 
'''
We first compute the joint scenario-probability distribution of the factors and the residual 
by juxtaposing the respective scenarios. -> Its SPD, therefore we have the Scenarios
'''
f_uz = (c_[u, rel_z], p_j)  # joint distribution of residual and risk factors

'''
Then we map the residual into the 0-th factor (Z0), 
and compute the corresponding exposure 
.
'''
z_0 = alpha + u  # map residuals to 0-th factor
beta_0 = 1  # exposure to residual
beta = append(beta_0, beta)  # updated exposures
'''
Accordingly, we map the scenarios of the residual {u}^{j} into scenarios 
of the factor {z0} to obtain the joint scenario-probability distribution of the factors.
'''
rel_z = c_[z_0, rel_z]  # joint scenarios of risk factors

# %% Save data 
'''
Save :
    1) the number of selected risk factors k
    2) and their labels, 
    3) the corresponding exposures 
    4) and the risk factor scenarios 
into database db_performance_attribution.
'''
out = DataFrame({'k_bar': Series(k_bar), 'risk_factors': Series(risk_factors), 'beta': Series(beta), 'z': Series(rel_z.reshape(-1))})
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step7b\db_performance_attribution.csv', index=False); del out;
