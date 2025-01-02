# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:08:47 2024

@author: Thomas


Step 3b: Multivariate quest and forecasting

This case study performs Step 3b of the "Checklist", namely Multivariate quest and forecasting. 
The purpose of this step is to obtain the distribution of the risk drivers process 
conditional on the current information. f_{X_{tnow -> thor}}, 
given the next-step functions fitted in the previous steps. 

Different approaches to forecasting are discussed extensively within Step 3b. 

Here we pursue the probabilistic historical approach with historical forecasting.

Part 1)
    Here we first construct the distribution of the joint invariants obtained in s_checklist_historical_step03a. 
    We start by specifying the flexible probabilities via joint time and state conditioning.

Part 2):
    We then forecast the risk drivers via historical bootstrapping.
"""

# %% Prepare environment
from numpy import arange, array, busday_offset, cumsum, datetime64, digitize, empty,\
                  exp, interp, isnan, log, max, min, ones, r_, sqrt, sum, zeros
from numpy.random import rand
from pandas import DataFrame, read_csv, Series, to_datetime
from seaborn import histplot, lineplot, scatterplot
from matplotlib.pyplot import bar, gcf, show, xticks, xlabel
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings; warnings.filterwarnings("ignore");
from matplotlib.colors import ListedColormap
from conditional_fp import conditional_fp

# %% Load data 
'''
From database db_valuation_historical created in s_checklist_historical_step01 we upload the current time t_now 
'''
## Current Values
db_valuation_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step2/db_valuation_historical.csv', index_col=[0])
t_now = datetime64(db_valuation_historical.loc['t_now'].values[0], 'D')  # current time


## Risk-Driver
'''
From database db_riskdrivers_historical created in s_checklist_historical_step02,
we upload time series of risk drivers for the stocks and options 
'''

db_riskdrivers_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step3/db_riskdrivers_historical.csv', 
                                     index_col=0, parse_dates=True)
x = db_riskdrivers_historical.values[:, :-5].astype('float64')  # risk drivers
risk_drivers_names = db_riskdrivers_historical.columns.values[:-5]  # risk driver names


## Invariants und Parmeter of Next-Step Function
'''
From database db_univariate_quest_historical created in s_checklist_historical_step03a,
we upload:
    1) the time series of the invariants and 
    2) corresponding dates, 
    3) the fitted parameters of the GARCH next-step models and (a,b,c,mu)
    4) the length of the invariant time series t_bar
    5) current garch element (as input for modell in t_now + 1)
'''
db_univariate_quest_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step3b\db_univariate_quest_historical.csv', 
                                          index_col=0, parse_dates=True)
t_bar = int(db_univariate_quest_historical.t_bar[0])  # length of invariants time series
eps = db_univariate_quest_historical.iloc[:t_bar, :-1].values  # invariant time series
dates = to_datetime(array(db_univariate_quest_historical.index[:t_bar]))  # dates
d_bar = eps.shape[1]  # number of market risk drivers
# indexes of invariants modeled using GARCH(1,1) models
d_garch = [d for d in range(d_bar) if not isnan(db_univariate_quest_historical.iloc[-1, d])]
# parameters of GARCH(1,1) models
a = array(db_univariate_quest_historical.loc['a'].dropna())
b = array(db_univariate_quest_historical.loc['b'].dropna())
c = array(db_univariate_quest_historical.loc['c'].dropna())
mu = array(db_univariate_quest_historical.loc['mu'].dropna())
sigma2_garch_tnow = array(db_univariate_quest_historical.iloc[-1, d_garch])  # current GARCH dispersion parameter

## Additional Data for Flexible Probability Ansatz
'''
From database db_vix we upload VIX compounded returns 
 for the same time period as invariants.
 '''
db_vix_data = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step3b/data_vix.csv', 
                        usecols=['date', 'VIX_close'], index_col=0, parse_dates=True)
# state indicator: VIX index compounded returns
db_vix_data['vix_rets'] = log(db_vix_data).diff()
c_vix = db_vix_data.vix_rets[dates].values  # extract data for analysis period

# %%% 1. Market state indicator 
'''
1) Smoothing:
    We first smooth VIX compounded return realizations via an 
    exponentially weighted moving average
    tau_hl = 21

2) Scoring:
    Then, we perform scoring
    tau_score = 5 * 21

-> Signal Zeitreihe

'''
########## inputs (you can change them) ##########
tau_smooth_hl = 21  # half-life for smoothing
tau_score_hl = 105  # half-life for scoring
z_star = -0.39  # target value 
# TODO: Warum ist das der Krit-Wert?
##################################################

# perform smoothing
c_vix_smooth = zeros(c_vix.shape[0])
for t in range(c_vix.shape[0]):
    # probabilities
    p_w = exp(-log(2)/tau_smooth_hl*arange(0, t + 1))[::-1].reshape(-1)  
    # sum of probabilities
    gamma_w = sum(p_w)
    # risk factor
    c_vix_smooth[t] = (p_w/gamma_w)@c_vix.reshape(c_vix.shape[0], -1)[0:t + 1, :]  

# perform scoring
c_vix_score = zeros(c_vix_smooth.shape[0])
for t in range(1, c_vix_smooth.shape[0]):
    # probabilities
    p_w = exp(-log(2)/tau_score_hl*arange(0, t + 1))[::-1].reshape(-1)
    # sum of probabilities
    gamma_w = sum(p_w)
    # exponentially weighted moving average
    ewma_t_x = (p_w/gamma_w)@c_vix_smooth[0:t + 1].reshape(-1,1)
    # exponentially weighted moving standard deviation
    ewm_sd_t = sqrt(((c_vix_smooth[0:t + 1].reshape(-1,1) - ewma_t_x).T*(p_w/gamma_w))@\
                       (c_vix_smooth[0:t + 1].reshape(-1,1) - ewma_t_x))
    # risk factor
    c_vix_score[t] = (c_vix_smooth[t] - ewma_t_x)/ewm_sd_t  
    
############################## plot ##############################
# VIX and market state
lineplot(x=dates, y=c_vix_score, label='Market state');
lineplot(x=dates, y=z_star*ones(len(dates)), label='Target value');
xticks(dates[arange(0, t_bar - 1, 200)], rotation=45); xlabel('');



# %% 2. Flexible probabilities 
'''
We set the flexible probabilities p_t via joint state and time conditioning, 
1) Using the market state indicator derived from the VIX of with a symmetric 
    range of alpha = 0.7 (crisp) (~inclusion of 70% Data)
2) Prior half-life of 4 years -> tau_hl = 4*252
 years, or in business days
'''

########## inputs (you can change them) ##########
alpha = 0.7  # leeway
tau_prior_hl = 1008  # prior half-life
##################################################

p_tau_hl_prior = exp(-(log(2)/tau_prior_hl)*abs(t_bar - arange(0, t_bar)))  # prior probabilities
p_tau_hl_prior = p_tau_hl_prior/sum(p_tau_hl_prior)  # rescaled probabilities
p_t = conditional_fp(c_vix_score, z_star, alpha, p_tau_hl_prior)  # posterior probabilities

######################## plot ########################
# time and state conditioning flexible probabilities
bar(dates, p_t) 
xticks(dates[arange(0, t_bar - 1, 100)], rotation=45);
# We then compute the effective number of scenarios.
ens = exp(-p_t[p_t > 0]@log(p_t[p_t > 0]))  # effective number of scenarios
print('ens =', int(ens))

'''
We estimate the distribution of the simultaneous invariants via historical 
with flexible probabilities distributions 
'''
############ input (you can change it) #############
i_plot = 0  # modeled invariant to plot from 1 to 96
####################################################

################################################## plots ##################################################
# HFP histogram - chosen invariant distribution
histplot(x=eps[:, i_plot - 1], bins=60, weights=p_t)
xlabel(risk_drivers_names[i_plot - 1] + ' invariant'); show();

# selected invariant observation weights
p_colors = [0 if p_t[t] >= max(p_t) else 1 if p_t[t] <= min(p_t) 
            else interp(p_t[t], array([min(p_t), max(p_t)]), array([1, 0])) for t in range(p_t.shape[0])]
scatterplot(x=array(dates), y=eps[:, i_plot - 1], c=p_colors, cmap='gray')
xticks(dates[arange(0, len(dates), 200)], rotation=45); 
plt.title(db_valuation_historical.index[i_plot])

# %% 3. Forecast risk drivers
'''
We consider a
    m = 1
monitoring time to a horizon of:
    t_hor = t_now + 1 Day    

We generate 
    j = 10.000

Scenarios for the joint invariants usinghistorical bootstrapping, where the 
thresholds are defined using the historical probabilities (from EP of VIX)???

The bootstrapped invariant scenarios are assigned equal probabilities 
    p^(j) = 1/ j 
for each scenario.

Then we feed the scenarios of the invariants into the respective next-step functions, 
thus obtaining j scenarios for the next-step risk drivers, all with equal probability 1/j

TODO:   Okay ich habe wahrscheinlichkeiten für die Ziehungen und die Szenarios?
        Ja, zwei Sorten, weil hier das genereller gehalten wird für m_bar > 1.
        Ansonsten, wenn p_t für Ziehung der Invariants genutzt wird, dann ist p_j = 1/j

'''

########## inputs (you can change them) ##########
m_bar = 1  # number of daily monitoring times 
j_bar = 10000  # number of projection scenarios
##################################################

t_hor = busday_offset(t_now, m_bar)  # investment horizon

# initialize invariant scenarios
eps_proj = zeros((j_bar, d_bar))

# initialize increments and dispersion parameters for GARCH(1,1) 
dx_thor = empty((j_bar, m_bar + 1, d_bar))
sigma2_garch = empty((j_bar, m_bar + 1, d_bar))
dx_thor[:, 0, d_garch] = x[-1, d_garch] - x[-2, d_garch]  # current values of increments
sigma2_garch[:, 0, d_garch] = sigma2_garch_tnow  # current values of dispersion parameters
'''
Brauche die Current Values, da das Garch diese Elemente von Info-Set in Xt+1 sind
'''

# initialize forecast risk drivers
x_thor = empty((j_bar, m_bar + 1, d_bar))
x_thor[:, 0, :] = x[-1, :]  # risk drivers at current time

for m in range(m_bar):
    # bootstrap routine
    # 1. subintervals
    s = r_[array([0]), cumsum(p_t)]  # TODO: Hier haben wir das Problem: Vix verändert sich net! 
    # 2. scenarios
    # a. uniform scenarios
    u = rand(j_bar)  # uniform scenario
    # b. categorical scenarios
    ind = digitize(u , bins=s) - 1
    # c. shock scenarios
    eps_proj = eps[ind, :]
    # apply next-step functions
    for d in range(d_bar):
        # risk drivers modeled with GARCH(1,1): stocks and S&P index
        if d in d_garch:
            # GARCH(1,1) volatility
            sigma2_garch[:, m + 1, d] = c[d] + b[d]*sigma2_garch[:, m, d] + a[d]*(dx_thor[:, m, d] - mu[d])**2
            # risk drivers increments
            dx_thor[:, m + 1, d] = mu[d] + sqrt(sigma2_garch[:, m + 1, d])*eps_proj[:, d]
            # projected risk drivers
            x_thor[:, m + 1, d] = x_thor[:, m, d] + dx_thor[:, m + 1, d]
        # risk drivers modeled as random walk: options
        else:
            # projected risk drivers
            x_thor[:, m + 1, d] = x_thor[:, m, d] + eps_proj[:, d]

p_scenario = ones(j_bar)/j_bar  # projection scenario probabilities
# TODO: Warum sind die alle gleich wahrscheinlich? -> p_t ist für die Invariants.
# Das hier ist standardmäßig auf 1/j gesetzt -> Ermöglicht aber zusätzliche Flexibilität,
# sodass man auch über das setzen von konkreten WSK hier was machen könnte.

#################### input (you can change it) ###################
d_plot = 1  # index of projected risk driver to plot from 1 to 96
##################################################################

################################ plot ################################
# projected risk driver distribution
histplot(x=x_thor[:, m_bar, d_plot - 1], bins=65, weights=p_scenario)
xlabel(risk_drivers_names[d_plot - 1] + ' projected risk driver');

# %% Save data
'''
Save the forecast risk drivers x_{tnow->thor} and bootstrap probabilities p^(j) 
into database db_forecast_historical, as well as the number of scenarios and the investment horizon 
'''

out = DataFrame(x_thor.reshape(-1, d_bar), columns=risk_drivers_names)
out = out.join([DataFrame(p_scenario, columns=['p_scenario']), DataFrame({'j_bar': Series(j_bar), 't_hor': Series(t_hor)})])
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step4/db_forecast_historical.csv', index=None); del out;
