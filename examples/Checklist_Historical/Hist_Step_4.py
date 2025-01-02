# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:40:25 2024

@author: Thomas


Step 4: Repricing

This case study performs Step 4 of the "Checklist", namely Repricing. The purpose of this step is to
compute the joint distribution of the ex-ante P&L of the instruments over the investment horizon, given the process of the risk drivers and the instrument-specific pricing functions. The pricing functions for the most common instruments are discussed in exact repricing.

Instruments     n       P&L Function
Stocks                      pi_i = v1*(exp(x_thor - x_tnow) - 1)
Call                        Komplexer -> CBS Pricing Function mit in put (exp(x_underlying),y,impl_vola,m,tau)
Puts                        Komplexer -> Über Put - Call Parität

Here we apply the pricing functions for 
    1) the stocks, 
    2) call option and 
    3) put option in our market 

to the paths of the risk drivers obtained in s_checklist_historical_step03b.
"""

# %% Prepare environment
from numpy import array, busday_count, datetime64, exp, log, meshgrid, r_, sqrt, zeros
from scipy.stats import norm as normal
from scipy.interpolate import LinearNDInterpolator
from pandas import DataFrame, read_csv
from seaborn import histplot


# %% Load data 
'''
From database db_valuation_historical created in s_checklist_historical_step01,
we upload:
    1) the values of the instruments 
    2) the option strike price 
    3) option expiry date 
    4) number of stocks 
    5) and current time 
.
'''
## Current Values
db_valuation_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step2\db_valuation_historical.csv', index_col=[0])
n_stocks = int(db_valuation_historical.loc['n_stocks'].values[0])  # number of stocks in market
n_bar = n_stocks + 2  # number of instruments
v_t_now = array(db_valuation_historical.iloc[:n_bar]).astype('float64').squeeze()  # values at current date
k_strk = array(db_valuation_historical.loc['k_strk'].astype('float64'))[0]  # option strike
t_opt_end = datetime64(db_valuation_historical.loc['t_option_end'].values[0], 'D')  # option expiry date
t_now = datetime64(db_valuation_historical.loc['t_now'].values[0], 'D')  # current time
n_options = n_bar - n_stocks  # number of option in market
instrument_names = db_valuation_historical.index[:-4]  # instrument names


## Risk Drivers
'''
From database db_riskdrivers_historical created in s_checklist_historical_step02,
we upload:
    1) the -moneyness grid 
    2) the number of points in the grid, 
    3) the times-to-expiry and 
    4) the level for the (flat, constant) yield curve 
.
'''

db_riskdrivers_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step3\db_riskdrivers_historical.csv', 
                                     index_col=0, parse_dates=True)
l_bar = int(array(db_riskdrivers_historical['l_bar'])[0])  # number of points on m-moneyness grid
m = db_riskdrivers_historical['m'].values[:l_bar]  # m-moneyness parametrization
tau = db_riskdrivers_historical['tau'].dropna().values  # times to expiry
y = array(db_riskdrivers_historical['y'])[0]  # level for yield curve (assumed flat and constant)


## Ex-Ante Distribution (SPD)
'''

From database db_forecast_historical created in s_checklist_historical_step03b,
we upload:
    1) the scenario-probability distribution of the forecast risk drivers {x,p}
    2) the current values of the risk drivers 
    3) the number of scenarios 
    4) the investment horizon 
    
'''
db_forecast_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step4\db_forecast_historical.csv', low_memory=False)
j_bar = int(db_forecast_historical['j_bar'][0])  # number of projection scenarios
t_hor = datetime64(db_forecast_historical['t_hor'][0], 'D')  # investment horizon
m_bar = busday_count(t_now, t_hor)  # number of business days between current date and investment horizon
p_scenario = array(db_forecast_historical['p_scenario'][:j_bar])  # probabilities
x_t_hor = db_forecast_historical.iloc[:, :-3].values.reshape(j_bar, m_bar + 1, -1)  # projected risk driver scenarios
x_t_hor_sigma = x_t_hor[:, :, n_stocks + 1:]


# %% 1. P&L distribution of stocks
'''
For each stock, we apply the stock P&L function to the forecast risk driver scenarios at the horizon.
'''

## P&L scenario at the horizon
pi_t_hor = zeros((j_bar, n_bar))
# lin_ret_t_hor = zeros((j_bar, n_bar))  # Eigene Ergänzung
for n in range(n_stocks):
    # lin_ret_t_hor[:, n] = exp(x_t_hor[:, -1, n] - x_t_hor[:, 0, n]) - 1  # Eigene Ergänzung
    pi_t_hor[:, n] = v_t_now[n]*(exp(x_t_hor[:, -1, n] - x_t_hor[:, 0, n]) - 1)  

#################################### plot ####################################
############# input (you can change it) #############
n_plot = 1  # index of instrument to plot from 1 to 5
#####################################################

# instrument P&L plot
histplot(x=pi_t_hor[:, n_plot-1], bins=30, weights=p_scenario).set(xlabel='P&L',
         title='Ex-ante P&L: '+instrument_names[n_plot-1]);

# %% 2. P&L distribution of call option
'''
We apply the call option P&L function to the forecast risk driver scenarios
for the options at the horizon. 

These include:
    1) the forecast points on the log-volatility surface and 
    2) the log-values of the underlying S&P index.

For each scenario of the option risk drivers, we first compute the values 
at the investment horizon of: 
    1) the (remaining) time to expiry 
    2) the forecast underlying price 
    3) the moneyness and 
    4) the projected log-implied volatility 
.
'''
tau_t_hor = busday_count(t_hor, t_opt_end)/252  # time to expiry at horizon
s_t_hor = exp(x_t_hor[:, -1, n_stocks])  # projected underlying price
m_t_hor = log(s_t_hor/k_strk)/sqrt(tau_t_hor)  # moneyness 
log_sigma_t_hor = x_t_hor_sigma[:, -1, :].reshape(j_bar, -1, l_bar)  # projected log-implied volatility

'''
# We then interpolate the log-implied volatilities.
'''
# interpolated log-implied volatility
grid = list(zip(*[grid.flatten() for grid in meshgrid(*[tau, m])]))  # grid 
log_sigma_interp_t_hor = zeros(j_bar)
for j in range(j_bar):
    # known values
    values_t_hor = log_sigma_t_hor[j, :, :].flatten()  
    # moneyness
    moneyness_t_hor = min(max(m_t_hor[j], min(m)), max(m))
    # log-implied volatility
    log_sigma_interp_t_hor[j] = LinearNDInterpolator(grid, values_t_hor)(*r_[tau_t_hor, moneyness_t_hor])   

'''
# Next we compute the scenarios of the value of the call option at the horizon .
'''
# call option via Black-Scholes-Merton formula
d1 = (moneyness_t_hor + (y + (exp(log_sigma_interp_t_hor)**2)/2)*sqrt(tau_t_hor))/exp(log_sigma_interp_t_hor)
d2 = d1 - exp(log_sigma_interp_t_hor)*sqrt(tau_t_hor)
v_call_t_hor = s_t_hor*(normal.cdf(d1) - exp(-(moneyness_t_hor*sqrt(tau_t_hor) +\
                                             y*tau_t_hor))*normal.cdf(d2)) 

'''
# Finally, we compute the P&L scenarios for the call option at the horizon.
'''
pi_t_hor[:, n_stocks] = v_call_t_hor - v_t_now[n_stocks]  # call option at the horizon

################################# plot #################################
# instrument P&L
histplot(x=pi_t_hor[:, 5], bins=30, weights=p_scenario).set(xlabel='P&L',
         title='Ex-ante P&L: '+instrument_names[5]);

# %% 4. P&L distribution of put option
'''
We compute the P&L scenarios for the put option, 
where the pricing function relies on put-call parity.
'''
# P&L
s_t_now = exp(x_t_hor[:, 0, n_stocks])  # current value of underlying
tau_opt_t_now = busday_count(t_now, t_opt_end)/252  # current time to expiry
pi_t_hor[:, n_stocks + 1] = pi_t_hor[:, n_stocks] - (s_t_hor - s_t_now) +\
                                  k_strk*(exp(-y*tau_t_hor)-exp(-y*tau_opt_t_now))

#################################### plot ####################################
# instrument P&L
histplot(x=pi_t_hor[:, 6], bins=30, weights=p_scenario).set(xlabel='P&L',
         title='Ex-ante P&L: '+instrument_names[6]);

# %% Save data
'''
Save ex-ante P&L scenarios for the stocks and options into database db_pricing_historical.
'''
out = DataFrame({instrument_names[n]: pi_t_hor[:, n] for n in range(n_bar)}) 
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step5\db_pricing_historical.csv', index=False); del out;
