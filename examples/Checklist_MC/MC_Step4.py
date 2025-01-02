# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:19:48 2024

@author: Thomas

s_checklist_montecarlo_step04
This case study performs Step 4 of the "Checklist", namely Repricing, 

    to compute the joint distribution of the ex-ante P&L of the instruments over the investment horizon 
    
"""

# %% Prepare Environment

### Packages
from numpy import any, array, arange, busday_count, busday_offset, cumprod, datetime64, exp, \
                  full, flip, log, meshgrid, ones, prod, r_, sum, sqrt, tile, where, zeros
from scipy.stats import norm as normal
from scipy import interpolate
from pandas import bdate_range, date_range, DataFrame, read_csv
from seaborn import histplot
from bond_value import bond_value

# %% Load data

### Market Risk Drivers Data (Montecarlo checklist - Step 2)

## Risk Drivers (Step 2)
db_riskdrivers_series = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_riskdrivers_series.csv', index_col=0)
x = db_riskdrivers_series.values  # market risk drivers 
d_bar = db_riskdrivers_series.shape[1]  # number of risk drivers 

## Additional variables (Step 2)
db_riskdrivers_tools = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_riskdrivers_tools.csv', parse_dates=True)
t_now = datetime64(db_riskdrivers_tools.t_now[0], 'D')  # current date 
n_stocks = len(db_riskdrivers_tools.stocks_names.dropna())  # number of stocks 
n_bonds = int(db_riskdrivers_tools.n_bonds.iloc[0])  # number of corporate bonds 
n_bar = n_stocks + n_bonds + 3  # number of instruments 

# options informations 
t_option_end = datetime64(db_riskdrivers_tools['t_option_end'][0], 'D')  # expiry date of options
k_strk = db_riskdrivers_tools['k_strk'][0]  # strike value of options on S&P 500 (US dollars)
l_bar = int(db_riskdrivers_tools.l_bar.iloc[0])  # number of points on m-moneyness grid
m = db_riskdrivers_tools['m'].values[:l_bar]  # m-moneyness parametrization
tau = db_riskdrivers_tools['tau'].values  # times to expiry

# bonds informations  
y = db_riskdrivers_tools['y'][0]  # risk-free rate
t_end_ge = datetime64(db_riskdrivers_tools['t_end_ge'][0], 'D')  # maturity GE
t_end_jpm = datetime64(db_riskdrivers_tools['t_end_jpm'][0], 'D')  # maturity JPM
coupon_ge = db_riskdrivers_tools['coupon_ge'][0]  # coupon rate GE
coupon_jpm = db_riskdrivers_tools['coupon_jpm'][0]  # coupon rate JPM
c_bar = int(db_riskdrivers_tools.c_bar.iloc[0])  # number of ratings classes

# index of risk drivers for options and corporate bonds
risk_drivers_names = array(db_riskdrivers_series.columns).squeeze()
idx_options = [i for i, element in enumerate(risk_drivers_names) if 'option' in element]  # index options
idx_ge_bond = [i for i, element in enumerate(risk_drivers_names) if 'ge_bond' in element]  # index bond GE 
idx_jpm_bond = [i for i, element in enumerate(risk_drivers_names) if 'jpm_bond' in element]  # index bond JPM


## Current Values (Step 1)
# values of stocks, S&P 500, options and bonds at current time (Montecarlo checklist - Step 2)
db_v_tnow = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_v_tnow.csv')
v_t_now = db_v_tnow.values[0]  # values at current date



### Market Estimation and Projection (Montecarlo checklist - Step 3)

## Additional informations (Montecarlo checklist - Step 3)
db_projection_tools = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_projection_tools.csv')
j_bar = int(db_projection_tools['j_bar'][0])  # number of scenarios 
t_hor = datetime64(db_projection_tools['t_hor'][0], 'D')  # investment horizon
m_bar = busday_count(t_now, t_hor)  # number of business days between current date and invesment horizon

## Risk Drivers Forecast (Montecarlo checklist - Step 3)
db_projection_riskdrivers = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_projection_riskdrivers.csv')
x_fcst = db_projection_riskdrivers.values.reshape(j_bar, m_bar + 1, d_bar)  # risk driver forecast scenarios

## Scenario probabilities forecast (Montecarlo checklist - Step 3)
db_scenario_probs = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_scenario_probs.csv')
p_scenario = db_scenario_probs['p_scenario'].values 


### Credit Estimation and Projection (Montecarlo checklist - Step 3)
# credit ratings forecast (Montecarlo checklist - Step 3)
db_projection_ratings = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_projection_ratings.csv')


# %% 1. P&L of the stocks after one day and at the horizon 
'''
PI = v_t*(exp(x_u - v_t) - 1)
'''
# P&L of stocks
pi_t_now_t_hor = zeros((j_bar, n_bar))
pi_one_day = zeros((j_bar, n_bar))
for n in range(n_stocks):
    # after one day
    pi_one_day[:, n] = v_t_now[n]*(exp(x_fcst[:, 1, n] - x[-1, n]) - 1)  # v_tnow * (exp(Delta RD) - 1), RD ist in logs, v_t in Prices
    # at the horizon
    pi_t_now_t_hor[:, n] = v_t_now[n]*(exp(x_fcst[:, -1, n] - x[-1, n]) - 1)  

# %% 2. P&L of the S&P 500 index after one day and at the horizon 
# P&L of S&P 500 index
pi_one_day[:, n_stocks] = v_t_now[n_stocks]*(exp(x_fcst[:, 1, n_stocks] - x[-1, n_stocks]) - 1)  # after one day
pi_t_now_t_hor[:, n_stocks] = v_t_now[n_stocks]*(exp(x_fcst[:, -1, n_stocks] - x[-1, n_stocks]) - 1)  # at the horizon

mean = pi_t_now_t_hor.mean(axis=0) / v_t_now

# %% 3. P&L of call and put options after one day and at the horizon 
t_1 = busday_offset(t_now, 1, roll='forward')  # next business day 

######################################################### after one day #########################################################
# options parameters
tau_opt_one_day = busday_count(t_1, t_option_end)/252  # time to expiry
s_one_day = exp(x_fcst[:, 1, n_stocks])  # underlying price forecast
m_one_day = log(s_one_day/k_strk)/sqrt(tau_opt_one_day)  # moneyness
log_sigma_one_day = x_fcst[:, 1, idx_options].reshape(j_bar, -1, l_bar)  # log-implied volatility forecast

# interpolated log-implied volatility 
log_sigma_interp_one_day = zeros(j_bar)
grid = list(zip(*[grid.flatten() for grid in meshgrid(*[tau, m])])) # grid
for j in range(j_bar):
    # interpolation
    # known values
    values_one_day = log_sigma_one_day[j, :, :].flatten()  
    # moneyness 
    moneyness_one_day = min(max(m_one_day[j], min(m)), max(m))
    # log-implied volatility
    log_sigma_interp_one_day[j] = interpolate.LinearNDInterpolator(grid, values_one_day)(*r_[tau_opt_one_day, moneyness_one_day])      
    '''
    Interpolation...ich denke mal, da Tau_Opt_One_Day kein Integer... müssen wir zwischen den beiden nächsten Werten Interpolieren
    '''
    
### Black Scholes Merton Pricing
# options values 
d1 = (moneyness_one_day + (y + (exp(log_sigma_interp_one_day)**2)/2)*sqrt(tau_opt_one_day))/exp(log_sigma_interp_one_day)
d2 = d1 - exp(log_sigma_interp_one_day)*sqrt(tau_opt_one_day)

# call option via Black-Scholes-Merton formula
v_call_one_day = s_one_day*(normal.cdf(d1) - exp(-(moneyness_one_day*sqrt(tau_opt_one_day) +\
                                                          y*tau_opt_one_day))*normal.cdf(d2)) 
# put option via put-call parity
v_put_one_day = v_call_one_day - s_one_day + k_strk*exp(-y*tau_opt_one_day)   

# P&L
pi_one_day[:, n_stocks + 1] = v_call_one_day - v_t_now[n_stocks + 1]  # call  
pi_one_day[:, n_stocks + 2] = v_put_one_day - v_t_now[n_stocks + 2]  # put


####################################################### at the horizon #######################################################
'''
Identisch zu oben, nur anderer tau_opt
'''
# options parameters 
tau_opt_t_hor = busday_count(t_hor, t_option_end)/252  # time to expiry
s_t_hor = exp(x_fcst[:, -1, n_stocks])  # underlying price forecast
m_t_hor = log(s_t_hor/k_strk)/sqrt(tau_opt_t_hor)  # moneyness 
log_sigma_t_hor = x_fcst[:, -1, idx_options].reshape(j_bar, -1, l_bar)  # log-implied volatility forecast

# interpolated log-implied volatility
log_sigma_interp_t_hor = zeros(j_bar)
grid = list(zip(*[grid.flatten() for grid in meshgrid(*[tau, m])]))
for j in range(j_bar):
    # interpolation
    # known values
    values_t_hor = log_sigma_t_hor[j, :, :].flatten()
    # moneyness
    moneyness_t_hor = min(max(m_t_hor[j], min(m)), max(m))
    # log-implied volatility
    log_sigma_interp_t_hor[j] = interpolate.LinearNDInterpolator(grid, values_t_hor)(*r_[tau_opt_t_hor, moneyness_t_hor])
    '''
    Interpolator nochmal verstehen.
    '''
# options values 
d1 = (moneyness_t_hor + (y + (exp(log_sigma_interp_t_hor)**2)/2)*sqrt(tau_opt_t_hor))/exp(log_sigma_interp_t_hor)
d2 = d1 - exp(log_sigma_interp_t_hor)*sqrt(tau_opt_t_hor)
v_call_t_hor = s_t_hor*(normal.cdf(d1) - exp(-(moneyness_t_hor*sqrt(tau_opt_t_hor) +\
                                                      y*tau_opt_t_hor))*normal.cdf(d2)) # call option via Black-Scholes-Merton 
v_put_t_hor = v_call_t_hor - s_t_hor + k_strk*exp(-y*tau_opt_t_hor)  # put option via put-call parity 

# P&L
pi_t_now_t_hor[:, n_stocks + 1] = v_call_t_hor - v_t_now[n_stocks + 1]  # call
pi_t_now_t_hor[:, n_stocks + 2] = v_put_t_hor - v_t_now[n_stocks + 2]  # put 


# %% 4. Bonds value scenarios without credit risk at the horizon 
'''
Das sind die reinen Bond-Values ohne Credit-Element... und OHNE Beitrag der Cupons, die im Zeitablauf gezahlt wurden.
'''
## GE

# dates of coupon payments from t_now to time of maturity assumed to be equal to record dates
r_ge = flip(date_range(start=t_end_ge, end=t_now, freq='-180D'))
r_ge = busday_offset(array(r_ge).astype('datetime64[D]'), 0, roll='forward')  # Next Business Day
coupon_ge_semi = coupon_ge / 2  # semiannual coupon rate
c_ge = coupon_ge_semi*ones(len(r_ge))  # coupon values

# Coupon Bond Value without Credit Risk
'''
Verwendung der Nelson-Siegel Parameter
'''
v_ge_bond_t_hor = zeros((j_bar, m_bar + 1))
v_ge_bond_t_hor[:, 0] = v_t_now[n_stocks + 3]
for m in range(1, m_bar + 1):
    # business day forecast
    t_m = busday_offset(t_now, m, roll='forward')  
    
    # Nelson-Siegel parameters
    theta_ge = x_fcst[:, m, idx_ge_bond]  
    
    # last parameter is squared
    theta_ge[:, 3] = theta_ge[:, 3]**2  
    
    # coupons paid on or after t_m
    r_ge_tm, c_ge_tm = r_ge[r_ge >= t_m], c_ge[r_ge >= t_m]  
    
    # coupon bond value at the horizon
    v_ge_bond_t_hor[:, m] = bond_value(t_m, 
                                       theta_ge,  # NS Parameter
                                       [], 
                                       c_ge_tm, 
                                       r_ge_tm, 
                                       'ns')   

## JPM 
# dates of coupon payments from t_now to time of maturity assumed to be equal to record dates
r_jpm = flip(date_range(start=t_end_jpm, end=t_now, freq='-180D'))
r_jpm = busday_offset(array(r_jpm).astype('datetime64[D]'), 0, roll='forward')
coupon_jpm_semi = coupon_jpm/2  # semiannual coupon rate
c_jpm = coupon_jpm_semi*ones(len(r_jpm))  # coupon values

# Coupon Bond Value without credit risk
v_jpm_bond_t_hor = zeros((j_bar, m_bar + 1))
v_jpm_bond_t_hor[:, 0] = v_t_now[n_stocks + 4]
for m in range(1, m_bar + 1):
    # business day forecast
    t_m = busday_offset(t_now, m, roll='forward')  
    # Nelson-Siegel parameters
    theta_jpm = x_fcst[:, m, idx_jpm_bond]  
    # last parameter is squared
    theta_jpm[:, 3] = theta_jpm[:, 3]**2
    # coupons paid on or after business day forecast
    r_jpm_tm, c_jpm_tm = r_jpm[r_jpm >= t_m], c_jpm[r_jpm >= t_m]  
    # coupon bond value at the horizon 
    v_jpm_bond_t_hor[:, m] = bond_value(t_m, theta_jpm, [], c_jpm_tm, r_jpm_tm, 'ns')  


# %% 5. Cumulative cash-flow scenarios for corporate bonds without credit risk 
'''
DAs sind die Cash-Flows.
Die werden mit einem Investmentfaktor "hoch gezinst auf die aktuelle Periode"
'''

########## input (you can change it) ##########
d_tm = 1/252  # one day
###############################################

inv = exp(y*d_tm)*ones((j_bar, m_bar))  # investment factor

# GE
r_ge_cf = r_ge[r_ge < datetime64(t_hor, 'D')]  # payment dates
t_now_t_hor_ge = array(bdate_range(t_now, min(t_end_ge, t_hor)))  # monitoring dates
c_ge_cf = ones((len(r_ge_cf)))*coupon_ge_semi  # coupon payments

# Scenarios of Cumulative Cash-Flow 
cf = zeros((inv.shape[0], inv.shape[1], r_ge_cf.shape[0]))
ml = array([where(t_now_t_hor_ge == rc) for rc in r_ge_cf], dtype=object).reshape(-1)
for l in arange(len(ml)):
    # Reinvestment Factors from each Payment Day
    if ml[l] != inv.shape[1]:
        inv_tmk = inv[:, ml[l]:]
    else:
        inv_tmk = ones((inv.shape[0], 1)) 
    # Scenarios for Cumulative Cash-Flow
    cf[:, ml[l]:, l] = c_ge_cf[l] *cumprod(inv_tmk, axis=1)
# cumulative reinvested cash-flow 
cf_ge = sum(cf, 2).squeeze()


## JPM
r_jpm_cf = r_jpm[r_jpm < datetime64(t_hor, 'D')]  # payment dates
c_jpm_cf = ones((len(r_jpm_cf)))*coupon_jpm_semi  # coupon payments
t_now_t_hor_jpm = array(bdate_range(t_now, min(t_end_jpm, t_hor)))  # monitoring dates
# scenarios of cumulative cash-flow 
cf = zeros((inv.shape[0], inv.shape[1], r_jpm_cf.shape[0]))
ml = array([where(t_now_t_hor_jpm == rc) for rc in r_jpm_cf], dtype=object).reshape(-1)
for l in arange(len(ml)):
    # reinvestment factors from each payment day
    if ml[l] != inv.shape[1]:
        inv_tmk = inv[:, ml[l]:]
    else:
        inv_tmk = ones((inv.shape[0], 1))
    # scenarios for cumulative cash-flow 
    cf[:, ml[l]:, l] = c_jpm_cf[l]*cumprod(inv_tmk, axis=1)
# cumulative reinvested cash-flow 
cf_jpm = sum(cf, 2).squeeze()


# %%6. P&L of the corporate bonds with credit risk after one day and at the horizon (Read more) 
########## inputs (you can change them) ##########
rec_rate_ge = 0.6  # recovery rate for GE bond
rec_rate_jpm = 0.7  # recovery rate for JPM bond
##################################################

ratings_fcst = db_projection_ratings.values.reshape((j_bar, m_bar + 1, 2))  # ratings forecast scenarios
default = any(ratings_fcst == c_bar, axis=1, keepdims=True).squeeze()  # default indicator
m_d = full((j_bar, 2), 0, dtype='int')  # time of default
# value of coupon bonds with credit risk
for n in range(2):
    for j in range(j_bar):
        if default[j, n]: # if default occurs
            # first date of default
            m_d[j, n] = where(ratings_fcst[j, :, n] == c_bar)[0][0]  
            # P&L forecast of underlying stock 
            pi_t_now_t_hor[j, n] = -v_t_now[n] 
            if ratings_fcst[j, 1, n] == c_bar:
                pi_one_day[j, n] = -v_t_now[n] 
            
'''
Wofür? -> Wenn ein Default occurs... dann setzt er auch die Stocks auf 0
Darüber ergibt sich eine gewisse Beziehung.
Allgemein wird hier aber angenommen, dass das Thema Credit Risk eher idiosynkratisch ist
'''
## GE
# values with market and credit risk after one day
v_mc_ge_bond_one_day = v_ge_bond_t_hor[:, 1].copy()  # coupon bond value
cf_mc_ge_one_day = cf_ge[:, 0].copy()  # reinvested cash-flow value
# values with market and credit risk at the horizon 
v_mc_ge_bond_t_hor = v_ge_bond_t_hor[:, -1].copy()  # coupon bond value
cf_mc_ge_t_hor = cf_ge[:, -1].copy()  # reinvested cash-flow value

# JPM
# values with market and credit risk after one day
v_mc_jpm_bond_one_day = v_jpm_bond_t_hor[:, 1].copy()  # coupon bond value
cf_mc_jpm_one_day = cf_jpm[:, 0].copy()  # reinvested cash-flow value 
# values with market and credit risk at the horizon 
v_mc_jpm_bond_t_hor = v_jpm_bond_t_hor[:, -1].copy()  # coupon bond value
cf_mc_jpm_t_hor = cf_jpm[:, -1].copy()  # reinvested cash-flow value 

# coupon bonds and cash flows values with credit risk
for j in range(j_bar):
    # GE
    if default[j, 0]:  # if default occurs
        if m_d[j, 0] == 1:  # if default at first future horizon
            v_mc_ge_bond_t_hor[j] = v_t_now[n_stocks + 3]*rec_rate_ge
            cf_mc_ge_t_hor[j] = 0
            # one day values
            v_mc_ge_bond_one_day[j] = v_t_now[n_stocks + 3]*rec_rate_ge
            cf_mc_ge_one_day[j] = 0
        else:
            # coupon bond value with credit risk
            v_mc_ge_bond_t_hor[j] = v_ge_bond_t_hor[j, int(m_d[j, 0]) - 1]*rec_rate_ge  
            # cash-flow with credit risk
            cf_mc_ge_t_hor[j] = cf_ge[j, int(m_d[j, 0]) - 1]*prod(inv[j, int(m_d[j, 0]):])
    # JPM
    if default[j, 1]:  # if default occurs
        if m_d[j, 1] == 1:  # if default at first future horizon
            v_mc_jpm_bond_t_hor[j] = v_t_now[n_stocks + 4]*rec_rate_jpm
            cf_mc_jpm_t_hor[j] = 0
            # one day values
            v_mc_jpm_bond_one_day[j] = v_t_now[n_stocks + 4]*rec_rate_jpm
            cf_mc_jpm_one_day[j] = 0
        else:
            # coupon bond value with credit risk
            v_mc_jpm_bond_t_hor[j] = v_jpm_bond_t_hor[j, int(m_d[j, 1]) - 1]*rec_rate_jpm
            # cash-flow with credit risk
            cf_mc_jpm_t_hor[j] = cf_jpm[j, int(m_d[j, 1]) - 1]*prod(inv[j, int(m_d[j, 1]):])
            
# GE
# coupon bond P&L with credit risk
pi_one_day[:, n_stocks + 3] = v_mc_ge_bond_one_day - tile(v_t_now[n_stocks + 3], j_bar) + cf_mc_ge_one_day  # after one day
pi_t_now_t_hor[:, n_stocks + 3] = v_mc_ge_bond_t_hor - tile(v_t_now[n_stocks + 3], j_bar) + cf_mc_ge_t_hor  # at the horizon

# JPM 
# coupon bond P&L with credit risk
pi_one_day[:, n_stocks + 4] = v_mc_jpm_bond_one_day - tile(v_t_now[n_stocks + 4], j_bar) + cf_mc_jpm_one_day  # after one day 
pi_t_now_t_hor[:, n_stocks + 4] = v_mc_jpm_bond_t_hor - tile(v_t_now[n_stocks + 4], j_bar) + cf_mc_jpm_t_hor  # at the horizon

############## input (you can change it) ##############
n_plot = 1  # index of instrument to plot from 1 to 10
#######################################################

########################################## plot ##########################################
# ex-ante P&L distribution 
histplot(x=pi_t_now_t_hor[:, n_plot - 1], weights=p_scenario, 
         bins=30).set(title='Ex-ante P&L: '+ db_v_tnow.columns[n_plot - 1], xlabel='P&L');


# %% Save data
# Ex-Ante P&Ls of Stocks, S&P 500, Options and Bonds after one day
out = {db_v_tnow.columns[n]: pi_one_day[:, n] for n in range(n_bar)}
names = [db_v_tnow.columns[n] for n in range(n_bar)]
out = DataFrame(out); out = out[list(names)]
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_oneday_pl.csv', index=False); del out;

# Ex-ante P&Ls of Stocks, S&P 500, Options and Bonds at the horizon
out = {db_v_tnow.columns[n]: pi_t_now_t_hor[:, n] for n in range(n_bar)}
names = [db_v_tnow.columns[n] for n in range(n_bar)]
out = DataFrame(out); out = out[list(names)]
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_pricing.csv', index=False); del out;


