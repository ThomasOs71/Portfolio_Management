# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:14:07 2024

@author: Thomas

s_checklist_montecarlo_step05
This case study performs Step 5 of the "Checklist", namely Aggregation, to 

    compute the value and the ex-ante performance of a given portfolio or policy

"""

# %% Packages
from numpy import floor, r_, repeat, zeros
from pandas import DataFrame, read_csv, Series
from seaborn import histplot
from pandas.plotting import register_matplotlib_converters

# %% Load data

## Ex-ante P&Ls of stocks, S&P 500, options and bonds at the horizon (Montecarlo checklist - Step 4)
db_pricing = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_pricing.csv')
pi_t_now_t_hor = db_pricing.values

# Initial Time Values of stocks, S&P 500, options and bonds (Montecarlo checklist - Step 2)
db_v_tinit = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_v_tinit.csv')
v_t_init = db_v_tinit.values.squeeze() 

# Current Tiem Values of stocks, S&P 500, options and bonds (Montecarlo checklist - Step 2)
db_v_tnow = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_v_tnow.csv')
v_t_now = db_v_tnow.values.squeeze()

# Scenario probabilities forecast (Montecarlo checklist - Step 4)
db_scenario_probs = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_scenario_probs.csv')
p_scenario = db_scenario_probs['p_scenario'].values

# additional variables (Montecarlo checklist - Step 2)
# import db_riskdrivers_tools database 
db_riskdrivers_tools = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_riskdrivers_tools.csv')
n_stocks = len(db_riskdrivers_tools.stocks_names.dropna())  # number of stocks
n_bonds = int(db_riskdrivers_tools.n_bonds.iloc[0])  # number of corporate bonds 

# %% 1. Portfolio holdings following the current allocation 
############### inputs (you can change them) ###############
v_h_t_init = 250e6  # budget at time t_init
v_stocks_t_init = 200e6  # maximum budget invested in stocks
h_sp = 0  # holding of S&P 500
h_put_spx = 16000  # holding of put options on S&P 500
h_call_spx = 16000  # holding of call options on S&P 500
h_bonds = 22e6  # notional for bonds
############################################################

# stocks holdings
h_stocks = zeros(n_stocks)
for n in range(n_stocks):
    h_stocks[n] = floor(1/n_stocks*v_stocks_t_init/v_t_init[n])
'''
Jeder Stock kriegt 20% (5 Stocks)
'''
# check sum(h_stocks * v_t_init[:n_stocks])
    
h_bonds = repeat(h_bonds, n_bonds)  # bonds holdings
h = r_[h_stocks, h_sp, h_call_spx, h_put_spx, h_bonds]  # portfolio holdings

# %% 2. Initial value of cash 
######## input (you can change it) ########
v_h_t_init = 250e6  # budget at time t_init
###########################################

cash_t_init = v_h_t_init - h.T@v_t_init # cash at initial time 
'''
Cash entspricht dem initalen Cash minus den Werten im Portfolio
'''

# %% 3. Current value of holdings 
cash_t_now = cash_t_init  # cash value at current time - keine Verzinsung von Cash
v_h_t_now = h.T@v_t_now + cash_t_now  # value of holding at current time


# %% 4. Portfolio ex-ante performance (Read more) 
y_h = pi_t_now_t_hor@h  # portfolio ex-ante performance 

#################################### plot ####################################
# portfolio ex-ante performance distribution
histplot(x=y_h, # Basierend auf den Pfaden aller Instrumente und holdings
         weights=p_scenario, # 1/J
         bins=30).set(xlabel='$Y_h$ (million USD)');


# %% Save data
# portfolio ex-ante performance
out = DataFrame({'Y_h': Series(y_h)})
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_exante_perf.csv', index=False); del out;

# portfolio holdings
out = {db_v_tnow.columns[i]: h[i] for i in range(len(h))}
out = DataFrame(out, index=[0]); out = out[list(db_v_tnow.columns)]
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data//db_holdings.csv', index=False); del out;

# current value of holdings and cash
out = DataFrame({'v_h_tnow': v_h_t_now, 'cash_tnow': cash_t_now}, index=[0])
+out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_aggregation_tools.csv', index=False); del out;
