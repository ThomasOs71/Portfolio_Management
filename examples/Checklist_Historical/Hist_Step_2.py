# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:02:00 2024

@author: Thomas


Step 2: Risk drivers identification
This case study performs Step 2 of the "Checklist", namely Risk drivers identification, for a market consisting of stocks and options on an index. The purpose of this step is to obtain the time series of the risk drivers 
 from the raw financial data available for the instruments under consideration. The most common risk drivers for each asset class are discussed extensively within Step 2. A summary is provided in this table.

Instrument      number	  Obs     Risk drivers       Dimension
GE stock        1           713     log-values              1        
JPM stock       2           713     log-values              1	
A stock	        s.o
AA stocks        s.o
AAPL stock	    s.o
Call option     6           713     log value Underly      1
  & Put         7           713     log-implied vola sur   90
       
Total           7           713                             96
    
  
"""

'''
Here we extract the risk drivers identified in the table above for each instrument in the market.
'''

# %% Prepare environment
from numpy import array, busday_count, c_, datetime64, exp, intersect1d, log, meshgrid, ones, r_, reshape, sqrt, zeros
from implvol_delta2m_moneyness import implvol_delta2m_moneyness
from pandas import concat, DataFrame, read_csv, Series
from seaborn import lineplot
from matplotlib.pyplot import gcf

# %% Load data

### Current Values
'''
From database db_valuation_historical created in script s_checklist_historical_step01 we upload the number of stocks stock names and current time 
'''
db_valuation_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step2/db_valuation_historical.csv', index_col=[0])
n_stocks = int(db_valuation_historical.loc['n_stocks'].values[0])  # number of stocks in market
stocks_names = db_valuation_historical.index.values[:n_stocks]  # names of stocks in market
t_now = datetime64(db_valuation_historical.loc['t_now'].values[0])  # current time

###################### input (you can change it) ######################
t_first = datetime64('2009-11-02')  # set start date for data selection 
#######################################################################

## Stock Prices
'''
From database db_stocks_sp we upload time series of the (dividend and split-adjusted) close values stocks in the S&P 500
'''
# import db_stocks_sp database 
db_stocks_sp = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step2/db_stocks_sp.csv',
                        skiprows=[0], index_col=0, parse_dates=True)
dates = db_stocks_sp.index[(db_stocks_sp.index >= t_first)&(db_stocks_sp.index <= t_now)]  # date range of interest

## Implied Volatility
'''
From database db_implvol_optionSPX for the same range of dates, we upload values of the underlying S&P 500
       and implied volatility surface grid for options on the S&P 500 of realizations of the implied volatility surface in 
-moneyness parametrization       for a grid 
 of 
-moneyness and times to expiry.
'''

# import db_implvol_optionSPX/data database 
db_implvol_optionSPX_data = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step2/data_implvol_sandp.csv', 
                                     index_col=['date'], parse_dates=['date'])

# select same date range in both databases
dates = intersect1d(dates, db_implvol_optionSPX_data.index).astype('datetime64[D]')
v_stocks = db_stocks_sp.loc[dates, stocks_names]  # time series of stocks
s = db_implvol_optionSPX_data.loc[(db_implvol_optionSPX_data.index.isin(dates)), 'underlying']  # values of underlying S&P 500
sigma_delta = db_implvol_optionSPX_data.loc[(db_implvol_optionSPX_data.index.isin(dates)),
                                            (db_implvol_optionSPX_data.columns != 'underlying')]  # implied volatility surface grid


# import db_implvol_optionSPX/params database
db_implvol_optionSPX_params = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step2\params_implvol_sandp.csv', index_col=False)
delta_grid = array(db_implvol_optionSPX_params.delta)  # delta-moneyness 
tau_grid = array(db_implvol_optionSPX_params.time2expiry.dropna())  # times to expiry



# %% 1. Risk drivers for stocks 
'''
For each stock, we have one risk driver, providing us with a total of  risk drivers in total for the stocks. 
Accordingly, we compute the time series of the risk drivers as the log-values for all stocks.
'''

# risk drivers for stocks
x_stock = log(array(v_stocks))

############################ plot ############################
############# input (you can change it) ##############
d_plot = 1  # index of risk driver to plot from 1 to 5
######################################################

# risk driver for stocks
lineplot(x=dates, y=x_stock[:, d_plot-1]).set(title='Stock '+\
         stocks_names[d_plot-1]+' log value', xlabel='date');
gcf().autofmt_xdate();

# %% 2. Risk drivers for options
'''
Next, we compute the risk drivers for the options. 

For the interest rate, we rely on the simplifying assumption that the yield curve driving the options prices is flat, and constant through time
for every 
 and every time to maturity 
. Hence, we do not include the respective entries among the risk drivers.
'''

'''
Step 1:
Risk Driver of Underlying
We first compute the time series of the log-values of the underlying for the options, namely the S&P 500 index.
'''
x_sp = log(array(s))  # risk driver of underlying (S&P 500 index)

'''
Step 2:
Risk Driver for Yield Curce
-> Entfällt. Nehmen einfach eine flate Yield Curve an,
y= 0.01 (s.u.)
'''

'''
Step 3: 
Risk Driver of Implied Volatility Surface
    I) We convert the time series of the implied volatility surface from the delta-moneyness parametrization 
        to the m-moneyness parametrization.
'''
################ inputs (you can change them) ################ 
l_bar = 9  # number of points on m-moneyness grid
y = 0.01  # level for yield curve (assumed flat and constant)
#############################################################

# convert time series of implied volatility surface from delta to m-moneyness parametrization
implvol_delta_moneyness_3d = zeros((len(dates), len(tau_grid), len(delta_grid)))
for k in range(len(tau_grid)):
    implvol_delta_moneyness_3d[:, k, :] = array(sigma_delta.iloc[:, k::len(tau_grid)])
sigma_grid, m_grid = implvol_delta2m_moneyness(implvol_delta_moneyness_3d, tau_grid, delta_grid,
                                               y*ones((len(dates), len(tau_grid))), tau_grid, l_bar)

'''
Step 3:
    II) We compute the risk drivers as the log-implied volatility surface 
'''
# implied volatility risk drivers for options 
x_sigma = log(reshape(sigma_grid, (len(dates), len(tau_grid)*(l_bar)), 'F'))  # log-implied volatility

########################### plot ###########################
# volatility surface risk drivers
lineplot(x=dates, y=x_sigma[:, d_plot-1]).set(xlabel='date');
gcf().autofmt_xdate();


# %% Save data
'''
1) Save risk drivers for the stocks and options -> db_riskdrivers_historical
2) Save the Start Date, values of the times-to-expiry , moneyness values in grid, 
    number of points in the moneyness grid, yield y
'''

x_names = ['stock ' + name + '_log_value' for name in stocks_names] + ['S&P index'] +\
          ['option_spx_logimplvol_mtau_' + str(i - 5) for i in range(6, 96)]
          
out = DataFrame(c_[x_stock, x_sp.reshape(-1, 1), x_sigma], columns=x_names)  # Risk Drivers

out = out.join(DataFrame({'t_first': Series(t_first), 'tau': Series(tau_grid), 
                             'm': Series(m_grid), 'l_bar': Series(l_bar), 'y': Series(y),}))
out.index = dates; out.index.name = 'dates'
out.to_csv('C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step3\db_riskdrivers_historical.csv'); del out;
