# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:15:04 2024

@author: Thomas


Step 5a: Value aggregation
This case study performs Step 5a of the "Checklist", namely Value aggregation. 
The purpose of this step is to:

                compute the value of a given portfolio, 

starting from the current values of the instruments.

Here we compute the current value v_h and portfolio weights w_h of a portfolio 
invested in (some of) the instruments under consideration, 
given the instrument holdings and cash.
"""

# %% Prepare environment
### Packages
from numpy import array, append
from pandas import read_csv, DataFrame, Series

###Load data
'''
From database db_valuation_historical created in script s_checklist_historical_step01,
we upload the values of the instruments at the current time 
'''

db_valuation_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step2/db_valuation_historical.csv', index_col=0)
v = db_valuation_historical.values[:7].astype('float').squeeze().T  # current values of all instruments


# %% 1. Portfolio current value 
'''
We compute the current value  v_h pf the portfolio h using linear pricing.
'''


########################## inputs (you can change them) ##########################
# stock holdings
h_stocks = array([[1952781, 
                   1092015, 
                   1082719, 4755846, 60510]])  

h_call = 16000  # S&P 500 call option holding

h_put = 16000  # S&P 500 put option holding

cash = 1828577  # cash holding
##################################################################################


h = append(h_stocks, [h_call, h_put])  # portfolio holdings

v_h = h.T@v + cash  # portfolio current value
print('v_h =', f'${v_h.round():,.0f}')

# %%2. Portfolio weights 
'''
We calculate the generalized portfolio weights w_h with
    the instrument denominators d set as the current instrument values v
        d = v
    the basis denominator d_h aset as the porftolio current value v_h
        dh = v_h

-> Das ist der Standard-Fall


Thus we multiply each instrument holding by the instrument's current value and 
divide by the portfolio value

w_h = h * v / v_h
'''

w_h = h*v/v_h  # portfolio weights
print('w_h =', w_h.round(4))

# %% Save data
'''
Save into database db_value_aggregation_historical:
    1) the holdings h and 
    2) portfolio weights 
    3) the current portfolio value 
    4) and cash holdings 
'''

out = DataFrame({'h': Series(h), 'w_h': Series(w_h), 'v_h': Series(v_h), 'cash': Series(cash)})
out.index = db_valuation_historical.index[:7]
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step5b/db_value_aggregation_historical.csv'); del out;
