# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:16:06 2024

@author: Thomas

Step 5b: Performance aggregation
This case study performs Step 5b of the "Checklist", namely Performance aggregation. 
The purpose of this step is:
    
    to compute the ex-ante performance of a given portfolio, 
    
    starting from the joint ex-ante P&L's of the instruments.

Here we compute the scenario-probability distribution of the ex-ante performance R_w 
of a portfolio invested in (some of) the instruments under consideration, given 
the holdings h and the historical repriced scenarios of the instruments' ex-ante P&L's Pi 

We consider the linear returns at the investment horizon as  the measure of 
the portfolio performance

    Y_h = R_w

and we compute its scenario-probability distribution  {r_w, p}^j.
"""

# %% Prepare environment
### Packages
from pandas import concat, DataFrame, read_csv, Series
from seaborn import histplot

### Load data
'''
From database db_valuation_historical created in script s_checklist_historical_step01,
we upload:
    1) the values of the instruments at the current time.
'''
db_valuation_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step2\db_valuation_historical.csv', index_col=0)
v = db_valuation_historical.values[:7].astype('float').squeeze().T  # current values of all instruments

'''
From database db_forecast_historical created in script s_checklist_historical_step03b,
we upload:
    1) scenario probabilities.
'''

db_forecast_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step4/db_forecast_historical.csv', 
                                  low_memory=False)
p_j = db_forecast_historical.p_scenario.dropna()  # probabilities

'''
From database db_pricing_historical created in script s_checklist_historical_step04,
we upload: 
    1) the joint P&L scenarios of the instruments at the investment horizon.
'''
db_pricing_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step5\db_pricing_historical.csv')
pi = db_pricing_historical.values  # ex-ante instrument P&L scenarios

'''
From database db_value_aggregation_historical created in script s_checklist_historical_step05a,
we upload:
    1) the holdings 
    2) and portfolio weights.
'''
db_value_aggregation_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step5b\db_value_aggregation_historical.csv', index_col=0)
h = db_value_aggregation_historical.h.values  # holdings
w_h = db_value_aggregation_historical.w_h.values  # portfolio weights

# %% 1. Instrument returns 
'''
Since the performance variable is the portfolio linear return, it is calculated 
as the generalized linear return with:

1) the instrument denominators d set as the current instrument value v
    d = v

2) the basis denominator d_h set as the portfolio current value v_h
    d_h = v_h

3) the cash benchmark 
    R_b = 0


We compute the instrument ex-ante return scenarios {r^(j)}^J  by scaling 
the scenarios of the P&L at the investment horizon {pi^(j)}^J by the 
current values of the instruments 
'''
 
r = pi/v  # instrument returns

# %% 2. Portfolio ex-ante return 
'''
We compute the distribution of the ex-ante portfolio return by 
    
    linearly aggregating the scenarios of the instrument returns {r^{j}}
    
obtaining the portfolio performance scenarios {r_w^{j}}.

The probabilities, inherited from the scenario-probability distribution 
of the instrument P&L's, are all equal p^{j} = 1/j. 

Combining the scenarios and probabilities, we obtain 
    1) the scenario-probability distribution of the performance 
'''

r_w = r@w_h  # portfolio ex-ante return

######################## plot ########################
# portfolio ex-ante performance distribution
histplot(x=r_w, bins=30, weights=p_j).set(ylabel='');

# %% Save data
'''
Save the ex-ante portfolio P&L scenarios and the instrument return scenarios,
into database db_exante_perf_historical.
'''

out1 = DataFrame({'r_w': Series(r_w)})
out2 = DataFrame(r, columns=db_pricing_historical.columns)
out = concat([out1, out2], axis=1)
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step6/db_exante_perf_historical.csv', index=False); del out;
