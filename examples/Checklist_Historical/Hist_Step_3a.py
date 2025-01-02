# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:30:02 2024

@author: Thomas


Step 3a: Univariate quest for invariance
This case study performs Step 3a of the "Checklist", namely Univariate quest for invariance. 
The purpose of this step is to fit the most suitable next-step model to the time series of risk drivers. 
The most popular next-step models are discussed extensively within Step 3a.

Instrument      Risk Driver     Next_Step_Model
Stocks          Log-Value       Garch(1,1)
Option Underly  Log-Value       Garch(1,1)
Option          log-implied vol Random Walk

Here we present a practical implementation of the Univariate quest for invariance, 
applied to the large dimensional vector of risk drivers obtained in s_checklist_historical_step02.

PART I: Next-Step Modelling and Invariants Extraction
Each risk driver has its own next-step function with specific parameters. 
The invariant dimension column summarizes the number of risk drivers that, for simplicity and clarity of exposition, 
we choose to model in a similar way, since they are similar in nature. 
Note also that, in general, the number of invariants may not coincide with the number of risk drivers.

PART II: Invariance Tests
Once the next-step models have been fitted, 
we perform invariance tests on the time series of each (candidate) invariant 
to verify whether it displays i.i.d. behavior across time. 
More precisely, we perform the invariance tests explored in Invariance tests: 
    1) the ellipsoid test; 
    2) the Kolmogorov-Smirnov test, and 
    3) the copula test.

If any of the invariance tests fails for any of the invariants, 
we must reconsider the model for the corresponding risk drivers, and so we must 
repeat the first three steps for the new model. 

PART III:
If the invariance tests succeed, then we proceed to Multivariate quest and forecasting.
"""

# %% Prepare environment
from numpy import array, append, diff, zeros
from arch import arch_model
from pandas import concat, DataFrame, read_csv, to_datetime
from invariance_test_copula import invariance_test_copula
from invariance_test_ellipsoid import invariance_test_ellipsoid
from invariance_test_ks import invariance_test_ks

# %% Load data
'''
From database db_valuation_historical created in s_checklist_historical_step01 we upload the number of stocks 
'''
db_valuation_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step2\db_valuation_historical.csv', index_col=[0])
n_stocks = int(db_valuation_historical.loc['n_stocks'].values[0])  # number of stocks in market

'''
From database db_riskdrivers_historical created in s_checklist_historical_step02 we upload time series of risk drivers for the stocks 
 and options .
'''
db_riskdrivers_historical = read_csv('C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step3\db_riskdrivers_historical.csv', 
                                     index_col=0, parse_dates=True)
x_stocks = db_riskdrivers_historical.values[:, :n_stocks].astype('float64')  # risk drivers for stocks
x_sp = db_riskdrivers_historical.values[:, n_stocks].astype('float64')  # risk driver for underlying S&P 500 index
x_sigma = db_riskdrivers_historical.values[:, n_stocks+1:-5].astype('float64')  # risk drivers for options
risk_drivers_names = db_riskdrivers_historical.columns.values[:-5]  # risk drivers names
dates = to_datetime(array(db_riskdrivers_historical.index))  # date range of interest 
t_bar = len(dates) - 1  # length of invariants time series
i_bar = len(risk_drivers_names)  # number of invariants

# %% PART 1) NEXT STEP MODELS
### 1. Stock: Next-Step model
'''
We fit the parameters of the next-step models to the time series of risk drivers 
as summarized in this table. We list the implementation steps below in detail for risk drivers of each instrument.
'''
# 1: We compute time series of risk driver increments for all stocks.
delta_x_stocks = diff(x_stocks, axis=0)  # Stock Risk driver increments

# 2: Fit a GARCH model to the risk driver increments using the maximum likelihood 
#    with flexible probabilities approach, obtaining the parameters 
#    as well the time series of (candidate) invariants, for all stocks.

db_invariants = zeros((t_bar, i_bar))
db_garch_param = {}; db_nextstep = {}
for i in range(n_stocks):
    garch = arch_model(delta_x_stocks[:, i], vol='garch', p=1, o=0, q=1, rescale=False)  # define GARCH(1,1) model
    garch_fitted = garch.fit(disp='off')  # fit GARCH(1,1)
    par = garch_fitted._params  # parameters
    sigma2 = garch_fitted._volatility  # realized variance
    eps = garch_fitted.std_resid  # GARCH(1,1) invariants
    # store invariants, GARCH(1,1) parameters and next-step function
    db_invariants[:, i] = eps
    db_garch_param[i] = dict(zip(['mu', 'c', 'a', 'b'] + ['sig2_' + str(t).zfill(3) for t in range(t_bar)], append(par, sigma2)))
    db_nextstep[i] = 'GARCH(1,1)'


### 2. Options(I): Next-Step model for Underlying
# 1. We first compute the increments of the underlying S&P 500 index risk driver.
delta_x_sp = diff(x_sp)  # underlying risk driver increments

# 2: Fit a GARCH model to the risk driver increments using the maximum likelihood 
#    with flexible probabilities approach, obtaining the parameters 
#    as well the time series of (candidate) invariants for the underlying.

garch = arch_model(delta_x_sp, vol='garch', p=1, o=0, q=1, rescale=False)  # define GARCH(1,1) model
garch_fitted = garch.fit(disp='off')  # fit GARCH(1,1)
par = garch_fitted._params  # parameters
sigma2 = garch_fitted._volatility  # realized variance
eps = garch_fitted.std_resid  # GARCH(1,1) invariants
# store invariants, GARCH(1,1) parameters and next-step function
db_invariants[:, n_stocks] = eps
db_garch_param[n_stocks] = dict(zip(['mu', 'c', 'a', 'b'] + ['sig2_' + str(t).zfill(3) for t in range(t_bar)], append(par, sigma2)))
db_nextstep[n_stocks] = 'GARCH(1,1)'


### 3. Options(II)
# 1. We compute the increments of each volatility surface risk driver of the options.
delta_x_sigma = diff(x_sigma, axis=0)  # volatility risk driver increments
# 2. Then we extract the realizations of the invariants 
#     for the options log-implied volatility assuming a random walk as next-step model. 
#     (assume Invariants from RW are invariants)

for i in range(x_sigma.shape[1]): 
    # time series of invariants
    eps = delta_x_sigma[:, i] 
    # store invariants and next-step function
    db_invariants[:, i + n_stocks + 1] = eps
    db_nextstep[i + n_stocks + 1] = 'Random walk'
        
# %% PART 2) INVARIANCE TESTS

### 1. Ellipsoid Test
'''
We first perform the ellipsoid test for invariance on each of the time series of 
(candidate) invariants.
We set maximum lag used to perform the ellipsoid invariance test and the 
copula invariance test as l =5
'''
########### inputs (you can change them) ###########
i_plot = 1  # invariant to be tested from 1 up to 96
l_bar = 5  # maximum lag used in invariance tests 
####################################################

# invariant 
invar = db_invariants[:, i_plot - 1]

################## plot ##################
# ellipsoid test
ellipsoid_test_results = invariance_test_ellipsoid(invar, l_bar)  # Hier lieber

### 2. Kolmogorov-Smirnov test for invariance
'''
Next we perform the Kolmogorov-Smirnov test on each of the invariants time series of (candidate) invariants 
'''

########### plot ############
# Kolmogorv-Smirnov test
ls_test_result = invariance_test_ks(invar)

# %% 3. Copula test for invariance
'''
Finally, we perform the copula invariance test on each of the time series of (candidate) invariants 
. The measure of dependence plotted is the Schweizer-Wolff measure.
'''

################ plot #################
# copula test
cop_test_result = invariance_test_copula(invar, l_bar)
'''

'''


# %% Save data
'''
Save invariants 
 and the parameters for the GARCH
 models into database db_univariate_quest_historical, as well as the length of the invariant time series 
.
'''
invar = DataFrame(db_invariants, index=dates[1:], columns=risk_drivers_names)
invar.index.name = 'dates'
garch_params = DataFrame({risk_drivers_names[i]: db_garch_param[i] for i in range(len(db_garch_param))})
out = concat([invar, garch_params])
out = out.join(DataFrame(t_bar, index = [dates[1]], columns = ['t_bar']))
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step3b\db_univariate_quest_historical.csv'); del out;
