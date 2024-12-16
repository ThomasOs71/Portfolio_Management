# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:50:14 2024

@author: Thomas

Besser Unterteilen und dann funktionen schreibn.
Brauchen klare Steps und Struktur, sonst wird das knifflig



"""

# %% Packages
### External
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model

### Internal
from pm_modules.estimation.fin_engineering import risk_driver_equity, risk_driver_fe_nelson_siegel
from pm_modules.estimation.invariance_tests import invariance_test_ellipsoid, invariance_test_ks, invariance_test_copula

# %% Global Data Settings
### Frequency of Data
freq = 'ME'


# %% Data Load: Equities
## Load from Pickle
os.chdir(r'C:\Projects\VENV\DataStream_DL\Sample_Data')
with open("equity", 'rb') as file:
    data = pickle.load(file)

## Obtain Data Information
equity_sheets = ["World_Equity","Region_Equity", "Country_Equity", 
                 "World_Sector", "Europe_Sector", "EMU_Sector"]

## Equity Sample Information
equity_data_info = {}
for i in equity_sheets:
    equity_data_info[i] = pd.read_excel("Sample_Data_Codes.xlsx",
                                         sheet_name=i)
## Data Equity
equity_data = {}
for i, name in enumerate(equity_sheets):
    equity_data[name] = data[i].unstack(-2).resample(freq).last()


### Additional Information of Data Sample
for i in equity_sheets:
    # Time Series Starting Point: Find First Element, that is not NAN
    if "first_msri" not in equity_data_info[i].columns:
        
        first_avail = equity_data[i].isnull().idxmin().loc["MSRI",:]
        first_avail = pd.DataFrame(first_avail,columns=["first_msri"],
                                   index = first_avail.index)
        equity_data_info[i] = pd.merge(equity_data_info[i], 
                                                     first_avail, how="left", 
                                                     left_on="Code",right_index=True)  # Merge with Information Elemente
        
    # Time Series Ending Point: Find Last Element, that is not NAN  
    if "last_msri" not in equity_data_info[i].columns:
        last_avail = equity_data[i].iloc[::-1,:].isnull().idxmin().loc["MSRI",:]  # Find First Elemente, that is not NA
        last_avail = pd.DataFrame(last_avail,columns=["last_msri"],index=last_avail.index)
        equity_data_info[i] = pd.merge(equity_data_info[i], 
                                                     last_avail, how="left", 
                                                     left_on="Code",right_index=True)  # Merge with Information Elemente
            
# %% Select Investment Universe
## Database of Assetclass
data_assets = equity_data["World_Equity"]["MSRI"]

## Select Data Start Date
t_start = "12-31-1993"
# TODO: 
    # Später auf endogen umstellen 
    # Linear Projection / Conditioning nutzen, da hohe Korrelation zw. Assets
    
## Subselect data of assets
data_raw = data_assets.loc[t_start:,:]

instrument_info = equity_data_info["World_Equity"].set_index("Code")

'''
Keine Wertpapiere, daher muss ich die Values in t = 0 vorgeben
'''
## Value at Start
v_t_now = np.array([100] * data_raw.shape[1], ndmin = 2).T

## Time Date Range
dates = np.array(data_raw.index)
## Length Time Sereis
t_bar = data_raw.shape[0]

## Instrument Columns
instrument_columns = [instrument_info.loc[i,"Name"] for i in data_raw.columns]
## Number of Instruments
n_bar = len(instrument_columns)

# %% Step 2: Identification of Risk Driver
'''
Risk Driver for Equities are log-values
'''
## Log-Values
rd_raw = np.log(data_raw).values

## Optional Graphing
rd_graph_bool = True

if rd_graph_bool:
    fig = plt.figure()
    for i in range(0,data_rd.shape[1]):
        sns.lineplot(x=dates, y = rd_raw[:, i])
    plt.legend(instrument_columns,prop={'size': 6})


d_bar = rd_raw.shape[1]  # number of market risk drivers


# %% Step 3a: Quest for Invariance - Univariate
'''
PATH CHOICE: -> HISTORICAL

Justification: The Amount of Data appears to be large enough, to avoid a parametrical approach

Hier setze ich morgen an und baue Cross-Section Dings bums ein



Notizen:
    - Unterschiede in der Datenlänge kann ausgeglichen werden

'''

# Mehr Schritte einbauen !








### 1. Modell Random Walk
## Invariants from Random Walk 
random_walk_invariants = np.diff(rd_raw,axis=0)
## Invariance Tests
random_walk_invariance_test = {}
random_walk_suited = {}
for i,name in enumerate(instrument_columns):
    # Test for AC
    ac_levels, ac_crit = invariance_test_ellipsoid(random_walk_invariants[:,0],6)
    ac_checks = [1 for i in ac_levels if abs(i) > abs(ac_crit[1])]
    ac_check = 1 if len(ac_checks) > 0 else 0
    ac_test = {"AC_check": ac_check,
               "AC_values": ac_levels, 
               "AC_crit": ac_crit}
    
    # Test for Heteroskedasticity
    hc_levels, hc_crit = invariance_test_ellipsoid(np.abs(random_walk_invariants[:,0]),6)
    hc_checks = [1 for i in hc_levels if abs(i) > abs(hc_crit[1])]
    hc_check = 1 if len(hc_checks) > 0 else 0
    hc_test = {"HC_check": hc_check,
               "HC_values": hc_levels, 
               "HC_crit": hc_crit}
    
    # Evaluation of Modell
    if (ac_check == 1) or (hc_check == 1):
        random_walk_suited[name] = 0  # RW NICHT geeignet für Variable
    else:
        random_walk_suited[name] = 1  # RW geeignet für Variable
    
    # Save Results
    random_walk_invariance_test[name] = {"AC":ac_test,"HC":hc_test}
    
# Zwei Tests noch einbauen
# 1) invariance_test_ks,  2) invariance_test_copula

# TODO: GARCH Specification Selection (BIC)


### 2. Modell: GARCH
garch_invariants = np.zeros((t_bar - 1, n_bar))
db_garch_param = {} 
db_garch_values = {}
db_nextstep = {}
for i, name in enumerate(instrument_columns):
    garch = arch_model(random_walk_invariants[:, i], vol='garch', p=1, o=0, q=1, rescale=False)  # define GARCH(1,1) model
    garch_fitted = garch.fit(disp='off')  # fit GARCH(1,1)
    garch_fitted
    par = garch_fitted._params  # parameters
    sigma2 = garch_fitted._volatility**2  # ! PRÜFEN
    # sigma2 = garch_fitted._volatility # realized variance TODO: !! DAS HIER IST DIE VARIANCE????
    eps = garch_fitted.std_resid  # GARCH(1,1) invariants np.mean(garch_fitted.std_resid)
    # store invariants, GARCH(1,1) parameters and next-step function
    
    garch_invariants[:, i] = eps
    db_garch_param[i] = dict(zip(['mu', 'c', 'a', 'b'], par))
    db_garch_values[i] = dict(zip(['sig2_' + str(t).zfill(3) for t in range(t_bar)],sigma2))
    db_nextstep[i] = 'GARCH(1,1)'

# np.std(random_walk_invariants[:,8])

np.mean(garch_invariants,axis=0)
garch_fitted._volatility


## Invariance Tests for Garch
Garch_invariance_test = {}
Garch_suited = {}
for i,name in enumerate(instrument_columns):
    # Test for AC
    ac_levels, ac_crit = invariance_test_ellipsoid(garch_invariants[:,0],6)
    ac_checks = [1 for i in ac_levels if abs(i) > abs(ac_crit[1])]
    ac_check = 1 if len(ac_checks) > 0 else 0
    ac_test = {"AC_check": ac_check,
               "AC_values": ac_levels, 
               "AC_crit": ac_crit}
    
    # Test for Heteroskedasticity
    hc_levels, hc_crit = invariance_test_ellipsoid(np.abs(garch_invariants[:,0]),6)
    hc_checks = [1 for i in hc_levels if abs(i) > abs(hc_crit[1])]
    hc_check = 1 if len(hc_checks) > 0 else 0
    hc_test = {"HC_check": hc_check,
               "HC_values": hc_levels, 
               "HC_crit": hc_crit}
    
    # Evaluation of Modell
    if (ac_check == 1) or (hc_check == 1):
        Garch_suited[name] = 0  # GARCH NICHT geeignet für Variable
    else:
        Garch_suited[name] = 1  # GARCH geeignet für Variable
    
    # Save Results
    Garch_invariance_test[name] = {"AC":ac_test,"HC":hc_test}

### Result of 3a
'''
### Result pro Risk Driver über Angabe des:
    1) Modells
    2) Modell Specification (Lags, etc.)
    3) Modell Parametern
    3) Fitted Values
# Welches modell?
# WElche Parameter wo verwendet
# Starting Values für Modells

'''
results_3a = {}
for i, name in enumerate(instrument_columns):
    results_3a[name] ={
        "model":"Garch", 
        "model_spec": [] , # Offen
        "param": db_garch_param[i],
        "values": db_garch_values[i],
        "invariants": garch_invariants}
                    
    
'''
GGf. müssen wir das als dictionary für jeden RD machen... immer geschlossen
'''



# %% Step 3b: Quest for Invariance - Multivariate
'''
Umfasst:
    1) Flexible Probability Determination (falls benötigt)
    2) Berechnung der Joint Invariants Distribution
'''

### Flexible Probabilities
## Prior - Exponentional Time Conditioning
########## inputs (you can change them) ##########
tau_prior_hl = t_bar / 2 # prior half-life
##################################################
t_bar_inv = garch_invariants.shape[0]

p_tau_hl_prior = np.exp(-(np.log(2)/tau_prior_hl)*abs(t_bar_inv - np.arange(0, t_bar_inv)))  # prior probabilities
p_tau_hl_prior = p_tau_hl_prior/sum(p_tau_hl_prior)  # rescaled probabilities

### Fully Flexible Sampling










## State-Conditioning
p_t = p_tau_hl_prior
'''
Erstmal ausgelassen
'''
plt.plot(p_t)

# %% Step 3c: Quest for Invariance - Projection
'''
Umfasst:
    1) Ziehung der Invariants für Projection
    2) Verwendung von Next-Step Functions für die Aggregation über Zeit
'''

########## inputs (you can change them) ##########
m_bar = 12  # number of monthly monitoring times 
j_bar = 10000  # number of projection scenarios
##################################################

# initialize increments and dispersion parameters for GARCH(1,1) 
dx_thor = np.empty((j_bar, m_bar + 1, d_bar))  #  dx_thor[:,0,:] <- Current Time
sigma2_garch = np.empty((j_bar, m_bar + 1, d_bar))  # dx_thor[:,0,:] <- Current Time
dx_thor[:, 0, :] = rd_raw[-1, :] - rd_raw[-2, :]  # current values of increments
sigma2_garch[:, 0, :] = [list(results_3a[i]["values"].values())[-1] for i in results_3a.keys()]  # current values of dispersion parameters
'''
Brauche die Current Values, da das Garch diese Elemente von Info-Set in Xt+1 sind
'''


# initialize invariant scenarios
eps_proj = np.zeros((j_bar, d_bar))  # Projections 

# initialize forecast risk drivers
x_thor = np.empty((j_bar, m_bar + 1, d_bar))
x_thor[:, 0, :] = rd_raw[-1, :]  # risk drivers at current time

for m in range(m_bar):
    # bootstrap routine
    # 1. subintervals
    s = np.r_[np.array([0]), np.cumsum(p_t)]  # TODO: Hier haben wir das Problem: Vix verändert sich net! 
    # 2. scenarios
    # a. uniform scenarios
    u = np.random.rand(j_bar)  # uniform scenario
    # b. categorical scenarios
    ind = np.digitize(u , bins=s) - 1  # ACHTUNG, 1: ist glaube ich nicht richtig und führt zu verzerrungen bei der Wahl
    # c. shock scenarios
    eps_proj = garch_invariants[ind, :]
    '''
    Wir ziehen aus den Rows -> Korrelationen bleiben erhalten
    '''
    # apply next-step functions
    for i, name in enumerate(instrument_columns):
        # Unpack Garch Parameter
        mu = results_3a[name]["param"]["mu"]
        c = results_3a[name]["param"]["c"]
        a = results_3a[name]["param"]["a"]
        b = results_3a[name]["param"]["b"]

        sigma2_garch[:, m + 1, i] = c + b * sigma2_garch[:, m, i] + a * (dx_thor[:, m, i] - mu)**2

        # risk drivers increments
        dx_thor[:, m + 1, i] = mu + np.sqrt(sigma2_garch[:, m + 1, i])*eps_proj[:, i]
        # np.std(dx_thor[:, 2 , 0])

        # projected risk drivers
        x_thor[:, m + 1, i] = x_thor[:, m, i] + dx_thor[:, m + 1, i]



        # risk drivers modeled as random walk: options
        # else:
            # projected risk drivers
            # x_thor[:, m + 1, d] = x_thor[:, m, d] + eps_proj[:, d]
        '''
        Das muss eigentlich für jeden Typ von Modell gemacht werden
        '''

p_scenario = np.ones(j_bar)/j_bar  # projection scenario probabilities


# Next Step Function -> mit variablen element 



# %% Step 4: Repricing
'''
Across all Steps
'''
pi_t_hor_path = np.zeros((j_bar,m_bar + 1, n_bar))

## P&L scenario in one year
pi_t_hor_one_year = np.zeros((j_bar, n_bar))
## P&L scenario in one year
pi_t_hor_six_month = np.zeros((j_bar, n_bar))

# lin_ret_t_hor = zeros((j_bar, n_bar))  # Eigene Ergänzung
for i, name in enumerate(instrument_columns):
    for m in range(1,m_bar + 1):
        pi_t_hor_path[:,m, i] = v_t_now[i,0]*(np.exp(x_thor[:, m, i] - x_thor[:, 0, i]) - 1)  

x_thor.shape[1]

m = 12

## P&L scenario in one year
pi_t_hor_one_year = pi_t_hor_path[:, -1, :]


pi_t_hor_path.shape[1]


    # lin_ret_t_hor[:, n] = exp(x_t_hor[:, -1, n] - x_t_hor[:, 0, n]) - 1  # Eigene Ergänzung


# Relative P&L
pi_t_hor_rel = pi_t_hor_one_year / v_t_now.T
np.mean(pi_t_hor_rel,axis=0)

corr1 = np.corrcoef(pi_t_hor_path[:,9,:].T)
corr2 = np.corrcoef(pi_t_hor_path[:,-1,:].T)


# %% Step 8: Portfolio Optimization

### Input Parameters
e_r = np.mean(pi_t_hor_rel,axis=0)

cov = np.cov(pi_t_hor_rel.T)
corr = np.corrcoef(pi_t_hor_rel.T)

# Vola ist viel zu hoch -> ggf. auch mal fragen


a = np.mean((pi_t_hor_path[:,:,:] / 100),axis=0)
b = np.cov((pi_t_hor_path[:,12,:] / 100).T)
# Std. Vola -> zu hoch
plt.plot(a.T)


np.mean(pi_t_hor_path[:,12,:],axis=0)




### Vergleich
ret = np.log(data_raw.iloc[1:,:]).diff().dropna()

aaaa = np.cov(ret.values.T, aweights = p_t[1:])

np.sqrt(np.diag(12*aaaa))


aaa = random_walk_invariants[:,0]




 
