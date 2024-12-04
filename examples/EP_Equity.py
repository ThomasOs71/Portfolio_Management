# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:25:43 2024

@author: Thomas
"""

### External
import os
import pickle
import pandas as pd
import numpy as np
import scipy.stats as stats

### Internal
from pm_modules.estimation.joint_dist import fit_locdisp_mlfp
from pm_modules.estimation.fin_engineering import risk_driver_equity


# %% Data Load: Equities
## Load from Pickle
os.chdir(r'C:\Projects\VENV\DataStream_DL\Sample_Data')
with open("equity", 'rb') as file:
    data = pickle.load(file)

## Obtain Data Information
data_sheet = ["World_Equity"]

## Data Equity
equity_data = {}
for i, name in enumerate(data_sheet):
    equity_data[name] = data[i].unstack(-2).resample("ME").last()

## Log-Returns
data = equity_data["World_Equity"]
data_moments = risk_driver_equity(data).diff().dropna().loc[:,"MSRI"]

## Moments of Log-Returns
nu_ = 5
moments = fit_locdisp_mlfp(data_moments.values,nu = nu_)

## RVS
np.random.seed(seed=233423)
sim = stats.multivariate_t.rvs(moments["mu"].squeeze(),
                               moments["sigma2"],moments["nu"],100000)

# Viele Draws für gute Annäherung an unterstellte Verteilung
# Alternative ist Moment Matching

# Moments
mean_ = np.mean(sim,axis=0)
cov_ = np.cov(sim.T)
skew_ = stats.skew(sim)
kurt = stats.kurtosis(sim,axis=0)


# %% Entropy Pooling

### Original

### Sequential (H1)


### Sequential (H2)





