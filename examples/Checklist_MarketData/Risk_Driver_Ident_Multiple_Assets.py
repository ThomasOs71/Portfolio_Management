# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:50:14 2024

@author: Thomas
"""

# %% Packages
### External
import os
import pickle
import pandas as pd

### Internal
from pm_modules.estimation.fin_engineering import risk_driver_equity, risk_driver_fe_nelson_siegel


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
            



# %% Data Load: Yield Curve
'''
Erstmal Fokus auf Sovereign Bonds
'''
## Load from Pickle
os.chdir(r'C:\Projects\VENV\DataStream_DL\Sample_Data')
with open("sov", 'rb') as file:
    data = pickle.load(file)
    data = data[0]

## Obtain Data Information
fi_sheets = ["Sov_Bonds"]

# Sample Information
fi_data_info = {}
for i in fi_sheets:
    fi_data_info[i] = pd.read_excel("Sample_Data_Codes.xlsx",
                                         sheet_name=i)

'''
TODO: Hier morgen weitermachen
'''


## Data 
fi_data = {}
for i, name in enumerate(fi_sheets):
    fi_data[name] = data[i].unstack(-2).resample(freq).last()




### USA



'data'.keys()

### Euro





# %% Data Load: Foreign Exchange 


# %% Data Load: ALternatives






# %% Risk Driver Calculation:
    
### Equity
equity_rd = {}
for i, name in enumerate(equity_sheets):
    equity_rd[name] = risk_driver_equity(equity_data[name])
    
### Fixed Income

### Foreign Exchange    
    
### Commodities



# %% Risk Driver Testing - Routine
# - NAN in Data?
