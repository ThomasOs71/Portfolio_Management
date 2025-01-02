# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:30:17 2024

@author: Thomas

s_checklist_montecarlo_step02

Wichtig:    Hier wird implizit eine Flat / Constant YC angenommen,
            daher ist die Yield Curve NICHT modelliert!

This case study performs Step 2 of the "Monte Carlo Checklist", 
namely Risk drivers identification, for a market consisting of:
    1) five stocks, 
    2) the S&P 500 index, 
    3) two options and 
    4) two defaultable corporate bonds.
    
"""

from numpy import array, arange, concatenate, datetime64, full, fill_diagonal,\
                  intersect1d, max, nan, sqrt, where, zeros
from pandas import concat, DataFrame, read_csv, Series, MultiIndex
from matplotlib.pyplot import imshow, xlabel, xticks, ylabel, yticks

from bootstrap_nelson_siegel import bootstrap_nelson_siegel
from aggregate_rating_migrations import aggregate_rating_migrations


# %% Load data
######################### inputs (you can change them) #########################
t_first = datetime64('2009-11-02')  # set start date for data selection 
t_init = datetime64('2012-08-30')  # set initial portfolio construction date 
t_now = datetime64('2012-08-31')  # set current date
'''
For the Corporate Bonds, the Data is short (3.1.2012)
'''
################################################################################

### Market
'''
Für Stocks, S&P500 und Options sind die Daten DIREKT die Risk Driver
Basiert auf der Checklist für den Historical Approach
'''

## Import Values at Current and Initial Time
'''
Import Values at Initial Time (t_init):
    1) Values of stocks, 
    2) S&P 500 and 
    3) Options at initial time
'''
db_v_tinit_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_v_tinit_historical.csv')
n_bar = db_v_tinit_historical.shape[1]  # number of instruments 
v_t_init = dict(zip(range(n_bar), db_v_tinit_historical.values.squeeze()))  # initial values of instruments

'''
Import Values: Current Time (t_now): 
    1) Values of Stocks, 
    2) S&P 500 and 
    3) Options
'''
db_v_tnow_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_v_tnow_historical.csv')
v_t_now = dict(zip(range(n_bar), db_v_tnow_historical.values.squeeze()))  # current values of instruments
v_t_now_names = dict(zip(range(n_bar), db_v_tnow_historical.columns.values))  # instruments names 

### Risk Driver  
'''
Import Risk Drivers: (Calculated in Historical Checklist)
    1) Stocks: Log Values (5)
    2) S&P 500: Log Values (1)
    3) Options Log Implied Vola (90)
'''
db_riskdrivers_series_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_riskdrivers_series_historical.csv', 
                                               index_col=0, parse_dates=True)

d_bar = db_riskdrivers_series_historical.shape[1]  # number of market risk drivers
dates = db_riskdrivers_series_historical.index  # date range of interest
t_bar = len(dates)  # length of market risk drivers time series

# risk driver series
db_risk_drivers = {}  
for d in range(d_bar):
    db_risk_drivers[d] = array(db_riskdrivers_series_historical.iloc[:, d])
risk_drivers_names = dict(zip(range(d_bar), db_riskdrivers_series_historical.columns.values))


### Time Series and Data Information(for Calculation of Risk Drivers for Corporate Bonds)
'''
Corporate Bonds zur Berechnung der Bond Curve.
Import von:
    1) Time Series
    2) Information of Bonds (Expiry, Cupon, etc.)

TODO:   Umfasst das dann sämliche Zinselemente? 
        Unter der Annahme flat/constant YC (0.01), dann vermutlich schon.
'''

## Data Load: Corporate bonds: JPM 
# Corporate Bonds (tau: 1-20Y)
db_jpm = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\JPM/data.csv', 
                     index_col=['date'], parse_dates=['date'])

# import JPM/params database
jpm_param = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\JPM/params.csv', 
                        index_col=['expiry_date'], parse_dates=['expiry_date'], date_format="mixed")
jpm_param['link'] = ['dprice_' + str(i) for i in range(1, jpm_param.shape[0] + 1)]  # coupons and maturities

## Data Load: Corporate bonds: GE
# Corporate Bonds (tau: 1-20Y)
db_ge = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\GE/data.csv',
                    index_col=['date'], parse_dates=['date'])

# select same date range for both bonds 
index_dates_bonds = where((db_ge.index >= t_first)&(db_ge.index <= t_now))
dates_bonds = intersect1d(db_ge.index[index_dates_bonds], db_jpm.index).astype('datetime64[D]') 

# import GE/params database 
ge_param = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\GE/params.csv', 
                       index_col=['expiry_date'], parse_dates=['expiry_date'], date_format="mixed")
ge_param['link'] = ['dprice_' + str(i) for i in range(1, ge_param.shape[0] + 1)]  # coupons and maturities



################################################# credit #################################################
'''
Import von Credit Data:
    1) Data für Ratings (Evolution Ratings von verschiedenen Creditor über langen Zeitraum)
    2) Rating Skala (A-D)

TODO / ANmerkung: Für meine Belange irrelevant. Nehme ich nur mit, zum Durchlauf der ELemente
'''
# risk drivers
# import db_ratings/data database 
db_ratings_data = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\Credit\data.csv', 
                           parse_dates=['date'])

# import db_ratings/params.csv database
ratings = array(read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\Credit\params.csv', 
                         index_col=0).index)
c_bar = len(ratings)  # number of ratings classes


# %% 1. Market risk drivers for GE and JPM corporate bonds (Read more) 
'''
Berechnung der Yield Curve of Corporate Bonds. 
Basierend auf den Bond-Preisen des jeweiligen Emittenten (nach Berücksichtigung) der Cupons-Struktur
ergibt sich die Berechnung der Nelson-Siegel Parameter.
'''

## GE 
v_bond_ge = db_ge.loc[db_ge.index.isin(dates_bonds)]/100  # rescaled dirty prices
coupons_ge = array(ge_param.coupons/100)  # rescaled coupons rates 
coupons_dates_ge = ge_param.index.values.astype('datetime64[D]')  # coupons dates 
x_ge = bootstrap_nelson_siegel(v_bond_ge.values, dates_bonds, coupons_ge, coupons_dates_ge)  # Nelson-Siegel parameters
'''
Nelson Siegel unter Berücksichtigugn der Cupons.
'''
# Nelson-Siegel Parameters as Risk Drivers + Hinzufügen zur Dataframe der RD
for d in range(x_ge.shape[1]):
    if d == 3:
        db_risk_drivers[d_bar + d] = sqrt(x_ge[:, d])
    else:
        db_risk_drivers[d_bar + d] = x_ge[:, d]
    risk_drivers_names[d_bar + d] = 'ge_bond_nel_sieg_theta_' + str(d + 1)

## JPM 
v_bond_jpm = db_jpm.loc[db_ge.index.isin(dates_bonds)]/100  # rescaled dirty prices
coupons_jpm = array(jpm_param.coupons/100)  # rescaled coupons rates 
coupons_dates_jpm = jpm_param.index.values.astype('datetime64[D]')  # coupons dates 
x_jpm = bootstrap_nelson_siegel(v_bond_jpm.values, 
                                dates_bonds, 
                                coupons_jpm, 
                                coupons_dates_jpm)  # Nelson-Siegel parameters

# Nelson-Siegel parameters as risk drivers + Hinzufügen zur Dataframe der RD
for d in range(x_jpm.shape[1]):
    if d == 3:
        db_risk_drivers[d_bar + x_ge.shape[1] + d] = sqrt(x_jpm[:, d])
    else:
        db_risk_drivers[d_bar + x_ge.shape[1] + d] = x_jpm[:, d]
    risk_drivers_names[d_bar + x_ge.shape[1] + d] = 'jpm_bond_nel_sieg_theta_' + str(d + 1)

# missing values replaced with NaN's
for d in range(d_bar, d_bar + 8):
    db_risk_drivers[d] = concatenate((zeros(t_bar - len(dates_bonds)), db_risk_drivers[d]))
    db_risk_drivers[d][:t_bar - len(dates_bonds)] = nan

# %% 2. Aggregated credit risk drivers for GE and JPM corporate bonds (Read more) 

################# inputs (you can change them) #################
# credit ratings:
# AAA (1), AA (2), A (3), BBB (4), BB (5), B (6), CCC (7), D (8)
# initial credit rating
ratings_t_now = array([6,   # GE (corresponding to B)
                       4])  # JPM  (corresponding to BBB)
# dates for aggregate credit risk drivers
t_first_credit = datetime64('1995-01-01')  # start date 
t_last_credit = datetime64('2004-12-31')  # end date 
################################################################

# aggregated credit risk drivers
dates_credit, n_obligors, n_cum_trans, *_ = aggregate_rating_migrations(db_ratings_data, ratings, t_first_credit, t_last_credit) 
 
# number of obligors in each rating at each time
credit_types, credit_series = {}, {}
for c in arange(c_bar):
    credit_types[c] = 'n_oblig_in_state_' + ratings[c]
    credit_series[c] = n_obligors[:, c]

# cumulative number of credit rating transitions up to certain time for each pair of rating buckets
d_credit = len(credit_series) 
for i in arange(c_bar):
    for j in arange(c_bar):
        if i != j:
            credit_types[d_credit] = 'n_cum_trans_' + ratings[i] + '_' + ratings[j]
            # cumulative number of transitions
            credit_series[d_credit] = n_cum_trans[:, i, j]  
            # number of credit risk drivers 
            d_credit = len(credit_series)
            
display(DataFrame(n_cum_trans[-1, :, :], index=ratings, columns=ratings).map(lambda x: f"{x}"))

########################################### plot ###########################################
# cumulative number of credit rating transitions 
same = full((c_bar, c_bar), nan)
fill_diagonal(same, 1);
imshow(same, cmap='Greys')
imshow(array(n_cum_trans[-1, :, :], dtype=float), cmap='Blues', vmin=0, vmax=max(n_obligors))  
xlabel("To"); ylabel("From"); xticks(arange(0,8), ratings); yticks(arange(0,8), ratings);


# %% Save data
############################ inputs (you can change them) ############################
t_ge_end = datetime64('2013-09-16')  # maturity date of GE coupon bond to extract
t_jpm_end = datetime64('2014-01-15')  # maturity date of JPM coupon bond to extract
######################################################################################

# time series of market risk drivers
out = DataFrame({risk_drivers_names[d]: db_risk_drivers[d] for d in range(len(db_risk_drivers))}, index=dates)
out = out[list(risk_drivers_names.values())]
out.index.name = 'dates'
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_riskdrivers_series.csv'); del out;

# GE
coupon_ge = ge_param.loc[t_ge_end, 'coupons']/100  # coupon rate
v_t_init[n_bar] = v_bond_ge.loc[t_init, ge_param.loc[t_ge_end, 'link']]  # dirty price at initial time
v_t_now[n_bar] = v_bond_ge.loc[t_now, ge_param.loc[t_ge_end, 'link']]   # dirty price at current time
v_t_now_names[n_bar] = 'ge_bond'

# JPM 
coupon_jpm = jpm_param.loc[t_jpm_end, 'coupons']/100  # coupon rate
v_t_init[n_bar + 1] = v_bond_jpm.loc[t_init, jpm_param.loc[t_jpm_end, 'link']]  # dirty price at initial time
v_t_now[n_bar + 1] = v_bond_jpm.loc[t_now, jpm_param.loc[t_jpm_end, 'link']]  # dirty price at current time
v_t_now_names[n_bar + 1] = 'jpm_bond'

# values of stocks, S&P 500, options and bonds at initial time
out = DataFrame({v_t_now_names[n]: Series(v_t_init[n]) for n in range(len(v_t_init))})
out = out[list(v_t_now_names.values())]
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_v_tinit.csv', index=False); del out;

# values of stocks, S&P 500, options and bonds at current time
out = DataFrame({v_t_now_names[n]: Series(v_t_now[n]) for n in range(len(v_t_now))})
out = out[list(v_t_now_names.values())]
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_v_tnow.csv', index=False); del out;

# credit risk drivers
out = DataFrame({credit_types[d]: credit_series[d] for d in range(d_credit)}, index=dates_credit)
out = out[list(credit_types.values())]
out.index.name = 'dates'
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_riskdrivers_credit.csv'); del out;

# additional variables needed for subsequent steps
db_riskdrivers_tools_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_riskdrivers_tools_historical.csv')
out = DataFrame({'n_bonds': Series(2), 't_end_ge': Series(t_ge_end), 'coupon_ge': Series(coupon_ge), 
                 't_end_jpm': Series(t_jpm_end),'coupon_jpm': Series(coupon_jpm), 'c_bar': Series(c_bar),
                 't_last_credit': Series(t_last_credit),'d_credit': Series(d_credit), 
                 'ratings_t_now': Series(ratings_t_now), 'ratings': Series(ratings)})

# add variables from Historical checklist
out = concat([db_riskdrivers_tools_historical, out], axis=1, sort=False);
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_riskdrivers_tools.csv', index=False); del out;
