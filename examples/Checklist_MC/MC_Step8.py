    # -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:14:49 2024

@author: Thomas


s_checklist_montecarlo_step08
This case study performs Step 8 of the "Checklist", namely Construction, 
    to determine the optimal buy-and-hold allocation or dynamic policy 
    
    given our target measure of satisfaction and 
    a set of investments contraints (Read more).

Was fehlt hier noch:
    1) Parameter Uncertainty
    2) Expected Returns
    3) Benchmark?

    
"""
# %% Packages
from numpy import append, arange, array, argmax, eye, floor, max, min, r_, sum, zeros
import cvxopt
from pandas import DataFrame, read_csv
from seaborn import histplot, lineplot, scatterplot
from matplotlib.pyplot import axvline, fill_between, legend, get_cmap, show
from spectral_index import spectral_index

# %% Load data
# values of stocks, S&P 500, options and bonds at current time (Montecarlo checklist - Step 2)
# import db_v_tnow database
db_v_tnow = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_v_tnow.csv')
v_t_now = db_v_tnow.values.squeeze()

# scenario probabilities forecast (Montecarlo checklist - Step 3)
# import db_scenario_probs database
db_scenprob = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_scenario_probs.csv')
p_scenario = db_scenprob['p_scenario'].values

# ex-ante P&Ls of stocks, S&P 500, options and bonds at the horizon (Montecarlo checklist - Step 4)
# import db_pricing database
db_pricing = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_pricing.csv')
pi_t_now_t_hor = db_pricing.values

# portfolio ex-ante performance (Montecarlo checklist - Step 5)
# import db_exante_perf database
db_exante_perf = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_exante_perf.csv')
y_h = db_exante_perf.values.squeeze()

# additional variables (Montecarlo checklist - Step 2)
# import db_riskdrivers_tools database
db_riskdrivers_tools = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_riskdrivers_tools.csv')
n_stocks = len(db_riskdrivers_tools.stocks_names.dropna())
n_bonds = int(db_riskdrivers_tools.n_bonds[0])
n_bar = n_stocks + n_bonds + 3

# current value of holdings and cash (Montecarlo checklist - Step 5)
# import db_aggregation_tools database
db_aggregation_tools = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_aggregation_tools.csv')
v_h_t_now = db_aggregation_tools['v_h_tnow'][0]

# satisfaction measures (Montecarlo checklist - Step 6)
# import db_quantile_and_satis database
db_quantile_and_satis = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_quantile_and_satis.csv')
c_subquantile = db_quantile_and_satis['c_subquantile'][0]

# %% 1. Solving the first step of the mean-variance approach (Read more) 
'''
Das hier ist in absolut P&L?
-> Das ist interessant. Das klappt ebenfalls im Falle absoluter P&L
-> Später dann Umrechnung in Weights
'''
############## inputs (you can change them) ##############
lambda_inf = 1e-9  # minimum value of parameter lambda
lambda_sup = 1e-6  # maximum value of parameter lambda
lambda_step = 1e-9  # step in lambda grid
v_stocks_min = 200e6  # minimum budget to invest in stocks
##########################################################

### Optimization Parameter
## expectation and covariance of portfolio P&L
e_pi = p_scenario@pi_t_now_t_hor  
cv_pi = ((pi_t_now_t_hor - e_pi).T*p_scenario)@(pi_t_now_t_hor - e_pi)

## grid of parameters for mean-variance frontier
lambda_grid = arange(lambda_inf, lambda_sup, lambda_step)
l_bar = lambda_grid.shape[0]

### equality constraints
# budget constraint: h'*v_tnow = v_h_tnow
a_budget = v_t_now.reshape(1, -1)
b_budget = array(v_h_t_now)
# constraint: do not invest in the S&P
a_sp = zeros((1, n_bar))
a_sp[0, n_stocks] = 1
b_sp = array(0)
# combined equality constraints
a = cvxopt.matrix(r_[a_budget, a_sp])
b = cvxopt.matrix(r_[b_budget, b_sp])

### Inequality constraints
# holdings must be nonnegative (no short sale)
u_no_short = -eye(n_bar)
v_no_short = zeros(n_bar)
# investment composition constraint: invest at least v_stocks_min in stocks
u_comp = -append(v_t_now[:n_stocks], zeros(n_bonds + 3)).reshape(1, -1)
v_comp = -array(v_stocks_min)
# combined inequality constraints
u = cvxopt.matrix(r_[u_no_short, u_comp])
v = cvxopt.matrix(r_[v_no_short, v_comp])

# quadratic programming problem
h_lambda = zeros((l_bar, n_bar))
expectation = zeros(l_bar)
variance = zeros(l_bar)
cvxopt.solvers.options['show_progress'] = False
for l in range(l_bar):  # Loop über Lambda
    # objective function
    p_opt = cvxopt.matrix(2*lambda_grid[l]*cv_pi)
    q_opt = cvxopt.matrix(-e_pi)
    # solve quadratic programming problem
    h_lambda[l, :] = array(cvxopt.solvers.qp(p_opt, q_opt, u, v, a, b)['x']).squeeze()
    expectation[l] = e_pi@h_lambda[l, :].T
    variance[l] = h_lambda[l, :]@cv_pi@h_lambda[l, :].T

# portfolio weights
w_lambda = (h_lambda*v_t_now)/v_h_t_now

# %% 2. Solving the second step of the mean-variance approach (Read more) 
'''
Berechnet das beste Portfolio
'''
# spectrum function
def spectr_q_bar(x):
    return (1/(1 - c_subquantile))*(0 <= x and x <= 1 - c_subquantile)
'''
Function für Spectral Index
'''


# sub-quantile measure of satisfaction
subq_pih_lambda = zeros(l_bar)
for l in range(l_bar):
    subq_pih_lambda[l], _ = spectral_index(spectr_q_bar, pi_t_now_t_hor, p_scenario, h_lambda[l, :])

# quasi-optimal portfolio
idx_lambda_star = argmax(subq_pih_lambda) 
lambda_star = lambda_grid[idx_lambda_star]  # risk adversion
h_qsi = floor(h_lambda[idx_lambda_star, :].round(20))  # holdings 
subq_pih_qsi = subq_pih_lambda[idx_lambda_star]  # satisfaction 
y_h_subq_qsi = pi_t_now_t_hor@h_qsi  # ex-ante performance

##################################################### plots #####################################################
# optimized portfolio ex-ante P&L distribution
histplot(x=y_h, weights=p_scenario, bins=50, label='Current holdings').set(xlabel='$Y_h$ (million USD)');
histplot(x=y_h_subq_qsi, weights=p_scenario, bins=50, label='Optimal holdings'); legend(); show();

# mean-variance efficient frontiere 
lineplot(x=variance, y=expectation, label='Efficient frontier').set(xlabel='Variance', ylabel='Expectation')
scatterplot(x=[variance[idx_lambda_star]], y=[expectation[idx_lambda_star]], label ='Optimal holdings'); show();

# portfolio composition for each variances on efficient frontiere 
instruments = list(db_v_tnow)
colors = get_cmap('Spectral')(arange(n_bar)/n_bar)[:, :3]
for n in range(n_bar):
    if n == 0:
        fill_between(variance, w_lambda[:, n], zeros(l_bar), color=colors[n, :], label=instruments[n])
    else:
        fill_between(variance, sum(w_lambda[:, :n + 1], axis=1), sum(w_lambda[:, :n], axis=1), 
                         color=colors[n, :], label = instruments[n])        
axvline(x=variance[idx_lambda_star], color='k'); legend(); show();

# %% Save data
# quasi-optimal portfolio
out = DataFrame({db_v_tnow.columns[i]: h_qsi[i] for i in range(len(h_qsi))}, index = [0])
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_final_portfolio.csv', index=False); del out;
