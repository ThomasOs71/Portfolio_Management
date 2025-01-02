# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:17:58 2024

@author: Thomas


Step 8a: Portfolio optimization

This case study performs Step 8a of the Checklist, namely Portfolio optimization. 

The purpose of this step is to determine the optimal buy-and-hold allocation h, given :
    1) our target measure of satisfaction 
    2) and a set of investment constraints.

Here we compute a quasi-optimal buy-and-hold allocation h first using 
the two-step mean-variance approach: 
    Step I) Computing the efficient frontier, 
            
    Step II) Choosing the efficient allocation which maximizes the target satisfaction measure. 


Our satisfaction is measured by the sub-quantile (cVaR), and 
we face a range of investment constraints. 

Next, we limit the number of active positions, using backward selection. 

Finally, we take the transaction costs of the trades 
required to rebalance the portfolio into account, 
using forward selection to iteratively select the most effective trades.

"""

# %% Prepare environment
### Packages
from numpy import append, arange, argmax, argsort, array, atleast_1d, average, cumsum, \
                  count_nonzero, cov, diag, eye, inf, isfinite, isscalar, linspace, max, min, \
                  ones, r_, round, setdiff1d, sort, sqrt, sum, trace, where, zeros
from numpy.linalg import inv
from scipy.linalg import sqrtm
from pandas import DataFrame, read_csv
from cvxpy import Variable, SOC, Problem, Minimize, CLARABEL, quad_form, sum as cp_sum
from seaborn import lineplot, scatterplot, color_palette, histplot
from matplotlib.pyplot import show, fill_between, gca, legend, axvline
from backward_selection import backward_selection
from forward_selection import forward_selection

### Load data
'''
From database db_valuation_historical created in script s_checklist_historical_step01
we upload:
    1) the number of stocks 
'''
db_valuation_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step2\db_valuation_historical.csv', index_col=0)
n_stocks = int(db_valuation_historical.loc['n_stocks'].values)  # number of stocks
n_bar = n_stocks + 2  # number of all instruments

'''
From database db_forecast_historical created in script s_checklist_historical_step03b,
we upload:
     1) the scenario probabilities 
'''
db_forecast_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step4/db_forecast_historical.csv', low_memory=False)
p_j = db_forecast_historical.p_scenario.dropna()  # probabilities

'''
From database db_value_aggregation_historical created in script s_checklist_historical_step05a,
we upload:
    1) the portfolio current value 
    2) and portfolio weights 
'''

db_value_aggregation_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step5b/db_value_aggregation_historical.csv', index_col=0)
w_h = db_value_aggregation_historical.w_h.values  # portfolio weights
v_h = db_value_aggregation_historical.v_h.iloc[0]  # portfolio current value

'''
From database db_exante_perf_historical created in script s_checklist_historical_step05b,
we upload:
    1) the ex-ante portfolio return scenarios and 
    2) the instrument return scenarios 
'''
db_exante_perf_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step6/db_exante_perf_historical.csv')
r_w = db_exante_perf_historical.r_w  # ex-ante portfolio return scenarios
r = db_exante_perf_historical.iloc[:, 1:].values  # instrument return scenarios

'''
From database db_quantile_and_satis_historical created in script s_checklist_historical_step06,
we upload:
    1) the confidence level for the sub-quantile 
'''
db_quantile_and_satis_historical = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step7/db_quantile_and_satis_historical.csv')
c = db_quantile_and_satis_historical['c'][0]  # confidence level for sub-quantile

# %% 1. Set the inputs 
'''
The first step in portfolio optimization is to set the inputs of the mean-variance optimization, 
namely the expected return and the return covariance.
'''

'''
Step 1: Compute the HFP mean mu and HFP covariance sigma2 
        of the instrument return distribution 
'''

# expectation and covariance of instrument returns
e_r_hat = average(r, weights=p_j, axis=0)  # expected return
cv_r_hat = cov(r.T, aweights=p_j, bias=True)  # return covariance

'''
Step 2: Shrinkage of the Covariance Matrix
        We set the input covariance matrix by applying Ledoit-Wolf shrinkage to the sample covariance, with 
'''
############# input (you can change it) #############
gamma_shrink = 0.5  # shrinkage factor for covariance
#####################################################

# shrink covariance
sigma2_target = (1/n_bar)*trace(cv_r_hat)*eye(n_bar)  # target covariance
cv_r = (1 - gamma_shrink)*cv_r_hat + gamma_shrink*sigma2_target  # shrinkage covariance

'''
Step 3: Estimate Equilibirum Returns
        We set the input mean vector by using the implied returns with standardized risk aversion (gamma)
        and the strategic portfolio set as the equally weighted portfolio.
'''
# implied expected returns
w_str = ones(n_bar)/n_bar  # equally-weighted allocation
e_r = cv_r@w_str  # implied expected returns


# %% 2. Compute efficient frontier 
'''
The portfolio must satisfy the following investment constraints on positions:
    1) full investment budget constraint;
    2) no short holdings;
    3) invest at least 95% of the portfolio weight in stocks.
    
    
Given the inputs and the constraints, the mean-variance efficient frontier provides 
us with a subset of allocations to optimize over, which satisfy the intuition 
that we want the portfolio with the lowest variance ("risk") for a given expected return ("reward").

We compute the mean-variance efficient frontier for a grid of values

        {0, ..., lambda_l,...,1} 

for the trade-off parameter lambda.
'''

############## input (you can change it) ##############
l_bar = 1000  # number of trade-off parameters to test
#######################################################

# optimization variable
x = Variable(n_bar) 

affine_constraints = [      
    # full investment constraint
    cp_sum(x) == 1.]

box_constraints = [
    # no short holdings constraint
    x >= 0,
    # 95% weight in stocks
    cp_sum(x[:n_stocks]) >= 0.95]

# solve quadratic programming problem
w_lambda = zeros((l_bar, n_bar))
expectation = zeros(l_bar)
variance = zeros(l_bar)
lambda_grid = linspace(0, 1, l_bar, endpoint=False)
for l in range(l_bar):
    # optimization problem
    prob = Problem(Minimize((1 - lambda_grid[l])*quad_form(x, cv_r) - (lambda_grid[l])*e_r.T@x),
                   affine_constraints + box_constraints)
    # solve optimization
    prob.solve(solver=CLARABEL)
    if prob.status == 'optimal':
        # optimal portfolio weights
        w_lambda[l, :] = array(x.value).squeeze()
        # portfolio expected return
        expectation[l] = e_r@w_lambda[l, :].T
        # variance of portfolio return
        variance[l] = w_lambda[l, :]@cv_r@w_lambda[l, :].T 
    else:
        # optimal portfolio weights
        w_lambda[l, :] = zeros(n_bar)
        # portfolio expected return
        expectation[l] = -inf
        # variance of portfolio return
        variance[l] = inf

##################################################### plot #####################################################
# mean-variance profile of efficient frontier
lineplot(x=variance, y=expectation, label='Efficient frontier').set(xlabel='Variance', ylabel='Expected return');


# %% 3. Select portfolio with highest satisfaction 
'''
The quasi-optimal portfolio w_lambda is the allocation on the efficient frontier 
that allows us to achieve the maximum satisfaction, as measured by the sub-quantile (cVaR).
'''

'''
Step 1: Compute the sub-quantile for each allocation on the efficient frontier.
'''

alpha = 1 - c  # sub-quantile level
# sub-quantile measure of satisfaction
subq_r_lambda = zeros(l_bar)
for l in range(l_bar):
    # portfolio ex-ante return
    perf = r@w_lambda[l]
    # scenario sort order
    scen_sort = argsort(perf)
    # sorted performance scenarios
    r_w_sort = perf[scen_sort]
    # sorted probabilities
    p_sort = p_j[scen_sort]
    # cumulative sums of sorted probabilities
    u_sort = append(0, cumsum(p_sort))
    # weights
    w_subq = (1/alpha)*p_sort*(u_sort[1:].round(5) <= round(alpha, 2))
    # rescaled weights
    w_subq = w_subq/sum(w_subq)
    # sub-quantile
    subq_r_lambda[l] = r_w_sort@w_subq
    
    
'''
Step 2: We select the allocation on the efficient frontier with the highest satisfaction.
'''
# quasi-optimal portfolio
idx_lambda_star = argmax(subq_r_lambda)  # index
lambda_star = lambda_grid[idx_lambda_star]  # risk aversion
w_qsi = w_lambda[idx_lambda_star, :]  # weights
subq_r_qsi = subq_r_lambda[idx_lambda_star]  # satisfaction
e_r_qsi = expectation[idx_lambda_star]  # expected return
v_r_qsi = variance[idx_lambda_star]  # portfolio return variance
print('w_qsi =', w_qsi.round(4))

################################################### plots ###################################################
# mean-variance profile of efficient frontier
scatterplot(x=[v_r_qsi], y=[e_r_qsi], color='r', label='Quasi-optimal holdings')
lineplot(x=variance, y=expectation, label='Efficient frontier').set(
         xlabel='Variance', ylabel='Expected return'); show()

# portfolio composition for each variance on efficient frontier
instruments = list(db_valuation_historical.index[:n_bar])
colors = color_palette("Spectral", n_colors=n_bar)
for n in range(n_bar):
    if n == 0:
        fill_between(variance, w_lambda[:, n], zeros(l_bar), color=colors[n], label=instruments[n])
    else:
        fill_between(variance, sum(w_lambda[:, :n + 1], axis=1), sum(w_lambda[:, :n], axis=1),
                     color=colors[n], label=instruments[n])
lineplot(x=[variance[idx_lambda_star], variance[idx_lambda_star]], y=[-10, 10], 
         orient='y', color='black').set(xlabel='Variance', ylabel='Weights', 
                                        xlim=[min(variance), max(variance)], 
                                        ylim=[0, 1]);
axvline(v_r_qsi,0,c="red")  # Selected PF



# %% 4. Apply position constraints 
'''
In order to reduce operating costs, we wish to minimize 
    the number of active positions in the portfolio. 
To do this effectively we proceed in two steps.

Step 1) Impose an additional constraint on the number of positions. 
        To decide on the instruments to include, 
        we apply the selection routine using backward selection. 
        
        The evaluation function g is defined with:
        1) Fixed target expectation:
                 e = w'_lam @ mu_r equal to the expected return of 
                 the quasi-optimal portfolio w_lam. 
                 
        2) Relaxed Budget Contraint: w'1 <= 1
                 We relax the equality in the budget constraint 
                 in the evaluation function to be an inequality, 
                 that is we require, to ensure the optimization is well-defined.
'''


# Cardinality Function
def g(s_k=None):
    # selection set
    if s_k is None:
        s_k = arange(n_bar - 1)
    elif isscalar(s_k):
        s_k = array([s_k - 1])
    else:
        s_k = s_k - 1
    # sorted selection set
    s_k = sort(s_k)
    # length of selection set
    k = len(s_k)

    # optimization variable
    x = Variable(n_bar)
    
    # expectation constraint
    affine_constraints = [e_r@x == e_r_qsi]  # Fixed Target Expectation
    
    box_constraints = [      
        # budget constraint
        cp_sum(x) <= 1.,  # Relaxed Bzdget Constraint
        # no short holdings constraint
        x >= 0,
        # 95% weight in stocks
        cp_sum(x[:n_stocks]) >= 0.95]

    # non-active positions
    no_holdings = setdiff1d(range(n_bar), s_k)
    # position constraint (keep only selected positions)
    if k < n_bar:
        position_constraint = [x[no_holdings] == zeros(n_bar - k)]
    else:
        position_constraint = []
    
    # optimization problem
    prob = Problem(Minimize(quad_form(x, cv_r)), affine_constraints +\
                   box_constraints + position_constraint)
    # solve optimization
    prob.solve(solver=CLARABEL)
    if prob.status == 'optimal':
        # optimal portfolio weights
        w = array(x.value).squeeze()
        # variance of portfolio return
        v = (w.T@cv_r@w).squeeze()
        return w, v
    else:
        return zeros(n_bar), array(inf)

# select stocks using backward selection
w_bwd, v_r_bwd, s_k_bwd = backward_selection(g, n_bar)
w_bwd = array(w_bwd)  # portfolios
w_bwd = w_bwd[isfinite(v_r_bwd.squeeze())]  # select valid allocations
w_bwd = where(w_bwd > 1e-12, w_bwd, 0)  # set spurious small weights to zero
k_bwd = count_nonzero(w_bwd, axis=1)  # number of active positions
e_r_bwd = w_bwd@e_r  # portfolio expected returns
v_r_bwd = diag(w_bwd@cv_r@w_bwd.T)  # portfolio return variances
'''
We select the portfolio with k = 5 active positions by observing the 
trade-off between the ex-ante portfolio return variance and 
the number of active positions.
 '''
########## input (you can change it) ##########
k_pos = 5  # number of active positions
###############################################

# select portfolio
s_pos = s_k_bwd[k_pos-1]  # active positions
w_pos = w_bwd[k_bwd == k_pos]  # portfolio weights
e_r_pos = e_r_bwd[k_bwd == k_pos]  # expected portfolio return
v_r_pos = v_r_bwd[k_bwd == k_pos]  # portfolio return variance
print('w_pos =', w_pos.round(4))

############################################### plots ###############################################
# mean-variance profile of efficient frontier
scatterplot(x=[v_r_qsi], y=[e_r_qsi], color='r', label='Quasi-optimal portfolio')
scatterplot(x=v_r_pos, y=e_r_pos, color='orange', label='Chosen portfolio')
scatterplot(x=v_r_bwd, y=e_r_bwd, marker='x', label='Optimal portfolios with position constraints')
lineplot(x=variance, y=expectation, label='Efficient frontier').set(
         xlabel='Variance', ylabel='Expected return', xlim=[min(variance)*0.98, max(variance)*1.02]);
legend(loc='upper left'); show();

# variance profile of active positions
scatterplot(x=[max(k_bwd)], y=[v_r_qsi], color='r', label='Quasi-optimal portfolio');
scatterplot(x=k_pos, y=v_r_pos, color='orange', label='Chosen portfolio')
scatterplot(x=k_bwd, y=v_r_bwd, marker='x', label='Optimal portfolios with position constraints').set(
            ylabel='Variance', xlabel='Number of positions');

# %% 5. Limit trading costs 
'''
Since we have an existing portfolio h with weights w_h that we wish to rebalance, 
we need to take transaction costs into account. 

TODO / Anmerkung: Für Rebalancing Aspekte ist das natürlich sehr stark.

To do this, we limit the number of trades using the selection routine with 
the forward selection heuristic.

We first define the cardinality function, assuming:
    1) Fixed transaction costs f_cn = 100 for n = 1,..n 
    2) The market impact of trades follows a quadratic market impact model 
       with the scaling matrix defined as the scaled return standard deviations;
    3) The Active Positions are those determined in the previous step.
                                    
TODO / Anmerkung: Wäre das dann unser aktuelles Portfolio, was wir unter Berücksichtung von TC optimieren?                                        

Then we apply the forward selection using the cardinality function to find the 
best portfolio achieved by each number of trades.
'''

################## inputs (you can change them) ##################
fc = 100  # fixed cost per instrument traded
gamma_mi = 1e-9  # quadratic market impact scaling parameter
##################################################################

# transaction cost parameters
fc_tilde = fc/v_h  # fixed costs
q_tilde = sqrt(gamma_mi*v_h)*sqrt(cv_r*eye(n_bar))  # market impact weighting matrix

def g(s_k=None):
    # selection set
    if s_k is None:
        s_k = arange(n_bar - 1)
    elif isscalar(s_k):
        s_k = array([s_k - 1]) 
    else:
        s_k = s_k - 1
    # sorted selection set
    s_k = sort(s_k)
    # number of trades allowed
    k = len(s_k)
    # optimization variable
    x = Variable(n_bar + 1) 

    box_constraints = [
        # target expected return
        e_r@x[1:] <= e_r_pos,
        # long-only constraint
        x[1:] >= 0,
        # 95% weight in stocks
        cp_sum(x[1:n_stocks + 1]) >= 0.95]

    # non-active positions
    no_holdings = setdiff1d(range(n_bar), s_pos - 1) + 1
    # position constraint (keep only selected positions)
    if k_pos < n_bar:
        position_constraint = [x[no_holdings] == zeros(n_bar - k_pos)]
    else:
        position_constraint = []

    # non-traded instruments
    no_trade = setdiff1d(setdiff1d(range(n_bar), s_k) + 1, no_holdings)
    # trade constraint (trade in only selected instruments)
    if len(no_trade) > 0:
        trade_constraint = [x[no_trade] == w_h[no_trade - 1]]
    else:
        trade_constraint = []

    # fixed transaction costs
    fi = fc_tilde*(k + n_bar - k_pos)

    # move quadratic target to constraints
    sig = sqrtm(cv_r)

    soc_constraints = [
        # budget constraint with transaction costs
        SOC(sqrt(0.25*sum(inv(q_tilde**2)) - fi + 1 - sum(w_h)), 
            q_tilde@(x[1:] - w_h) + 0.5*inv(q_tilde)@ones(n_bar)),
        # quadratic target constraint
        SOC(x[0], sig@x[1:])]

    # optimization problem
    prob = Problem(Minimize(x[0]), soc_constraints + box_constraints +\
                   position_constraint + trade_constraint)
    # solve optimization
    prob.solve(solver=CLARABEL)
    
    if prob.status == 'optimal':
        w = array(x.value).squeeze()
        w = w[1:]
        w = atleast_1d(w)
        u = w@cv_r@w
        return w, u
    else:
        return zeros(n_bar), inf

# select trades using forward selection
w_fwd, v_r_fwd, s_k_fwd = forward_selection(g, n_bar)
w_fwd = array(w_fwd)  # portfolio weights
w_fwd = w_fwd[isfinite(v_r_fwd.squeeze())]  # select valid allocations
w_fwd = where(w_fwd > 1e-13, w_fwd, 0)  # set spurious small weights to zero
k_fwd = sum((abs(w_fwd - w_h)> 1e-13), axis=1) - (n_bar - k_pos)  # number of trades
e_r_fwd = w_fwd@e_r  # portfolio expected returns
v_r_fwd = diag(w_fwd@cv_r@w_fwd.T)  # portfolio return variances

'''
TODO: Check ich net
We select the portfolio w_new by trading off between future performance 
and projected transaction costs. 

In particular, we select the portfolio achieved using ktra = 2 trades 
by choosing the portfolio with the highest expected return. 

Note that we also trade an additional 2 instruments to divest from the instruments where we chose not to maintain active positions.
'''

########## input (you can change it) ##########
k_tra = 2  # number of trades
###############################################

# select portfolio
s_tra = s_k_fwd[k_tra]  # active positions
w_new = w_fwd[k_fwd == k_tra]  # portfolio weights
e_r_new = e_r_fwd[k_fwd == k_tra]  # expected portfolio return
v_r_new = v_r_fwd[k_fwd == k_tra]  # portfolio return variance
r_w_new = r@w_new.squeeze()  # ex-ante performance
print('w_new =', w_new.round(4))

##################################################### plots ####################################################
# mean-variance profile of efficient frontier
scatterplot(x=[v_r_qsi], y=[e_r_qsi], color='r', label='Quasi-optimal portfolio')
scatterplot(x=v_r_pos, y=e_r_pos, color='orange', label='Chosen portfolio with position constraints')
scatterplot(x=v_r_fwd, y=e_r_fwd, marker='x', label='Optimal portfolios with transaction costs')
scatterplot(x=v_r_new, y=e_r_new, color='darkgreen', label='Chosen portfolio');
lineplot(x=variance, y=expectation, label='Efficient frontier').set(
         xlabel='Variance', ylim=[min(expectation)*0.999, max(expectation)*1.001]); show();

# portfolio weights
colors = color_palette("Spectral", n_colors=n_bar)
for n in range(n_bar+1):
    if n == 0:
        fill_between(arange(1, k_pos+1), w_fwd[:k_pos, n], zeros(k_pos), color=colors[n], label=instruments[n])
    elif n < n_bar:
        fill_between(arange(1, k_pos+1), sum(w_fwd[:k_pos, :n+1], axis=1), 
                     sum(w_fwd[:k_pos, :n], axis=1), color=colors[n], label=instruments[n])
    else:
        fill_between(arange(1, k_pos+1), ones(k_pos), sum(w_fwd[:k_pos, :n], axis=1), color='darkslategray')
lineplot(x=[k_tra, k_tra], y=[-10, 10], orient='y', color='black').set(
         ylim=[0, 1], xlabel='Number of trades', xticks=arange(1, k_pos+1), xlim=[1, k_pos]); show();

# optimized portfolio ex-ante return distribution
histplot(x=r_w, weights=p_j, bins=50, binrange=[min(r_w), max(r_w)], label='Current portfolio')
histplot(x=r_w_new, weights=p_j, bins=50, binrange=[min(r_w), max(r_w)], label='Rebalanced portfolio').set(
         xlabel='', ylabel=''); gca().legend();

# %% Save data
'''
Save the portfolio weights into database db_final_portfolio_historical.
'''
out = DataFrame({instruments[i]: w_new[0, i] for i in range(w_pos.shape[1])}, index=[0])
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_Historical\Data\Input_Step9/db_final_portfolio_historical.csv', index=False); del out;
