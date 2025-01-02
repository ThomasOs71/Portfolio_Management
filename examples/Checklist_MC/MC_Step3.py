# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:36:20 2024

@author: Thomas


s_checklist_montecarlo_step03
This case study performs Step 3 of the "Monte Carlo Checklist", namely Quest for invariance,
to estimate the joint distribution of the invariants based on copula-marginal implementation.
"""

# %% Packages
from numpy import append, arange, argsort, array, average, busday_count, cov, cumsum, datetime64, diag, empty,\
                  full, isnan, isfinite, log, minimum, nan, ones, r_, sort, spacing, sqrt, squeeze, sum, where, zeros, linalg
from scipy.stats import multivariate_t, t
import statsmodels.multivariate.factor as fa
from bisect import bisect_left, bisect_right
from statsmodels.tsa.ar_model import AutoReg
from pandas import read_csv, Series, DataFrame, MultiIndex, to_datetime
from seaborn import lineplot, scatterplot, histplot
from matplotlib import rcdefaults
from matplotlib.pyplot import imshow, show
# Invariance Testing
from invariance_test_ellipsoid import invariance_test_ellipsoid
from invariance_test_ks import invariance_test_ks
from invariance_test_copula import invariance_test_copula
# Estimation
from fit_trans_matrix_credit import fit_trans_matrix_credit
from fit_locdisp_mlfp import fit_locdisp_mlfp
from fit_locdisp_mlfp_difflength import fit_locdisp_mlfp_difflength
from twist_prob_mom_match import twist_prob_mom_match
from simulate_markov_chain_multiv import simulate_markov_chain_multiv
from project_trans_matrix import project_trans_matrix

# %% Load data

############################################### market ###############################################
###
## Risk Drivers (Montecarlo checklist - Step 2)
# import db_riskdrivers_series database 
db_riskdrivers_series = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_riskdrivers_series.csv', 
                                    index_col=0, parse_dates=True)
x = db_riskdrivers_series.values  # market risk drivers 
t_bar = x.shape[0] - 1  # length of market risk drivers time series
d_bar = x.shape[1]  # number of market risk drivers
risk_drivers_names = db_riskdrivers_series.columns  # names of market risk drivers 

## Risk Driver (Historical Checklist)
# invariants (Historical checklist - Step 3b) 
db_invariants_series = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_invariants_series_historical.csv', 
                                   index_col=0, parse_dates=True) 
eps = db_invariants_series.values  # invariants series
dates = to_datetime(array(db_invariants_series.index))  # market risk drivers dates 
db_invariants = {}
for i in range(db_invariants_series.shape[1]):
    db_invariants[i] = array(db_invariants_series.iloc[:, i])
    
# GARCH(1,1) parameters (Historical checklist - Step 3b)
'''
Modelling from Historical Checklist
'''
db_invariants_garch_param = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_invariants_garch_param.csv', 
                                     index_col=0) 
# fitted parameters of GARCH(1,1) model
a = db_invariants_garch_param.loc['a'].values
b = db_invariants_garch_param.loc['b'].values
c = db_invariants_garch_param.loc['c'].values
mu = db_invariants_garch_param.loc['mu'].values

# next step models for stocks, S&P and options (Historical checklist - Step 3a)
'''
Modellübersicht der Univariate Modells
'''
db_invariants_nextstep = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_invariants_nextstep_historical.csv')
db_nextstep = dict(zip(range(db_invariants_series.shape[1]), db_invariants_nextstep.values.squeeze()))

# flexible probabilities (Historical checklist - Step 3b)
'''
Basiert auf Time and State EP Conditioning on VIX
'''
db_estimation_flexprob = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_estimation_flexprob.csv', 
                                  index_col=0, parse_dates=True) 
p = db_estimation_flexprob.p[dates].values  # flexible probabilities

############################################### credit ###############################################
### Credit Risk Drivers (Montecarlo checklist - Step 2)
db_riskdrivers_tools = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_riskdrivers_tools.csv')
n_bonds = int(db_riskdrivers_tools.n_bonds.dropna().iloc[0]) # number of bonds 
c_bar = int(db_riskdrivers_tools.c_bar.dropna().iloc[0])  # number of ratings classes
t_now = datetime64(db_riskdrivers_tools.t_now[0], 'D')  # current date 
ratings_t_now = array(db_riskdrivers_tools.ratings_t_now.dropna())  # ratings at current time 
ratings = db_riskdrivers_tools.ratings.dropna()  # ratings labels 
ratings.name = '$$\\boldsymbol{p}^{\\mathit{credit}}$$'

# import db_riskdrivers_credit database 
db_riskdrivers_credit = read_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data/db_riskdrivers_credit.csv', 
                                 index_col=0, parse_dates=True)
dates_credit = array(db_riskdrivers_credit.index).astype('datetime64[D]')  # credit risk drivers dates 
idx_ns_bonds = arange(d_bar - n_bonds*4, d_bar)   # index of Nelson-Siegel parameters 
t_bonds = sum(isfinite(x[:, idx_ns_bonds[0]])) - 1  # index of non-NaN entries
x_obligor = x[t_bar - t_bonds:, idx_ns_bonds]  # credit risk drivers 

# cumulative number of transitions
mapper = {}
n_obligors = db_riskdrivers_credit.iloc[:, :c_bar]  # obligors for each rating 
n_cum_trans = db_riskdrivers_credit.iloc[:, c_bar:c_bar**2]  # cumulated number of transitions
for col in n_cum_trans:
    (rating_from, _, rating_to) = col[12:].partition('_')
    mapper[col] = (rating_from, rating_to)
from_to_index = MultiIndex.from_product([ratings, ratings], names=('rating_from', 'rating_to'))
n_cum_trans = n_cum_trans.rename(columns=mapper).reindex(columns=from_to_index).fillna(0)
n_cum_trans = n_cum_trans.values.reshape((-1, c_bar, c_bar), order='C')  # array format


# %% 1. AR(1) next-step functions for Nelson-Siegel parameters (Read more) 
# next step model for Nelson Siegel parameters 
eps_obligor = zeros((t_bonds, n_bonds*4))
b_ar_obligor = zeros(n_bonds*4) 
for i in range(n_bonds*4):
    # fitted AR(1)
    ar_model = AutoReg(x_obligor[:, i], lags=1).fit()
    # AR(1) parameter
    b_ar_obligor[i] = ar_model.params[1]
    # invariants
    eps_obligor[:, i] = x_obligor[1:, i] - b_ar_obligor[i]*x_obligor[:-1, i]

# store invariants, AR(1) parameters and next-step functions
k = 0
db_ar1_param = {} 
for i in idx_ns_bonds:
    db_invariants[i] = r_[full(t_bar - t_bonds, nan), eps_obligor[:, k]]
    db_nextstep[i] = 'AR(1)'
    db_ar1_param[i] = {'b': b_ar_obligor[k]}
    k = k + 1

# %% 2. Tests for invariance (Read more) 
############ inputs (you can change them) ############
i_plot = 1  # invariant to be tested from 1 up to 104
lag = 5  # maximum lag used in invariance tests 
######################################################

# invariant
invar = db_invariants[i_plot - 1][~isnan(db_invariants[i_plot - 1])]

###################### plots ######################
# ellipsoid test
_ = invariance_test_ellipsoid(invar, lag); show();

# Kolmogorov-Smirnov test 
_ = invariance_test_ks(invar); show();

# copula test 
_ = invariance_test_copula(invar, lag); show();


# %% 3. Marginal distributions for stocks, S&P 500 and options invariants (Read more) 
'''
Historical Approach for the RD bzw. Invariantes 1 bis 96
'''
# invariants to be modeled nonparametrically 
index_non_parametric = arange(0, d_bar - n_bonds*4)
df_estimation_non_parametric = {} 
# marginal distributions for stocks, sp500 and options 
for i in index_non_parametric:
    if (db_invariants_nextstep.iloc[0, i] == 'GARCH(1,1)'): 
        # nonparametric estimation for stocks and S&P 500
        df_estimation_non_parametric[i] = twist_prob_mom_match(eps[:, i], 0, 1, p)  # Nutzen EP um die EPS zu vereinfachen
    elif (db_invariants_nextstep.iloc[0, i] == 'Random walk'):
        # nonparametric estimation options
        df_estimation_non_parametric[i] = p


#a = df_estimation_non_parametric[i]
#b = p
#i = 0
'''
Warum machen wir das?
Warum machen wir das beim Random Walk anders?
-> Vermutlich wegen Standardisierung...[ist das ne Annahme im Garch?]
'''

# %% 4. Marginal distributions for corporate bonds's invariants (Read more) 
'''
Marginal Parameteric Distribution (student t) für 97 - 104
-> Robuste Schätzung per Student t
'''
#################### inputs (you can change them) ####################
nu_min = 3  # lower bound for the degrees of freedom for t marginals
nu_max = 100  # upper bound for the degrees of freedom for t marginals
######################################################################

# invariants to be modeled parametrically
index_parametric = arange(d_bar - n_bonds*4, d_bar)
# marginal distributions for bonds
p_bonds = p[-t_bonds:]/sum(p[-t_bonds:])  # rescaled probabilities 
nu_vec = arange(nu_min, nu_max + 1)  # grid of degrees of freedom
db_estimation_parametric = {} 
for i in index_parametric:
    if (DataFrame([db_nextstep]).iloc[0, i] == 'AR(1)'):
        mu_nu = zeros(nu_vec.shape[0])
        sigma2_nu = zeros(nu_vec.shape[0])
        llike_nu = zeros(nu_vec.shape[0]) 
        for j in range(nu_vec.shape[0]):
            nu = nu_vec[j]
            # fit Student t model
            mu_nu[j], sigma2_nu[j] = fit_locdisp_mlfp(eps_obligor[:, i-96], p=p_bonds, nu=nu)
            # log-likelihood of Student t distribution
            llike_nu[j] = sum(p_bonds*(log(sqrt(sigma2_nu[j])) +\
                                          t.logpdf(eps_obligor[:, i-96], nu, mu_nu[j], sqrt(sigma2_nu[j]))))
        # degree of freedom that gives highest log-likelihood
        llike_nu_max = argsort(llike_nu)[-1]
        db_estimation_parametric[i] = {'invariant': i, 'nu': nu_vec[llike_nu_max], 
                                       'mu': mu_nu[llike_nu_max], 'sig2': sigma2_nu[llike_nu_max]}

# %% 5. Parameters of Student copula with static estimation (Read more) 
'''
Step 1: Generation of 1) Invariant Grads and 2) Marginal CDF 
    -> Using Empirical CDF for Historical Invariants
    -> Using Student t CDF(nu_i) for Parameteric Invariants

Step 2: Joint Estimation of Copula
    -> Grades eingebaut in Standard Quantile (mit loc = 0 und disp = 1)
    -> Estimation of Copula by Standard Max.Likelihood Student t optimization
    -> Factor-Ansatz für Shrinkage der Correlation
    -> Prüfung auf Positive Semi-Definiteness der Korrelation
    -> Prüfung der DoF (nu) über Log-Likelihood
'''

###################### inputs (you can change them) ######################
nu_min_copula = 4  # lower bound for the degrees of freedom for t copula
nu_max_copula = 70  # upper bound for the degrees of freedom for t copula
k_bar = 7  # number of factors for factor analysis
##########################################################################

# invariants grades
u = zeros((t_bar, d_bar))
eps_t = zeros((t_bonds, d_bar))
for i in range(d_bar): 
    # nonparametric estimation: stocks, S&P 500 and options
    if i in index_non_parametric:
        # copula-marginal separation process             
        x_grid, ind_sort = sort(eps[:, i], axis=0), argsort(eps[:, i], axis=0)  # sorted scenarios
        eps_i = eps[:, i]  # scenarios
        p_i = df_estimation_non_parametric[i]  # probabilities
        eps_i_sort = Series(eps_i).iloc[argsort(eps_i)]  # sorted scenarios 
        p_i_sort = p_i[argsort(eps_i)]  # sorted probabilities
        u_sort = append(0, cumsum(p_i_sort))  # cumulative sums of sorted probabilities
        cindx = [bisect_right(eps_i_sort, x_grid[k]) for k in range(x_grid.shape[0])]
        cdf_x = squeeze(u_sort[cindx])  # marginal cdf
        u[:, i][ind_sort] = cdf_x  # copula scenarios
        # clear spurious outputs
        u[:, i][u[:, i] >= 1] = 1 - spacing(1)
        u[:, i][u[:, i] <= 0] = spacing(1)  
    # parametric estimation (Student t): bonds
    elif i in index_parametric:
        eps_t[:, i] = (eps_obligor[:, i-96] - db_estimation_parametric[i]['mu'])/sqrt(db_estimation_parametric[i]['sig2'])
        u[-t_bonds:, i] = t.cdf(eps_t[:, i], db_estimation_parametric[i]['nu'])
        # values must be < 1 for Student t
        u[-t_bonds:, i] = minimum(u[-t_bonds:, i], 0.99999999)
        u[:-t_bonds, i] = nan

# standardized invariants and maximum likelihood
grid_nu = arange(nu_min_copula, nu_max_copula + 1)  # grid of degrees of freedom 
eps_tilde = zeros((t_bar, d_bar, len(grid_nu)))
rho2_copula_vec = zeros((d_bar, d_bar, len(grid_nu)))
llike_nu = zeros(len(grid_nu))

# joint estimation of t copula 
'''

'''
for l in range(len(grid_nu)):  # Grid über NU des Student Copula
    # standardized invariants    
    for i in range(d_bar):    
        eps_tilde[:, i, l] = t.ppf(u[:, i], grid_nu[l])  #  Änderung auf T-Verteilung (gucken wo das steht)
    # copula parameters with maximum likelihood
    _, sigma2 = fit_locdisp_mlfp_difflength(eps_tilde[:, :, l],  
                                            p = p,  # TODO: Wahrscheinlichkeitsthema 
                                            nu=grid_nu[l], 
                                            threshold=10**-3, 
                                            maxiter=1000)
    
    # Dimension Reduction via Factor Analysis Shrinkage ~ Vereinfachung der Korrelationsstruktur
    s_vol = sqrt(diag(sigma2))  # target volatility vector
    cr_x = diag(1/s_vol)@sigma2@diag(1/s_vol)  # target correlation
    paf = fa.Factor(n_factor=k_bar, corr=cr_x, method='pa', smc=True).fit()  # fitted model
    beta = paf.loadings  # loadings
    delta2 = paf.uniqueness  # variances
    sigma2_fa = beta@beta.T + diag(delta2)
    
    # (sigma2_fa.T == sigma2_fa).all()
    
    # correlation matrix
    rho2_copula_vec[:, :, l] = diag(1/sqrt(diag(sigma2_fa)))@sigma2_fa@diag(1/sqrt(diag(sigma2_fa)))
    
    # (rho2_copula_vec[:, :, l].T == rho2_copula_vec[:, :, l]).all()
    # Problem mit der semi-definiteness?
    # import numpy as np
    # a,aa = np.linalg.eig(rho2_copula_vec[:, :, l])    
    # -> Am besten den negativen Eigenvalue dann auf 0 setzen
    # a[a<0] = 0
    # rho2_neu = aa@np.diag(a)@aa.T
    # rho2_alt = rho2_copula_vec[:, :, l]
    # np.linalg.norm(rho2_alt - rho2_neu)
    # np.linalg.cholesky(rho2_neu)
    
    # Test auf Positive Definiteness:
    eigenvalues, eigenvectors = linalg.eig(rho2_copula_vec[:, :, l])
    if (eigenvalues < 0).any():
        eigenvalues[eigenvalues < 0] = 0
        rho2_copula_vec[:, :, l] = eigenvectors@diag(eigenvalues)@eigenvectors.T   
    
    # non missing invariants 
    non_missing_eps_tilde = eps_tilde[~isnan(u).any(axis=1), :, l]
    
    
    # log-likelihood with no missing invariants 
    llike_nu[l] = sum(p[-t_bonds:]*(multivariate_t.logpdf(non_missing_eps_tilde, zeros(d_bar), rho2_copula_vec[:, :, l], grid_nu[l]) -\
                                    sum(t.logpdf(non_missing_eps_tilde, grid_nu[l]), axis=1)))
'''
Test auf Positive Definiteness kommt von mir:
    Sonst häufig Fehler in der multivariate_t.logpdf, da diese nur mit pos-semidefinitiven Covaraincen klappt
'''
# degree of freedom that gives highest log-likelihood
l_nu = argsort(llike_nu)[-1]
nu_copula = int(grid_nu[l_nu])
rho2_copula = rho2_copula_vec[:, :, l_nu]      

################ plot ################
# estimated copula correlation matrix
rcdefaults()
imshow(rho2_copula, vmin=-1, vmax=1);


# %% 6. Invariants scenarios from copula-marginal distribution 
'''
-> copula-marginal combination algorithm

Wir benötigen:
    1) Den Copula
    2) Die Marginalen Verteilungen
    
    Wir ziehen aus dem Copula Zufallsvektoren 
    -> sezten in die Marginalen Quantile-Funktionen ein.
    -> Erhalten darüber Werte für jeden Invariant.
    
    Korrelationstruktur ist berücksichtigt, weil die im Copula ist.
    


'''
############ inputs (you can change them) ############
t_hor = datetime64('2012-10-26')  # investment horizon
j_bar = 500  # number of forecast scenarios
######################################################

m_bar = busday_count(t_now, t_hor)  # number of daily monitoring times 
p_marginal = array(list(df_estimation_non_parametric.values())).T  # marginal probabilities

# invariants modeled parametrically as Student t distributed
idx_parametric = arange(d_bar - n_bonds*4, d_bar)
# invariants modeled non parametrically
idx_nonparametric = list(set(range(d_bar)) - set(idx_parametric))

# invariants forecast at future time steps
eps_fcst = zeros((j_bar, m_bar, j_bar))
for m in range(m_bar):
    # standardized invariants scenarios
    eps_tilde_fcst = multivariate_t.rvs(zeros(d_bar), rho2_copula, nu_copula, j_bar)
    '''
    Ziehung aus Verteilung, die dem Copula ähnelt (ist aber kein Copula)
    '''
    # invariants modeled nonparametrically: stocks, SP500, implied volatility
    for i in idx_nonparametric:  
        # grades
        u_fcst = t.cdf(eps_tilde_fcst[:, i], nu_copula)
        '''
        Standardizieren über einsetzen in Standard student t für Grades
        '''
        # scenarios
        eps_i = eps[:, i]  
        # probabilities
        p_marginal_i = p_marginal[:, i]  
        # sorted scenarios 
        eps_i_sort = Series(eps_i).iloc[argsort(eps_i)]  
        # sorted probabilities
        p_i_sort = p_marginal_i[argsort(eps_i)] 
        # cumulative sums of sorted probabilities
        u_sort = append(0, cumsum(p_i_sort))  
        qindx = [max(0, min(bisect_left(u_sort, u_fcst[j]) - 1, t_bar - 1)) for j in range(j_bar)]  # ?
        eps_fcst[:, m, i] = squeeze(eps_i_sort[qindx])
        '''                                                                 
        Effektiv ziehen wir aus einer Copula-Marginal Distribution -> Darüber erhalten wir dann wohl p 
        Über p und eps können wir dann die empirische PPF bilden bzw. da einsetzen und erhalten den Invariant.
        Gebenen einer Wsk aus u, könnnen wir dann einen eps_fcst ziehen
        '''
        
    # invariants modeled parametrically: bonds 
    '''
    Copula Ziehung wird
    '''
    for i in idx_parametric:  
        # grades 
        u_fcst = t.cdf(eps_tilde_fcst[:, i], nu_copula) 
        # location 
        mu_marg = db_estimation_parametric[i]['mu']  
        # dispersion
        sigma2_marg = db_estimation_parametric[i]['sig2']
        # degrees of freedom 
        nu_marg = db_estimation_parametric[i]['nu']
        # invariants
        eps_fcst[:, m, i] = mu_marg + sqrt(sigma2_marg)*t.ppf(u_fcst, nu_marg)  
        '''
        Randomness aus der Copulaziehung -> Einfügen in die Quantile-Funktion für Elliptical Distributions
        '''

# scenario probabilities forecast
p_scenario = ones(j_bar)/j_bar  # flat probabilities

# %% 7. Risk drivers forecast via Monte Carlo simulations (Read more) 
'''
Next Step Functions für die Erstellung der Pfade
'''
# increments and dispersion parameter for GARCH(1,1)
'''
Starting Parameter für Garch Model
'''
d_garch = [d for d in range(d_bar) if DataFrame([db_nextstep]).iloc[0, d] == 'GARCH(1,1)'] 
dx_fcst = empty((j_bar, m_bar + 1, d_bar))  # J_bar Simulations, m_bar Forcasts, d_bar Risk Drivers
sigma2_garch = empty((j_bar, m_bar + 1, d_bar))
for d in d_garch:
    # dispersion
    sigma2_garch[:, 0, d] = db_invariants_garch_param.iloc[-1, d]  # Garch-Parameter der letzten Periods (Start-Werte für Garch)
    # increments
    dx_fcst[:, 0, d] = x[-1, d] - x[-2, d]  # Delta_x Werte der letzten Periods (Start-Werte für Garch)



# Risk Drivers Forecast at future time steps
db_ar1_par = DataFrame({risk_drivers_names[i]: db_ar1_param[i] for i in idx_ns_bonds})
x_fcst = empty((j_bar, m_bar + 1, d_bar))
x_fcst[:, 0, :] = x[-1, :]  # risk drivers at current time
for m in range(1, m_bar + 1):
    for d in range(d_bar):
        risk_driver = risk_drivers_names[d]
        # risk drivers modeled as random walk: implied volatility
        if DataFrame([db_nextstep]).iloc[0, d] == 'Random walk':
            # risk drivers forecast
            x_fcst[:, m, d] = x_fcst[:, m - 1, d] + eps_fcst[:, m - 1, d]  # Update des vorherigen Wertes um neuen Invariant
            '''
            x_{t+1} = x_{t} + eps_{t+1} = x_{t} + Delta_x_{t+1}, weil Delta_x_{t+1} = eps_{t+1}
            '''
        # risk drivers modeled as GARCH(1,1): stocks, S&P500
        elif DataFrame([db_nextstep]).iloc[0, d] == 'GARCH(1,1)':
            # GARCH(1,1) volatility
            sigma2_garch[:, m, d] = c[d] + b[d]*sigma2_garch[:, m - 1, d] + a[d]*(dx_fcst[:, m - 1, d] - mu[d])**2  # Garch basiert auf vorherige Peridoe
            # risk drivers increments
            dx_fcst[:, m, d] = mu[d] + sqrt(sigma2_garch[:, m, d])*eps_fcst[:, m - 1, d]  # Update der Vola für aktuelle Periode 
            # risk drivers forecast
            x_fcst[:, m, d] = x_fcst[:, m - 1, d] + dx_fcst[:, m, d]
            '''
            sigma2_{t+1} = c + b * sigma2 + a * ( delta_x_{t} - mu)**2 (Garch 1,1)
            -> Einsetzen in:
            dx_fcst -> delta_x_{t+1} = mu + sigma * eps_{t+1}
            -> Einsetzen in:             
            x_{t+1} = x_{t} + delta_x_{t+1}
            '''
             
        # risk drivers modeled as AR(1): bonds 
        elif DataFrame([db_nextstep]).iloc[0, d] == 'AR(1)':
            # AR(1) parameter
            b_ar1 = db_ar1_par.loc['b', risk_driver]
            # risk drivers forecast 
            x_fcst[:, m, d] = b_ar1*x_fcst[:, m - 1, d] + eps_fcst[:, m - 1, d]
            '''
            x_{t+1} = b*x_{t} + eps(t+1)
            '''
            
################## inputs (you can change them) ##################
d_plot = 1  # index of risk drivers forecast to plot from 1 to 104
n_paths = 25  # number of paths to plot
##################################################################
               
############################################ plots ############################################
# forecast moments at each time step
mu_t_hor = zeros(m_bar + 1)
mu_t_hor_2 = zeros(m_bar + 1)
sigma_t_hor = zeros(m_bar + 1)
for m in range(0, m_bar + 1):
    # location
    mu_t_hor[m] = average(x_fcst[:, m, d_plot - 1])  
    # standard deviation
    sigma_t_hor[m] = sqrt(cov(x_fcst[:, m, d_plot - 1].reshape(-1, 1).T))  

# risk driver forecast paths and moments to horizon
grid_hist = arange(t_bar - 2, t_bar + 1)  # grid for known values
lineplot(x=grid_hist, y=db_riskdrivers_series.iloc[-3:, d_plot - 1])
scatterplot(x=grid_hist, y=db_riskdrivers_series.iloc[-3:, d_plot - 1])
grid_fcst = arange(t_bar, t_bar + m_bar + 1)  # grid
[lineplot(x=grid_fcst, y=x_fcst[j, :, d_plot - 1], c='gray') for j in range(min(j_bar, n_paths))]
lineplot(x=grid_fcst, y=mu_t_hor, label='Location')
lineplot(x=grid_fcst, y=mu_t_hor + 2*sigma_t_hor, label='+ 2 st. dev.')
lineplot(x=grid_fcst, y=mu_t_hor - 2*sigma_t_hor, label='- 2 st. dev.'); show();

# risk driver forecast distribution
histplot(x_fcst[:, -1, d_plot - 1], stat='probability');


# %% 8. Annual credit transition matrix (Read more) 
########### input (you can change it) ###########
tau_hl_credit = 5  # half-life parameter in years
#################################################

# annual credit transition matrix
p_credit = fit_trans_matrix_credit(dates_credit, n_obligors.values, n_cum_trans, tau_hl_credit)
print('p_credit \n', p_credit.round(3))

# %% 9. Parameters of credit structural invariant distribution (Read more)
index_credit = array([where(db_invariants_series.columns == 'stock GE_log_value')[0][0],
                         where(db_invariants_series.columns == 'stock JPM_log_value')[0][0]])

rho2_credit = rho2_copula_vec[:, index_credit, l_nu][index_credit, :]  # correlation
print('rho2_credit =', rho2_credit[0][1].round(2))
nu_credit = int(grid_nu[l_nu])  # degrees of freedom
print('nu_credit =', nu_credit)

'''
Ist die Copula-Sache für die Credit-Elemente völlig losgelöst von der Market-Berechnung?
'''

# %% 10. Credit ratings forecast via Monte Carlo simulations (Read more) 
# daily credit transition matrix
p_credit_daily = project_trans_matrix(p_credit, 1/252, credit=True)
# ratings forecast at future time steps
ratings_fcst = simulate_markov_chain_multiv(ratings_t_now, p_credit_daily, m_bar,
                                            rho2=rho2_credit, nu=nu_credit, j_bar=j_bar)
    
# paths forecast with ratings transitions
ind_j_plot_GE = zeros(1)  # GE 
ind_j_plot_JPM = zeros(1)  # JPM 
for k in range(1, n_paths):
    for j in range(j_bar):
        if (j not in ind_j_plot_GE and ratings_fcst[j, -1, 0] != ratings_fcst[k, -1, 0]):
            ind_j_plot_GE = append(ind_j_plot_GE, j)
        if (j not in ind_j_plot_JPM and ratings_fcst[j, -1, 1] != ratings_fcst[k, -1, 1]):
            ind_j_plot_JPM = append(ind_j_plot_JPM, j)

############################################## plots ##############################################
# credit ratings forecast paths for GE coupon bond
for j in ind_j_plot_GE:
    plot = lineplot(x=arange(m_bar + 1), y=ratings_fcst[int(j), :, 0])
plot.invert_yaxis()
plot.set_yticks(ticks=arange(10), labels=['','AAA','AA','A','BBB','BB','B','CCC','D','']); show();

# credit ratings forecast paths for JPM coupon bond
for j in ind_j_plot_JPM:
    plot = lineplot(x=arange(m_bar + 1), y=ratings_fcst[int(j), :, 1])
plot.invert_yaxis()
plot.set_yticks(ticks=arange(10), labels=['','AAA','AA','A','BBB','BB','B','CCC','D','']);

# %% Save data
# Scenarios for forecasted risk drivers
out = DataFrame({risk_drivers_names[d]: x_fcst[:, :, d].reshape((j_bar*(m_bar + 1),)) for d in range(d_bar)})
out = out[list(risk_drivers_names[:d_bar].values)]
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_projection_riskdrivers.csv', index=None); del out;

# Scenarios for forecasted credit ratings
out = DataFrame({'GE': ratings_fcst[:, :, 0].reshape((j_bar*(m_bar + 1),)),
                 'JPM': ratings_fcst[:, :, 1].reshape((j_bar*(m_bar + 1),))})
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\db_projection_ratings.csv', index=None); del out;

# number of scenarios and future investment horizon
out = DataFrame({'j_bar': Series(j_bar), 't_hor': Series(t_hor)})
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\/db_projection_tools.csv', index=None); del out;

# Probabilities associated with risk drivers scenarios 
out = DataFrame({'p_scenario': Series(p_scenario)})
out.to_csv(r'C:\Projects\Portfolio_Management\examples\Checklist_MC\data\/db_scenario_probs.csv', index=None); del out;
