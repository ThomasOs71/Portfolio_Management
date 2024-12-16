# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:52:58 2024

@author: Thomas
"""

import numpy as np
import pandas as pd
from bisect import bisect_right
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from schweizer_wolff import schweizer_wolff
from scipy.stats import ks_2samp
import math
import seaborn as sns
from scipy import stats
import matplotlib.patches as patches


def invariance_test_ellipsoid(eps, l_bar, *, conf_lev=0.95, fit=0, r=2, title='Invariance test',
                              bl=None, bu=None, plot_test=True):
    """This function performs an invariance test on a given set of data by computing sample 
    autocorrelations, confidence intervals, and plotting an ellipsoid in a scatter plot, 
    with options for adjusting the plot and fitting distributions.

    Parameters
    ----------
        eps : array, shape(t_bar)
        l_bar : scalar
        conf_lev : scalar, optional
        fit : scalar, optional
        r : scalar, optional
        title : string, optional
        bl : scalar, optional
        bu : scalar, optional
        plot_test : boolean, optional

    Returns
    -------
        rho : array, shape(l_bar)
        conf_int : array, shape(2)
    """

    if len(eps.shape) == 2:
        eps = eps.reshape(-1)

    if bl is None:
        bl = np.percentile(eps, 0.25)
    if bu is None:
        bu = np.percentile(eps, 99.75)

    # settings
    np.seterr(invalid='ignore')
    sns.set_style('white')
    nb = int(np.round(10*np.log(eps.shape)))  # number of bins for histograms

    # compute the sample autocorrelations
    rho = np.array([stats.pearsonr(eps[:-k]-np.mean(eps[:-k], axis=0),
                                eps[k:]-np.mean(eps[k:], axis=0))[0]
                    for k in range(1, l_bar+1)])

    # compute confidence interval
    alpha = 1-conf_lev
    z_alpha_half = stats.norm.ppf(1-alpha/2) / np.sqrt(eps.shape[0])
    conf_int = np.array([-z_alpha_half, z_alpha_half])

    # plot the ellipse, if requested
    if plot_test:
        # Ellipsoid test: location-dispersion parameters
        x = eps[:-l_bar]
        eps = eps[l_bar:]
        z = np.concatenate((x.reshape((-1, 1)), eps.reshape((-1, 1))), axis=1)

        # compute the sample mean and sample covariance and generate figure
        mu_hat = np.mean(z, axis=0)
        sigma2_hat = np.cov(z.T)

        f = plt.figure()
        f.set_size_inches(16, 9)
        gs = plt.GridSpec(9, 16, hspace=1.2, wspace=1.2)

        # max and min value of the first reference axis settings,
        # for the scatter and histogram plots

        # scatter plot (with ellipsoid)
        xx = x.copy()
        yy = eps.copy()
        xx[x < bl] = np.nan
        xx[x > bu] = np.nan
        yy[eps < bl] = np.nan
        yy[eps > bu] = np.nan
        ax_scatter = f.add_subplot(gs[1:6, 4:9])
        ax_scatter.scatter(xx, yy, marker='.', s=10)
        ax_scatter.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        plt.xlabel('obs', fontsize=17)
        plt.ylabel('lagged obs.', fontsize=17)
        
        lam, e = np.linalg.eig(sigma2_hat)  
        center = mu_hat; width = 2*r*np.sqrt(lam[0]); height = 2*r*np.sqrt(lam[1]); angle = np.degrees(math.atan(e[1, 0]/e[1, 1]))
        ellipse = patches.Ellipse(center, width, height, angle=angle, fill=False, color='#f56502', lw=2)
        plt.gca().add_patch(ellipse)
        
        plt.suptitle(title, fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax_scatter.set_xlim(np.array([bl, bu]))
        ax_scatter.set_ylim(np.array([bl, bu]))
        
        ax = f.add_subplot(gs[7:, 4:9])

        # histogram plot of observations
        xxx = x[~np.isnan(xx)]
        px = np.ones(xxx.shape[0]) / xxx.shape[0]
        hist_kws = {'weights': px.flatten(), 'edgecolor': 'k'}
        fit_kws = {'color': 'orange', 'cut': 0}
        if fit == 1:  
            # normal
            sns.histplot(x=xxx, weights=px.flatten(), edgecolor='k', alpha = 0.5, bins=nb, kde=False, stat='density', 
                         common_norm=False, ax=ax)
            plt.legend(['Normal fit', 'Marginal distr'], fontsize=14)
            
        elif fit == 2 and sum(x < 0) == 0:  
            # exponential
            sns.histplot(x=xxx, weights=px.flatten(), edgecolor='k', alpha = 0.5, bins=nb, kde=False, stat='density', 
                         ax=ax, common_norm=False)

            # calculate the pdf
            x_0, x_1 = ax.get_xlim()  # extract the endpoints for the x-axis
            x_pdf = np.linspace(x_0, x_1, 100)
            y_pdf = stats.expon.pdf(x_pdf, *stats.expon.fit(xxx))
            
            ax.plot(x_pdf, y_pdf, 'orange')                                                   

            plt.legend(['Exponential fit', 'Marginal distr'], fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.ylabel('')
        elif fit == 3 and sum(x < 0) == 0:  # Poisson
            ax.hist(xxx, bins=nb, weights=px, density=True, facecolor=[0.8, 0.8, 0.8], edgecolor='k')
            k = np.arange(x.max() + 1)
            mlest = x.mean()
            plt.plot(k, stats.poisson.pmf(k, mlest), 'o', linestyle='-', lw=1, markersize=3, color='orange')
            plt.legend(['Poisson fit', 'Marginal distr.'], loc=1, fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        else:
            ax.hist(xxx, bins=nb, weights=px, density=True, facecolor=[0.8, 0.8, 0.8], edgecolor='k')
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            
        ax.get_xaxis().set_visible(False)
        ax.set_xlim(np.array([bl, bu]))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        
        ax = f.add_subplot(gs[1:6, 0:3])
        # histogram plot of lagged observations
        yyy = eps[~np.isnan(yy)]
        py = np.ones(yyy.shape[0]) / yyy.shape[0]
        hist_kws = {'weights': py.flatten(), 'edgecolor': 'k'}
        fit_kws = {'color': 'orange', 'cut': 0}
        if fit == 1:
            sns.histplot(y=yyy, weights=py.flatten(), edgecolor='k', alpha = 0.5, bins=nb, kde=False, stat='density', ax=ax)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            
        elif fit == 2 and sum(eps < 0) == 0:
            sns.histplot(y=yyy, weights=py.flatten(), edgecolor='k', alpha = 0.5, bins=nb, stat='density', kde=False, ax=ax)

            # calculate the pdf
            y0, y1 = ax.get_ylim()  # extract the endpoints for the x-axis
            y_pdf = np.linspace(y0, y1, 100)
            x_pdf = stats.expon.pdf(y_pdf, *stats.expon.fit(yyy))
            ax.plot(x_pdf, y_pdf, 'orange') 
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel('')
            
        elif fit == 3 and sum(eps < 0) == 0:
            ax.hist(yyy, bins=nb, weights=py, density=True, facecolor=[0.8, 0.8, 0.8], edgecolor='k', orientation='horizontal')

            mlest = eps.mean()
            k = np.arange(eps.max() + 1)
            plt.plot(stats.poisson.pmf(k, mlest), k, 'o', linestyle='-', lw=1, markersize=3, color='orange')
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        else:
            ax.hist(yyy, bins=nb, weights=py, density=True, facecolor=[0.8, 0.8, 0.8], edgecolor='k', orientation='horizontal')
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(np.array([bl, bu]))
        ax.invert_xaxis()
        
        # autocorrelation plot
        ax = f.add_subplot(gs[1:6, 10:])
        xx = np.arange(1, l_bar + 1)
        xxticks = xx
        if len(xx) > 15:
            xxticks = np.linspace(1, l_bar + 1, 10, dtype=int)
        plt.bar(xx, rho[:l_bar], 0.5, facecolor=[.8, .8, .8], edgecolor='k')
        plt.bar(xx[l_bar - 1], rho[l_bar - 1], 0.5, facecolor='orange', edgecolor='k')  # highlighting the last bar
        plt.plot([0, xx[-1] + 1], [conf_int[0], conf_int[0]], ':k')
        plt.plot([0, xx[-1] + 1], [-conf_int[0], -conf_int[0]], ':k')
        plt.xlabel('lag', fontsize=17)
        plt.ylabel('Autocorrelation', fontsize=17)
        plt.axis([0.5, l_bar + 0.5, -1, 1])
        plt.xticks(xxticks)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    return rho, conf_int





def invariance_test_ks(eps, *, conf_lev=0.95, title='Kolmogorov-Smirnov test', plot_test=True):
    """This function conducts a Kolmogorov-Smirnov test on a given dataset by computing 
    the sample autocorrelations, confidence intervals, and plotting the results in a histogram, 
    with options for adjusting the plot and fitting distributions.

    Parameters
    ----------
        eps : array, shape (t_bar, )
        conf_lev : scalar, optional
        title : string, optional
        plot_test : boolean, optional

    Returns
    -------
        z_ks : float
            KS statistic
        z : float
            Confidence interval
    """

    # generate two random mutually exclusive partitions of observations
    t_bar = eps.shape[0]
    half_t_bar = t_bar // 2
    # if t_bar is odd, remove one element at random to ensure dimension match on partitions
    if t_bar % 2 != 0:
        eps = np.delete(eps, np.random.randint(t_bar))
    # random permutation
    eps_perm = np.random.permutation(eps)
    # create two partitions
    eps_a = eps_perm[:half_t_bar]
    eps_b = eps_perm[half_t_bar:]
    # compute hfp cdfs of two partitions
    cdf_a = np.arange(len(eps_a))/len(eps_a)
    cdf_b = np.arange(len(eps_b))/len(eps_b)
    # compute KS statistic
    z_ks, _ = ks_2samp(eps_a, eps_b)
    # compute confidence interval
    alpha = 1 - conf_lev
    z = np.sqrt(-np.log(alpha/2)*(len(eps_a) + len(eps_b))/(2*len(eps_a)*len(eps_b)))

    # generate figure
    if plot_test:
        # build band for Kolmogorov-Smirnov test
        band_mid = 0.5*(cdf_a + cdf_b)
        band_up = band_mid + 0.5*z 
        band_low = band_mid - 0.5*z

        # colors
        blue = [0.2, 0.2, 0.7]
        l_blue = [0.2, 0.6, 0.8]
        orange = [.9, 0.6, 0]
        d_orange = [0.9, 0.3, 0]

        # max and min value of first reference axis settings
        xlim_1 = np.percentile(eps, 1.5)
        xlim_2 = np.percentile(eps, 98.5)

        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        
        # plot histogram of Sample 1
        nx1, _, _ = ax1.hist(eps_a, bins=int(round(10 * np.log(len(eps_a.flatten())))), density=True, facecolor=orange, edgecolor='k')
        ax1.set_xlabel('Sample 1')
        ax1.set_xlim((xlim_1, xlim_2))
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax1.grid(False)

        # plot histogram of Sample 2
        nx2, _, _ = ax2.hist(eps_b, bins=int(round(10*np.log(len(eps_a.flatten())))), density=True, facecolor=l_blue, edgecolor='k')
        ax2.set_xlabel('Sample 2')
        ax2.set_xlim((xlim_1, xlim_2))
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax2.grid(False)
        
        ylim = max(np.max(nx1), np.max(nx2))
        ax1.set_ylim([0, ylim])
        ax2.set_ylim([0, ylim])

        # plot CDFs
        sorted_eps_a = np.sort(eps_a)
        sorted_eps_b = np.sort(eps_b)
        
        ax3.scatter(sorted_eps_a, cdf_a, color=d_orange, s=2)
        ax3.scatter(sorted_eps_b, cdf_b, color=blue, s=2)
        
        ax3.scatter(eps_a, 0.002*np.ones(len(eps_a)), color=d_orange, s=0.5)
        ax3.scatter(eps_b, 0.002*np.ones(len(eps_b)), color=blue, s=0.5)
        
        ax3.plot(sorted_eps_a, band_up, '-', color='k', lw=0.5)
        ax3.plot(sorted_eps_b, band_low, '-', color='k', lw=0.5)
        
        ax3.set_xlabel('Data')
        ax3.set_ylabel('Cdf')
        ax3.set_xlim([xlim_1, xlim_2])
        ax3.set_ylim([-0.05, 1.05])
        ax3.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))

        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return z_ks, z
















def invariance_test_copula(eps, lag_bar, k_bar=None):
    """
       This function assesses copula invariance by conducting a Schweizer-Wolff dependence test and plots the results,
       including a 3D histogram of bivariate data and a bar plot illustrating dependence over various lags.

    Parameters
    ----------
        eps : array, shape (t_bar,)
        lag_bar : scalar
        k_bar: int

    Returns
    -------
        sw: array, shape(lag_bar,)

    """

    t_bar = eps.shape[0]

    # Schweizer-Wolff dependence for lags
    sw = np.zeros(lag_bar)
    for l in range(lag_bar):
        sw[l] = schweizer_wolff(np.column_stack((eps[(l + 1):], eps[: -(l + 1)])))

    # grades scenarios
    x_lag = eps[:-lag_bar]
    y_lag = eps[lag_bar:]
    p = np.ones(np.column_stack((x_lag, y_lag)).shape[0])/np.column_stack((x_lag, y_lag)).shape[0]  # equal probabilities
    x_grid, ind_sort = np.sort(np.column_stack((x_lag, y_lag)), axis=0), np.argsort(np.column_stack((x_lag, y_lag)), axis=0)  # sorted scenarios
    # marginal cdf's
    cdf_x = np.zeros(np.column_stack((x_lag, y_lag)).shape)
    for n in range(np.column_stack((x_lag, y_lag)).shape[1]):
        x_bar = x_grid[:, n]
        x_bar = np.atleast_1d(x_bar)
        x = np.column_stack((x_lag, y_lag))[:, n]

        # sorted scenarios-probabilities
        sort_x = np.argsort(x)
        x_sort = pd.Series(x).iloc[sort_x]
        p_sort = p[sort_x]

        # cumulative sums of sorted probabilities
        u_sort = np.zeros(x.shape[0] + 1)
        for j in range(1, x.shape[0] + 1):
            u_sort[j] = np.sum(p_sort[:j])

        # output cdf
        cindx = [0]*x_bar.shape[0]
        for k in range(x_bar.shape[0]):
            cindx[k] = bisect_right(x_sort, x_bar[k])
        cdf_x[:, n] = u_sort[cindx]
        
    # copula scenarios
    u = np.zeros(np.column_stack((x_lag, y_lag)).shape)
    for n in range(np.column_stack((x_lag, y_lag)).shape[1]):
        u[ind_sort[:, n], n] = cdf_x[:, n]
    u[u >= 1] = 1 - np.spacing(1)
    u[u <= 0] = np.spacing(1)  # clear spurious outputs
    
    # normalized histogram
    if k_bar is None:
        k_bar = np.floor(np.sqrt(7*np.log(t_bar)))
    k_bar = int(k_bar)
    
    p = np.ones(u.shape[0])/u.shape[0]  # uniform probabilities
    min_x_1 = np.min(u[:, 0])
    min_x_2 = np.min(u[:, 1])

    # bin width
    h_1 = (np.max(u[:, 0]) - min_x_1)/k_bar
    h_2 = (np.max(u[:, 1]) - min_x_2)/k_bar

    # bin centroids
    xi_1 = np.zeros(k_bar)
    xi_2 = np.zeros(k_bar)
    for k in range(k_bar):
        xi_1[k] = min_x_1 + (k + 1 - 0.5)*h_1
        xi_2[k] = min_x_2 + (k + 1 - 0.5)*h_2

    # normalized histogram heights
    f = np.zeros((k_bar, k_bar))
    for k_1 in range(k_bar):
            for k_2 in range(k_bar):
                # take edge cases into account
                if k_1 > 0 and k_2 > 0:
                    ind = ((u[:, 0] > xi_1[k_1] - h_1/2)&(u[:, 0] <= xi_1[k_1] + h_1/2) &
                           (u[:, 1] > xi_2[k_2] - h_2/2)&(u[:, 1] <= xi_2[k_2] + h_2/2))
                elif k_1 > 0 and k_2 == 0:
                    ind = ((u[:, 0] > xi_1[k_1] - h_1/2)&(u[:, 0] <= xi_1[k_1] + h_1/2) &
                           (u[:, 1] >= min_x_2)&(u[:, 1] <= xi_2[k_2] + h_2/2))
                elif k_1 == 0 and k_2 > 0:
                    ind = ((u[:, 0] >= min_x_1)&(u[:, 0] <= xi_1[k_1] + h_1/2) &
                           (u[:, 1] > xi_2[k_2] - h_2/2) & (u[:, 1] <= xi_2[k_2] + h_2/2))
                else:
                    ind = ((u[:, 0] >= min_x_1)&(u[:, 0] <= xi_1[k_1] + h_1/2) &
                           (u[:, 1] >= min_x_2)&(u[:, 1] <= xi_2[k_2] + h_2/2))

                f[k_1, k_2] = np.sum(p[ind])/(h_1*h_2)

    ############################################################## plots ##############################################################
    # 2D histogram
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xpos, ypos = np.meshgrid(xi_1 - (xi_1[1] - xi_1[0])/2, xi_2 - (xi_2[1] - xi_2[0])/2)  # adjust bin centers to left edges
    ax.bar3d(xpos.flatten('F'), ypos.flatten('F'), np.zeros_like(xpos.flatten('F')), xi_1[1] - xi_1[0], xi_2[1] - xi_2[0], f.flatten())
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f')); ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f')); ax.invert_xaxis()
    plt.xlabel('Grade obs.'); plt.ylabel('Grade lagged obs.')
    
    # dependence plot
    fig = plt.figure()
    plt.bar(range(1, lag_bar + 1), sw, 0.5, facecolor='#969696', edgecolor='#212529')
    plt.bar(range(1, lag_bar + 1)[lag_bar - 1], sw[lag_bar - 1], 0.5, facecolor='#f56502', edgecolor='#212529')
    plt.xlabel('Lag'); plt.ylabel('Dependence'); plt.ylim([0, 1]); plt.xticks(np.arange(1, lag_bar + 1))

    return sw