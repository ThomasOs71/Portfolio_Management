#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
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

