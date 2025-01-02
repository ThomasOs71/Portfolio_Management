#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt


def invariance_test_ks(eps, *, conf_lev=0.95, title='Kolmogorov-Smirnov test', plot_test=True):
    """This function conducts a Kolmogorov-Smirnov test on a given dataset by computing 
    the sample autocorrelations, confidence intervals, and plotting the results in a histogram, 
    with options for adjusting the plot and fitting distributions.

    # TODO -> Unterscheidung Absolut -> Garch, Normal -> AK

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
