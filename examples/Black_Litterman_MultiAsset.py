# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:57:58 2024

@author: Thomas
"""

# %% Import Data


# %% Estimation of Covariance


# %% Black Litterman


# %% Portfolio Optimization

import numpy as np
w = np.array([0.2,0.2,0.6]).reshape(-1,1)

lam = 2.5
cov = np.array([[0.04,0.02,0.0],
                [0.02,0.04,0.0],
                [0.0,0.0,0.004]])

lam * cov@w
