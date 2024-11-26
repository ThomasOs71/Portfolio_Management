# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 19:09:22 2024

@author: Thomas
"""

# %% Packages
import numpy as np
import pandas as pd
import scipy.linalg as lg


# %% PF-Optimierungen
'''
Was brauche ich?

Optimierung auf Absolut Risk?
Constraints auf Tracking Error
Max. Active Positions

Ridge und Lasso in Static und Dynamic From

Static Lambda*|x|
Dynamic Lambda*|x-x0|


'''



# -> ggf. primär Risk Folio nutzen


a = np.array([0.2,0.3,0.4,0.6]).reshape(-1,1)

b = a@a.T

a.T == a

a = lg.sqrtm(b).real

a.T == a

n = np.array(range(1,5+1))
b = np.array([1,3])

np.setdiff1d(n,b)
b = np.zeros((1,3))
