# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:06:15 2024

@author: Thomas

Umfang:
    Support Functionen die für jedes Package benötigt werden.
    Z.B. Dimension-Checks

"""

import numpy as np



# %%

def check_dim_2d(array: np.ndarray):
    if array.ndim == 2:
        return True
    else:
        return False


a = np.array([5,1])

e = check_dim_2d(a)


def test_data():
    # generates test data
    
    return None


# %% Analyse der Laufzeit

# Timeit
# Cprofile
# Ordnung (Komplexitätsordnung bestimmen)
a =6