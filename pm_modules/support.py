# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:06:15 2024

@author: Thomas

Umfang:
    Support Functionen die für jedes Package benötigt werden.
    Z.B. Dimension-Checks

"""

import numpy as np

a = np.ones(4)

def check_dim_2d(array: "np.array"):
    if a.ndim == 2:
        return True
    else:
        return False

