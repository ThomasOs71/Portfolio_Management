# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:13:57 2024

@author: Thomas
"""

## Test

import numpy as np

import matplotlib.pyplot as plt

z = np.random.normal(0,1,1000)



### Univariate
z = np.random.normal(0,1,1000).reshape(-1,1)
z_star = np.array(0).reshape(-1,1)
h = np.array(1).reshape(-1,1)
gamma = 2

p = state_conditioning_smooth_kernel_fp(z,z_star,h,gamma)


plt.plot(p_smooth)

plt.plot(np.ones(len(p_smooth))/len(p_smooth))



### Multivariate
z = np.random.multivariate_normal([0,0],np.eye(2),1000)
z_star = np.array([0.1,0.1],ndmin = 2)
h = np.array([[1,0],
                     [0,1]])
gamma = 2

p = state_conditioning_smooth_kernel_fp(z,z_star,h,gamma)

plt.plot(p)

effective_number_of_scenarios(p)



z = np.random.normal(0,1,1000).reshape(-1,1)
z_star = np.array(0).reshape(-1,1)
gamma = 2



a = conditional_fp(z,
                   z_star,
                   0.4,
                   p)









