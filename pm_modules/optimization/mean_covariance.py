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
Wesentliche Parameter:
    
    1) Absolut Risk Aversion
        - Objective Function
        - Constraint (QPQC / SOC)
        
    2) Relative Risk Aversion (BM)
        - Objective Function
        - Einbeziehung der as Penality in Obj. Funktion (Lamda_B: Aversion to Tracking Error)
        - Constraint (QPQC / SOC)
        
    3) Return:
        - Objective Function
        - Constraint (Linear)
        
    5) Inclusion of External Information:
        - Sector / Factor / Regional Exposure
        - Membership to Assetclass
        - etc.
    
    6) Modelling of Max. Number of Active Positions

    ... Was noch?
    
    
[0, 0, 1, 0, 0] @ x = 0.2
    
       
    
Optionale Faktoren:
    1) Einbeziehung von Transaktionskosten
        - Heuristic Routine (ARPM)
        - Einbeziehung der TK as Penality in Obj. Funktion (Lamda_A: Aversion to Turnover)


Notiz:
    1) Ridge und Lasso in Static und Dynamic From
        Static Lambda*|x|
        Dynamic Lambda*|x-x0| -> Y in Arpm

'''



# -> ggf. primär Risk Folio nutzen für CVaR

# %% Estimation Theory: Bayes Action and Decision Framework

