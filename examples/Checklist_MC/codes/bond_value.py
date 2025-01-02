# -*- coding: utf-8 -*-
import numpy as np
from zcb_value import zcb_value


def bond_value(t_hor, x_thor, tau, c, r, rd, facev=1, eta=0.013):
   
    """This function calculates the value of a bond at a given time horizon, based on the scenario matrix, 
    with coupon payments, corresponding maturities, and interest rates. It computes the value 
    of both zero-coupon bonds and coupon bonds, considering only coupons after the horizon time, 
    and returning the bond values.

    Parameters
    ----------
        t_hor : date
        x_thor : array, shape(j_bar, d_bar)
        tau : array, shape(k_bar,)
        c : array, shape(k_bar,)
        r : array, shape(k_bar,)
        rd : string, optional
        facev : scalar, optional
        eta : scalar, optional

    Returns
    -------
        v : array, shape(j_bar,)

    """
    
    if (rd != 'sr') and (rd != 'ns'):
        rd = 'y'
    
    # consider only coupons after the horizon time
    c = c[r >= t_hor]
    r = r[r >= t_hor]
    
    # compute zero-coupon bond value 
    v_zcb = zcb_value(t_hor, x_thor, tau, r, rd)
    # include notional
    c[-1] = c[-1] + 1  

    # compute coupon bond value
    v_zcb = v_zcb.reshape(1, v_zcb.shape[0]) if np.ndim(v_zcb) == 1 else v_zcb 
    v = facev*(v_zcb@c)
    
    return v.reshape(-1)
