from scipy import stats
import numpy as np
import pandas as pd


def sum_square_residual(x,y,slope,intercept):
    return np.sum(np.square(y - intercept - slope * x))

def t_crit(confidence_level,n):
    gamma = 1 - confidence_level
    return stats.t.ppf(1-gamma/2.,n)

def uncertainty(x,y,slope,intercept,confidence_level):
    #### INPUT
    # Experimental x & y values should have the same length (array)
    # Computed slope & intercept (float)
    #### OUTPUT
    # Uncertainty on the prediction of simple linear regression
    
    n = len(x)
    t = t_crit(confidence_level,n-2)
    sum_epsilon_square = sum_square_residual(x,y,slope,intercept)
    mean_x = np.mean(x)
    var_x = np.sum(np.square(x - mean_x))
    epsilon = y - intercept - slope*x    
    
    return t*np.sqrt((1/(n-2)*sum_epsilon_square*(1/n + (epsilon - mean_x)**2/var_x)))

def confidence_interval_upper(x,y,slope,intercept,uncertainty):
    return slope*x + intercept + uncertainty

def confidence_interval_lower(x,y,slope,intercept,uncertainty):
    return slope*x + intercept - uncertainty