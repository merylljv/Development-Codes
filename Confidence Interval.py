import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md

from scipy import stats
import numpy as np
import pandas as pd

#### Confidence interval computations based on the following wiki article
#### https://en.wikipedia.org/wiki/Simple_linear_regression
#### See section 'Confidence Intervals'



def sum_square_residual(x,y,slope,intercept):
    #### Computes the sum of the square of the residual of the linear regression
    return np.sum(np.square(y - intercept - slope * x))

def t_crit(confidence_level,n):
    #### Computes for the critical t-value from two-tailed student's t-distribution 
    #### given the desired confidence level
    gamma = 1 - confidence_level
    return stats.t.ppf(1-gamma/2.,n)

def uncertainty(x,y,slope,intercept,confidence_level):
    #### INPUT
    # Experimental x & y values should have the same length (array)
    # Computed slope & intercept (float)
    #### OUTPUT
    # Uncertainty on the prediction of simple linear regression
    
    n = float(len(x))
    t = t_crit(confidence_level,n-2)
    sum_epsilon_square = sum_square_residual(x,y,slope,intercept)
    mean_x = np.mean(x)
    var_x = np.sum(np.square(x - mean_x))
    
    return t*np.sqrt((1/(n-2)*sum_epsilon_square*(1/n + (x - mean_x)**2/var_x)))

def confidence_interval_upper(x,y,slope,intercept,confidence_level):
    #### Computes for the upper bound values of the confidence interval
    delta = uncertainty(x,y,slope,intercept,confidence_level)
    return slope*x + intercept + delta

def confidence_interval_lower(x,y,slope,intercept,confidence_level):
    #### Computes for the lower bound values of the confidence interval
    delta = uncertainty(x,y,slope,intercept,confidence_level)
    return slope*x + intercept - delta
    
def test():
    #### Test run of the algorithm
    #### Simulate data points
    x = np.arange(0,100,1)
    y = x + 20*np.random.rand(len(x))
    
    #### Compute simple linear regression parameters
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    #### Compute upper and lower bound of confidence interval for 99.99% level
    upper = confidence_interval_upper(x,y,slope,intercept,0.9999)
    lower = confidence_interval_lower(x,y,slope,intercept,0.9999)
    
    #### Plot the results
    plt.plot(x,y,'.',c = 'blue',label = 'Data points')
    plt.plot(x,slope*x + intercept,c = 'red',label = 'Simple LR')
    plt.plot(x,upper,'--',c = 'red',label = 'Confidence Interval')
    plt.plot(x,lower,'--',c = 'red')
    plt.legend(loc = 'upper left')
    plt.show()
    
    