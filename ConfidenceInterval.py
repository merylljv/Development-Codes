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

def uncertainty(x,y,slope,intercept,confidence_level,x_array = None,interval = 'confidence'):
    #### INPUT
    # x,y -> Experimental x & y values should have the same length (array)
    # Input x_array to evaluate uncertainty at the specified array
    # Computed slope & intercept (float)
    # interval - type of interval to compute (confidence or prediction)
    #### OUTPUT
    # Uncertainty on the prediction of simple linear regression
    
    n = float(len(x))
    t = t_crit(confidence_level,n-2)
    sum_epsilon_square = sum_square_residual(x,y,slope,intercept)
    mean_x = np.mean(x)
    var_x = np.sum(np.square(x - mean_x))
    if x_array == None:
        if interval == 'confidence':
            return t*np.sqrt((1/(n-2)*sum_epsilon_square*(1/n + (x - mean_x)**2/var_x)))
        elif interval == 'prediction':
            return t*np.sqrt((1/(n-2)*sum_epsilon_square*(1. + 1./n + (x - mean_x)**2/var_x)))
        else:
            print "Invalid type of interval"
            raise ValueError
    else:
        if interval == 'confidence':
            return t*np.sqrt((1/(n-2)*sum_epsilon_square*(1/n + (x_array - mean_x)**2/var_x)))
        elif interval == 'prediction':
            return t*np.sqrt((1/(n-2)*sum_epsilon_square*(1. + 1/n + (x_array - mean_x)**2/var_x)))
        else:
            print "Invalid type of interval"
            raise ValueError

def confidence_interval_upper(x,y,slope,intercept,confidence_level,x_array = None,interval = 'confidence'):
    #### Computes for the upper bound values of the confidence interval
    delta = uncertainty(x,y,slope,intercept,confidence_level,x_array,interval)
    return slope*x + intercept + delta

def confidence_interval_lower(x,y,slope,intercept,confidence_level,x_array = None,interval = 'confidence'):
    #### Computes for the lower bound values of the confidence interval
    delta = uncertainty(x,y,slope,intercept,confidence_level,x_array,interval)
    return slope*x + intercept - delta

def slope_unc(x,y,slope,intercept,confidence_level,interval = 'confidence'):
    #### Computes for the uncertainty of the measurement of slope for specified type of interval
    #### x,y - data points
    #### slope, intercept - linear regression results
    #### confidence_level - specified confidence level
    #### interval - choose from confidence or prediction

    MSE = 1 / float(len(x) - 2) * sum_square_residual(x,y,slope,intercept)
    t = t_crit(confidence_level,len(x)-2)
    if interval == 'confidence':
        return t*np.sqrt(MSE / np.sum(np.power(x - np.mean(x),2)))
    elif interval == 'prediction':
        return t*np.sqrt(MSE / np.sum(np.power(x - np.mean(x),2)) + MSE)
    else:
        print "Invalid type of interval"
        raise ValueError

def intercept_unc(x,y,slope,intercept,confidence_level,interval = 'confidence'):
    #### Computes for the uncertainty of the measurement of intercept for specified type of interval
    #### x,y - data points
    #### slope, intercept - linear regression results
    #### interval - choose from confidence or prediction    

    MSE = 1 / float(len(x) - 2) * sum_square_residual(x,y,slope,intercept)
    t = t_crit(confidence_level,len(x)-2)

    if interval == 'confidence':
        return t*np.sqrt(MSE * (1/float(len(x)) + np.mean(x)**2 / np.sum(np.power(x - np.mean(x),2))))
    elif interval == 'prediction':
        return t*np.sqrt(MSE * (1 + 1/float(len(x)) + np.mean(x)**2 / np.sum(np.power(x - np.mean(x),2))))
    else:
        print "Invalid type of interval"
        raise ValueError

def dist_point_to_line(x,y,slope,intercept):
    #### Computes for the distance between point x,y to a line with given slope and intercept
    #### x,y - array
    #### slope, intercept - linear regression results
    return np.abs(slope*x - y + intercept) / np.sqrt(slope**2 + 1)

def test():
    #### Test run of the algorithm
    #### Simulate data points
    x = np.arange(0,100,1)
    y = x + 20*np.random.rand(len(x))
    
    #### Compute simple linear regression parameters
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    #### Compute upper and lower bound of confidence interval for 95.00% level
    upper = confidence_interval_upper(x,y,slope,intercept,0.950)
    lower = confidence_interval_lower(x,y,slope,intercept,0.950)
    upper_pred = confidence_interval_upper(x,y,slope,intercept,0.950,interval = 'prediction')
    lower_pred = confidence_interval_lower(x,y,slope,intercept,0.950,interval = 'prediction')
    #### Plot the results
    plt.plot(x,y,'.',c = 'blue',label = 'Data points')
    plt.plot(x,slope*x + intercept,c = 'red',label = 'Simple LR')
    plt.plot(x,upper,'--',c = 'red',label = 'Confidence Interval')
    plt.plot(x,lower,'--',c = 'red')
    plt.plot(x,upper_pred,'--',c = 'green',label = 'Prediction Interval')
    plt.plot(x,lower_pred,'--',c = 'green')
    plt.legend(loc = 'upper left')
    plt.show()
    
    