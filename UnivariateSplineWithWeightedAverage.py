import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import gaussian
from scipy.ndimage import filters

#### Function univ_spline based on the following reference:
#### http://www.nehalemlabs.net/prototype/blog/2014/04/12/how-to-fix-scipys-interpolating-spline-default-behavior/

def GoodnessOfSplineFit(x,y,sp):
    #### Calculate sum of square residuals, coeffient of determination, and root mean square of spline fit
    mean = np.mean(y)
    n = float(len(y))
    SS_tot = np.sum(np.power(y-mean,2))
    SS_res = np.sum(np.power(y-sp(x),2))
    coef_determination = 1 - SS_res/SS_tot
    RMSE = np.sqrt(SS_res/n)
    
    return SS_res,coef_determination,RMSE    

def moving_average(series,sigma = 3):
    #### Moving weighted gaussian average with window = 39
    b = gaussian(39,sigma)
    average = filters.convolve1d(series,b/b.sum())
    var = filters.convolve1d(np.power(series-average,2),b/b.sum())
    
    return average,var

def univ_spline(x,y):
    #### Univariate spline with weights corresponding to variance at each point
    _,var = moving_average(y)
    
    return UnivariateSpline(x,y,w = 1/np.sqrt(var))

def demo():
    #### Demo of univariate spline
    #### Generate test data points
    x = np.arange(0,100,0.1)
    y = np.sin(x) + 0.1*np.random.rand(len(x))
    
    #### Compute for default cubic spline interpolation
    sp = univ_spline(x,y)
    
    #### Interpolated points
    ### Zeroth derivative
    y_int = sp(x)
    
    ### First derivative
    y_der = sp.derivative(n=1)(x)
    ### Second derivative
    y_2_der = sp.derivative(n=2)(x)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(x,y,'.',label = 'Data points')
    ax1.plot(x,y_int,'-',label = 'Spline interpolation')
    ax1.legend(loc = 'upper left',fontsize = 10)    
    
    ax2 = fig.add_subplot(223)
    ax2.plot(x,y_der)
    ax2.set_title('First Derivative')
    
    ax3 = fig.add_subplot(224)
    ax3.plot(x,y_2_der)
    ax3.set_title('Second Derivative')
    
    SS_res,r2,RMSE = GoodnessOfSplineFit(x,y,sp)
    
    print "Spline fit evaluation:\n\nSum of square residuals = {}\nCoefficient of determination (r^2) = {}\nRoot-mean square error = {}".format(SS_res,r2,RMSE)    
    
    plt.show()

