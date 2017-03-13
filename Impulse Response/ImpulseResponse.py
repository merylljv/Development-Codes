# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:33:30 2017

@author: Win8
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize
from scipy.ndimage import filters

def step_fcn(t,A,d):
    return np.where(np.logical_and(t>=0,t<=d),1,0)*A

def gaussian(A,d,std):
    return A*signal.gaussian(d+1,std = std)

def RMSE(predicted,actual):
    return np.sqrt(np.sum(np.power(predicted - actual,2))/len(predicted))

def exp1(t,T,D):
    return np.exp(-np.log(2)*np.power((t-T)/D,2))

def exp2(t,L):
    return np.exp(-t*np.log(2)/L)

def transfer_function(n,T,D,L):
    t = np.arange(0,n)
    return filters.convolve1d(exp1(t,T,D),exp2(t,L)/exp2(t,L))

def model_output_signal(trans,input_signal,A):
    return A*filters.convolve1d(input_signal,trans)

def optimize_fun(par,input_signal,output_signal,n):
    trans = transfer_function(n,par[0],par[1],par[2])
    predicted = model_output_signal(trans,input_signal,par[3])
    return RMSE(predicted,output_signal)
def demo():
    ###MAIN
    ###Use SLSQP to obtain transfer function
    
    #Input signals
    n = 50 ### length of IRF
    
    #### Generating test signals
    t = np.arange(0,150,1)
    test_input = 5.0*np.cos(t/7.14)+np.random.rand(len(t))
    test_output = 2.0*np.sin(t/7.14)+np.random.rand(len(t))
    
    #### Parameter criteria
    bnds = ((0,None),(0,None),(0,None),(0,None))
    ###Finding optimized parameters using SLSQP
    
    ### Starting parameters
    T = 13 #### Lag in acceleration phasse
    D = 45 #### Duration of the acceleration phase
    L = 1 #### Recession constant
    A = 2/5. ### Normalization constant 
    
    
    res = minimize(optimize_fun,(T,D,L,A),args = (test_input,test_output,n),method = 'SLSQP',bounds = bnds)
    
    ####Optimized transfer function
    print res
    
    optimized_trans = transfer_function(n,res.x[0],res.x[1],res.x[2])
    predicted_output = model_output_signal(optimized_trans,test_input,res.x[3])
    
    ####Plot results
    plt.plot(t,test_input,label = 'Input')
    plt.plot(t,predicted_output,label = 'Predicted')
    plt.plot(t,test_output,label = 'Actual')
    plt.legend(loc = 'upper left')
    plt.show()