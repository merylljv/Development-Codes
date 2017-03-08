# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:33:30 2017

@author: Win8
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize

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

def transfer_function(t,T,D,L):
    return np.convolve(exp1(t,T,D),exp2(t,L),'same')*(t[1]-t[0])

def model_output_signal(trans,input_signal,A):
    return A*np.convolve(trans,input_signal,'same')

def optimize_fun(par,input_signal,output_signal,time_range):
    trans = transfer_function(time_range,par[0],par[1],par[2])
    predicted = model_output_signal(trans,input_signal,par[3])
    return RMSE(predicted,output_signal)

####MAIN
####Use SLSQP to obtain transfer function

##Input signals
t = np.arange(0,150,1)
test_input = 5.0*np.cos(t/7.14)+np.random.rand(len(t))
test_output = 5.0*np.sin(t/7.14)+np.random.rand(len(t))
bnds = ((0,None),(0,None),(0,None),(0,None))
###Finding optimized parameters using SLSQP
res = minimize(optimize_fun,(16,30.,1,5),args = (test_input,test_output,t),method = 'SLSQP',bounds = bnds)

####Optimized transfer function
print res

optimized_trans = transfer_function(t,res.x[0],res.x[1],res.x[2])
predicted_output = model_output_signal(optimized_trans,test_input,res.x[3])

####Plot results
plt.plot(t,test_input,label = 'Input')
plt.plot(t,predicted_output,label = 'Predicted')
plt.plot(t,test_output,label = 'Actual')