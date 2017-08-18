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

#### Impulse Response Function based on 
#### "The application of an innovative inverse model for understanding and predicting landslide movements Belle, Anunay et. al 2014"

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)


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
    return filters.convolve1d(exp1(t,T,D),exp2(t,L))

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
    test_input = 5.0*np.cos(t/7.14) + np.random.random(len(t))
    test_output = 2.0*np.sin(t/7.14) + np.random.random(len(t))
    
    #### Parameter criteria
    bnds = ((0,None),(0,None),(0,None),(0,None))
    
    ###Finding optimized parameters using SLSQP
    
    ### Starting parameters
    T = 13 #### Lag in acceleration phasse
    D = 45 #### Duration of the acceleration phase
    L = 1  #### Recession constant
    A = 2/5. ### Normalization constant 
    
    
    res = minimize(optimize_fun,(T,D,L,A),args = (test_input,test_output,n),method = 'SLSQP',bounds = bnds,options = {'maxiter':1000})
    
    ####Optimized transfer function
    print res
    
    optimized_trans = transfer_function(n,res.x[0],res.x[1],res.x[2])
    predicted_output = model_output_signal(optimized_trans,test_input,res.x[3])
    
    ####Plot results
    fig = plt.figure(num = 1)
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(t,test_input,color = tableau20[0],label = 'Input',lw = 1.5)
    ax.plot(t,predicted_output,color = tableau20[4],label = 'Predicted',lw = 1.5)
    ax.plot(t,test_output,color = tableau20[6],label = 'Actual',lw = 1.5)
    ax.legend(loc = 'upper left',fancybox = True,framealpha=0.5)
    
    #### Plot transfer functions
    fig = plt.figure(num = 2)
    ax = fig.add_subplot(111)
    ax.grid()
    t1 = np.arange(0,150,0.01)
    ax.plot(t1,exp1(t1,T,D),color = tableau20[6],label = 'Gaussian',lw = 1.5)
    ax.plot(t1,exp2(t1,L),color = tableau20[4],label = 'Exp Decay',lw = 1.5)
#    ax.plot(t,transfer_function(n,T,D,L),color = tableau20[0],label = 'Transfer Function',lw = 1.5)
    ax.legend(loc = 'upper left',fancybox = True,framealpha=0.5)