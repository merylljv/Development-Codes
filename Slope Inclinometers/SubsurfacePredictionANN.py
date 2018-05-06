# -*- coding: utf-8 -*-
"""
Created on Sun May 06 17:04:19 2018

@author: Data Scientist 1
"""

import SlopeInclinometers as si
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
from scipy import stats
from sklearn.neural_network import MLPRegressor

data_path = os.path.dirname(os.path.realpath(__file__))

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def GetSubsurfaceData(time_start,time_end,sensor_column,node_id,compute_vel = False):
    '''
    Gets the displacement data frame given the sensor name, node and timestamps
    
    Parameters
    ------------------------------
    time_start - string
        Desired start time
    time_end - string
        Desired end time
    sensor_column - string
        Name of sensor column
    node_id - int
        Node number of interest
    compute_vel - Boleean
        Adds velocity if True
    
    Returns
    -------------------------------
    subsurface_data - pd.DataFrame()
        Subsurface data frame
    sensor_column - string
        Name of sensor column
    '''
    
    #### Convert into tuple timestamps the given times
    timestamp = (pd.to_datetime(time_start),pd.to_datetime(time_end))
    
    #### Call the GetDispDataFrame from slope inclinometers
    subsurface_data,name = si.GetDispDataFrame(timestamp,sensor_column,compute_vel)
    
    #### Embed name to data frame
    subsurface_data['name'] = sensor_column
    
    return subsurface_data[subsurface_data.id == node_id][['ts','id','name','xz']]

def GetDisplacementPlots(subsurface_data):
    '''
    Plots the subsurface data plot, and displacement plot with linear regression
    
    Parameters
    -----------------------------------
    subsurface_data - pd.Dataframe()
        Subsurface data frame
    sensor_column - string
        Name of sensor column
    
    Returns
    -----------------------------------
    subsurface_data - pd.Dataframe()
        Subsurface data with displacement and time column
    '''
    
    #### Get displacement column
    subsurface_data['disp'] = subsurface_data.xz - subsurface_data.xz.shift(1)
    
    #### Get time column
    subsurface_data['time'] = (subsurface_data.ts - subsurface_data.ts.values[0]) / np.timedelta64(1,'D')
    
    #### Get linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(subsurface_data.time[1:],subsurface_data.disp[1:])
    
    ##############################
    #### Plot subsurface data ####
    ##############################
    
    #### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot subsurface data
    ax.plot(subsurface_data.ts,subsurface_data.xz)
    
    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 14)
    ax.set_ylabel('Cumulative subsurface displacement (m)',fontsize = 14)
    
    #### Set fig size, borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.15)
    
    #### Set save path
    save_path = "{}/Subsurface Displacement//".format(data_path)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/{} {}.png'.format(save_path,subsurface_data.name.values[0],int(subsurface_data.id.values[0])),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w',bbox_inches = 'tight')
    
    ################################
    #### Plot displacement data ####
    ################################
    
    ### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot displacment data
    ax.plot(subsurface_data.ts,subsurface_data.disp,label = 'Displacement')
    
    #### Plot linear regression
    ax.plot(subsurface_data.ts.values[1:],slope*subsurface_data.time.values[1:] + intercept,
            '--',label = 'Linear Fit')
    
    #### Plot legend
    ax.legend(fontsize = 12)
    
    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 14)
    ax.set_ylabel('Displacement (m)',fontsize = 14)
    
    #### Set fig size, borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.15)
    
    #### Set save path
    save_path = "{}/Displacement//".format(data_path)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/{} {}.png'.format(save_path,subsurface_data.name.values[0],int(subsurface_data.id.values[0])),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w',bbox_inches = 'tight')
    
    return subsurface_data

def GetFeatureVectors(subsurface_data,m,train_ratio,ave_window = 48):
    '''
    Creates the normalized input feature vectors and output vector from the time series divided according to train and test set
    
    Parameters
    -----------------------------
    subsurface_data - pd.DataFrame()
        DataFrame with displacement and time column
    m - int
        Embedding dimension
    train_ratio - float
        Desire ratio between training and testing sets
    ave_window - int
        Moving average window
    
    Returns
    -----------------------------
    train_x, train_y, test_x, test_y - np.array
    '''
    
    #### Drop na from data frame
    subsurface_data.dropna(inplace = True)
    
    #### Get smoothened displacement data
    displacement = subsurface_data.disp.rolling(ave_window).mean().dropna().values
    
#    #### Smoothen displacement data by moving averages
#    displacement = movingaverage(displacement,ave_window)
    
    #### Normalize displacement values
    displacement = (displacement - min(displacement))/(max(displacement) - min(displacement))
    
    #### Initialize results list
    X = []
    Y = []
    
    #### Iterate through the data frame based on m
    for i in range(len(displacement)-m):
        
        #### Add to list
        X.append(displacement[i:i+m])
        Y.append(displacement[i+m])
    
    #### Convert results to array
    X = np.array(X)
    Y = np.array(Y)
    
    #### Get last index of training set
    last_index = int(len(X)*train_ratio)
    
    #### Split into training set and test set
    train_x = X[0:last_index]
    train_y = Y[0:last_index]
    test_x = X[last_index:]
    test_y = Y[last_index:]
    
    return train_x, train_y, test_x, test_y
    
    
    
    
    
    
    
    
    
    
    