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
import itertools

data_path = os.path.dirname(os.path.realpath(__file__))

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

dyna_colors = [(22,82,109),(153,27,30),(248,153,29)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

for i in range(len(dyna_colors)):
    r_d,g_d,b_d = dyna_colors[i]
    dyna_colors[i] = (r_d / 255., g_d / 255., b_d / 255.)

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

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
    ax.plot(subsurface_data.ts,subsurface_data.xz,lw = 4.5,color = dyna_colors[1])
    
    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
    
    #### Set fontsize for ticks
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 20)
    ax.set_ylabel('Cumulative subsurface displacement (m)',fontsize = 20)
    
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
    ax.plot(subsurface_data.ts,subsurface_data.disp,label = 'Displacement',lw = 4.5)
    
    #### Plot linear regression
    ax.plot(subsurface_data.ts.values[1:],slope*subsurface_data.time.values[1:] + intercept,
            '--',label = 'Linear Fit',lw = 2.75)
    
    #### Plot legend
    ax.legend(fontsize = 16)
    
    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
    
    #### Set fontsize for ticks
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 20)
    ax.set_ylabel('Displacement (m)',fontsize = 20)
    
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

def GetFeatureVectors(subsurface_data,m,train_ratio,ave_window = '4H'):
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
    ave_window - str
        Resampling size
    
    Returns
    -----------------------------
    train_x, train_y, test_x, test_y - np.array
    '''
    
    #### Drop na from data frame
    subsurface_data.dropna(inplace = True)
    
    #### Get smoothened and resampled displacement data
    displacement = subsurface_data[['ts','disp']].resample(ave_window,label = 'right',closed = 'right',on = 'ts').mean().disp.dropna().values

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

def GetFeatureVectors2(subsurface_data,m,train_ratio,ave_window = '4H',smooth_window = 3):
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
    ave_window - str
        Resampling size
    smooth_window - int
        Moving average window
    
    
    Returns
    -----------------------------
    train_x, train_y, test_x, test_y - np.array
    '''
    
    #### Get smoothened and resampled displacement data
    subsurface_data_xz = subsurface_data[['ts','xz']].resample(ave_window,label = 'right',closed = 'right',on = 'ts').mean().rolling(smooth_window).mean()

    #### Get displacement column
    subsurface_data_xz['disp'] = subsurface_data_xz.xz - subsurface_data_xz.xz.shift(1)
    displacement = subsurface_data_xz.disp.dropna().values

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


def GetFeatureVectorsMultOut(subsurface_data, m, train_ratio, ave_window = '4H',
                             smooth_window = 3, out = 2):
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
    ave_window - str
        Resampling size
    smooth_window - int
        Moving average window
    output - int
        Number of outputs to be predicted by MLPR
    
    Returns
    -----------------------------
    train_x, train_y, test_x, test_y - np.array
    '''
    
    #### Get smoothened and resampled displacement data
    subsurface_data_xz = subsurface_data[['ts','xz']].resample(ave_window,label = 'right',closed = 'right',on = 'ts').mean().rolling(smooth_window).mean()

    #### Get displacement column
    subsurface_data_xz['disp'] = subsurface_data_xz.xz - subsurface_data_xz.xz.shift(1)
    displacement = subsurface_data_xz.disp.dropna().values

    #### Normalize displacement values
    displacement = (displacement - min(displacement))/(max(displacement) - min(displacement))
    
    #### Initialize results list
    X = []
    Y = []
    
    #### Iterate through the data frame based on m
    for i in range((len(displacement)-m)/out):
        
        #### Add to list
        X.append(displacement[out*i:out*i+m])
        Y.append(displacement[out*i+m:out*i+m+out])
    
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


def GetOptimalCLF(train_x,train_y,rand_starts = 8):
    '''
    Gets the optimal CLF function based on fixed settings
    
    Parameters
    ------------------------
    train_x - np.array
        Training feature vectors
    train_y - np.array
        Training label vectors
    rand_starts - int
        Number of random starts to do
        Default - 8 for 95% confidence and best 30%
    
    Returns
    ------------------------
    max_clf - sklearn function
        Optimal trained artificial neuron network
    '''
    
    #### Get number of feature inputs of training vector
    n_input = train_x.shape[1]
    
    #### Set initial loss value
    min_loss = 1e10
    
    #### Perform number of trainings according to random start set
    for i in range(rand_starts):
        
        #### Print current status
        print "Iteration number {}".format(i+1)
        
        #### Initialize ANN network
        clf = MLPRegressor(hidden_layer_sizes = (2*n_input,1), activation = 'logistic',solver = 'sgd', 
                           learning_rate = 'adaptive', max_iter = 100000000,tol = 1e-10,
                           early_stopping = True, validation_fraction = 1/3.)
        
        #### Fit data
        clf.fit(train_x,train_y)
        
        #### Get current loss
        cur_loss = clf.loss_
        
        #### Save current clf if loss is minimum
        if cur_loss < min_loss:
            
            #### Set min_loss to a new value
            min_loss = cur_loss
            print min_loss
            print clf            
            #### Set max_clf to new value
            max_clf = clf
    
    return max_clf

def GetOptimalCLF2(train_x,train_y,rand_starts = 8):
    '''
    Gets the optimal CLF function based on fixed settings
    
    Parameters
    ------------------------
    train_x - np.array
        Training feature vectors
    train_y - np.array
        Training label vectors
    rand_starts - int
        Number of random starts to do
        Default - 8 for 95% confidence and best 30%
    
    Returns
    ------------------------
    max_clf - sklearn function
        Optimal trained artificial neuron network
    '''
    
    #### Get number of feature inputs of training vector
    n_input = train_x.shape[1]
    
    #### Set initial loss value
    min_loss = 1e10
    
    #### Perform number of trainings according to random start set
    for i in range(rand_starts):
        
        #### Print current status
        print "Iteration number {}".format(i+1)
        
        #### Initialize ANN network
        clf = MLPRegressor(hidden_layer_sizes = (int(round(2*np.sqrt(n_input),0)),1), activation = 'logistic',solver = 'sgd', 
                           learning_rate = 'adaptive', max_iter = 100000000,tol = 1e-10,
                           early_stopping = True, validation_fraction = 1/3.)
        
        #### Fit data
        clf.fit(train_x,train_y)
        
        #### Get current loss
        cur_loss = clf.loss_
        print cur_loss
        print clf
        #### Save current clf if loss is minimum
        if cur_loss < min_loss:
            
            #### Set min_loss to a new value
            min_loss = cur_loss

            #### Set max_clf to new value
            max_clf = clf
    
    return max_clf


def GetTrainTestPlotResult(subsurface_data,clf,train_x,train_y,test_x,test_y,ave_window = '4H'):
    '''
    Gets the superimposed plots of the actual and predicted training and test data via the trained ANN.
    Computes the rmse error of the classifier.
    
    Parameters
    -------------------
    subsurface_data - pd.DataFrame()
        Subsurface data frame
    clf - sklearn classifier
        Trained ANN
    train_x - np.array
        Training feature vectors
    train_y - np.array
        Training label vectors
    test_x - np.array
        Test feature vectors
    test_y - np.array
        Test label vectors
    ave_window - int
        Moving average window
    
    Returns
    -------------------
    test_rmse - root mean square error of the classifier
    '''
    
    #### Get data predictions of ANN
    pred_train = clf.predict(train_x)
    pred_test = clf.predict(test_x)
    
    #### Get starting indexes for plotting
    start_index = len(subsurface_data) - (len(train_y) + len(test_y))
    
    ############################################
    #### Plot displacement data predictions ####
    ############################################
    
    #### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot measured data points
    ax.plot(subsurface_data.ts.values[start_index:],np.concatenate((train_y,test_y)),label = 'Measured')
    
    #### Plot predicted data points
    ax.plot(subsurface_data.ts.values[start_index:],np.concatenate((pred_train,pred_test)),lw = 0.85,label = 'Predicted')
    
    #### Plot predicted boundary
    ax.axvline(subsurface_data.ts.values[start_index + len(train_y)],ls = '--',color = tableau20[14])
    
    #### Plot legend
    ax.legend(bbox_to_anchor = (1.20,0.999),fontsize = 12)
    
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
    save_path = "{}/ANN//traintest//{} {}//".format(data_path,subsurface_data.name.values[0],int(subsurface_data.id.values[0]))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/m {}.png'.format(save_path,int(train_x.shape[1])),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w',bbox_inches = 'tight')
    
    #######################################################
    #### Plot cumulative displacement data predictions ####
    #######################################################
    
    #### Drop na from data frame
    subsurface_data.dropna(inplace = True)
    
    #### Get smoothened and resampled displacement data
    displacement = subsurface_data[['ts','disp']].resample(ave_window,label = 'right',closed = 'right',on = 'ts').mean().disp.dropna().values
    
    #### Denormalize quantities
    dn_actual = np.concatenate((train_y,test_y))*(max(displacement) - min(displacement)) + min(displacement)
    dn_pred = np.concatenate((pred_train,pred_test))*(max(displacement) - min(displacement)) + min(displacement)
    print 'dn_actual\n', dn_actual
    print 'dn_pred\n', dn_pred
    print 'test_y\n', test_y
    print 'pred_test\n', pred_test
    
    #### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot measured data points
    ax.plot(subsurface_data.ts.values[start_index:],np.cumsum(dn_actual),label = 'Measured')
    
    #### Plot predicted data points
    ax.plot(subsurface_data.ts.values[start_index:],np.cumsum(dn_pred),lw = 0.85,label = 'Predicted')
    
    #### Plot predicted boundary
    ax.axvline(subsurface_data.ts.values[start_index + len(train_y)],ls = '--',color = tableau20[14])
    
    #### Plot legend
    ax.legend(bbox_to_anchor = (1.20,0.999),fontsize = 12)
    
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
    save_path = "{}/ANN//traintestcumsum//{} {}//".format(data_path,subsurface_data.name.values[0],int(subsurface_data.id.values[0]))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/m {}.png'.format(save_path,int(train_x.shape[1])),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w',bbox_inches = 'tight')
    
    #### Compute the rmse
    rmse = np.sqrt(np.sum(np.power(test_y - pred_test,2))/float(len(test_y)))
    
    return rmse

def GetTrainTestPlotResult2(subsurface_data,clf,train_x,train_y,test_x,test_y,ave_window = '4H',smooth_window = 12):
    '''
    Gets the superimposed plots of the actual and predicted training and test data via the trained ANN.
    Computes the rmse error of the classifier.
    
    Parameters
    -------------------
    subsurface_data - pd.DataFrame()
        Subsurface data frame
    clf - sklearn classifier
        Trained ANN
    train_x - np.array
        Training feature vectors
    train_y - np.array
        Training label vectors
    test_x - np.array
        Test feature vectors
    test_y - np.array
        Test label vectors
    ave_window - int
        Moving average window
    
    Returns
    -------------------
    test_rmse - root mean square error of the classifier
    '''
    
    #### Get data predictions of ANN
    pred_train = clf.predict(train_x)
    pred_test = clf.predict(test_x)
    
    #### Get starting indexes for plotting
    start_index = len(subsurface_data) - (len(train_y) + len(test_y))
    
    ############################################
    #### Plot displacement data predictions ####
    ############################################
    
    #### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot measured data points
    ax.plot(subsurface_data.ts.values[start_index:],np.concatenate((train_y,test_y)),label = 'Measured',lw = 3.0)
    
    #### Plot predicted data points
    ax.plot(subsurface_data.ts.values[start_index:],np.concatenate((pred_train,pred_test)),lw = 1.5,label = 'Predicted')
    
    #### Plot predicted boundary
    ax.axvline(subsurface_data.ts.values[start_index + len(train_y)],ls = '--',color = tableau20[14])
    
    #### Plot legend
    ax.legend(bbox_to_anchor = (1.22,1.0),fontsize = 14)
    
    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 16)
    ax.set_ylabel('Displacement (m)',fontsize = 16)
    
    #### Set fig size, borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.15)
    
    #### Set save path
    save_path = "{}/ANN2//traintest//{} {}//".format(data_path,subsurface_data.name.values[0],int(subsurface_data.id.values[0]))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/m {} {} smooth {}.png'.format(save_path,int(train_x.shape[1]),ave_window,smooth_window),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w',bbox_inches = 'tight')
    
    #######################################################
    #### Plot cumulative displacement data predictions ####
    #######################################################
    
    #### Get smoothened and resampled displacement data
    subsurface_data_xz = subsurface_data[['ts','xz']].resample(ave_window,label = 'right',closed = 'right',on = 'ts').mean().rolling(smooth_window).mean()
    
    #### Get displacement column
    subsurface_data_xz['disp'] = subsurface_data_xz.xz - subsurface_data_xz.xz.shift(1)
    displacement = subsurface_data_xz.disp.dropna().values

    #### Denormalize quantities
    dn_actual = np.concatenate((train_y,test_y))*(max(displacement) - min(displacement)) + min(displacement)
    dn_pred = np.concatenate((pred_train,pred_test))*(max(displacement) - min(displacement)) + min(displacement)
    
    #### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot measured data points
    ax.plot(subsurface_data.ts.values[start_index:],np.cumsum(dn_actual),label = 'Measured',lw = 3.0)
    
    #### Plot predicted data points
    ax.plot(subsurface_data.ts.values[start_index:],np.cumsum(dn_pred),lw = 1.5,label = 'Predicted')
    
    #### Plot predicted boundary
    ax.axvline(subsurface_data.ts.values[start_index + len(train_y)],ls = '--',color = tableau20[14])
    
    #### Plot legend
    ax.legend(bbox_to_anchor = (1.22,1.0),fontsize = 14)
    
    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 16)
    ax.set_ylabel('Displacement (m)',fontsize = 16)
    
    #### Set fig size, borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.15)
    
    #### Set save path
    save_path = "{}/ANN2//traintestcumsum//{} {}//".format(data_path,subsurface_data.name.values[0],int(subsurface_data.id.values[0]))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/m {} {} smooth {}.png'.format(save_path,int(train_x.shape[1]),ave_window,smooth_window),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w',bbox_inches = 'tight')
    
    #### Compute the rmse
    rmse = np.sqrt(np.sum(np.power(test_y - pred_test,2))/float(len(test_y)))
    
    return rmse


def GetTrainTestPlotResultMultOut(subsurface_data,clf,train_x,train_y,test_x,test_y,
                                  ave_window = '4H',out=2,smooth_window = 12):
    '''
    Gets the superimposed plots of the actual and predicted training and test data via the trained ANN.
    Computes the rmse error of the classifier.
    
    Parameters
    -------------------
    subsurface_data - pd.DataFrame()
        Subsurface data frame
    clf - sklearn classifier
        Trained ANN
    train_x - np.array
        Training feature vectors
    train_y - np.array
        Training label vectors
    test_x - np.array
        Test feature vectors
    test_y - np.array
        Test label vectors
    ave_window - int
        Moving average window
    
    Returns
    -------------------
    test_rmse - root mean square error of the classifier
    '''
    
    #### Get data predictions of ANN
    pred_train = clf.predict(train_x)
    pred_test = clf.predict(test_x)
    
    train_y = np.array(list(itertools.chain.from_iterable(train_y)))
    test_y = np.array(list(itertools.chain.from_iterable(test_y)))
    pred_train = np.array(list(itertools.chain.from_iterable(pred_train)))
    pred_test = np.array(list(itertools.chain.from_iterable(pred_test)))
    
    #### Get starting indexes for plotting
    start_index = len(subsurface_data) - (len(train_y) + len(test_y))
    
    ############################################
    #### Plot displacement data predictions ####
    ############################################
    
    #### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot measured data points
    ax.plot(subsurface_data.ts.values[start_index:],np.concatenate((train_y,test_y)),label = 'Measured',lw = 3.0)
    
    #### Plot predicted data points
    ax.plot(subsurface_data.ts.values[start_index:],np.concatenate((pred_train,pred_test)),lw = 1.5,label = 'Predicted')
    
    #### Plot predicted boundary
    ax.axvline(subsurface_data.ts.values[start_index + len(train_y)],ls = '--',color = tableau20[14])
    
    #### Plot legend
    ax.legend(bbox_to_anchor = (1.22,1.0),fontsize = 14)
    
    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 16)
    ax.set_ylabel('Displacement (m)',fontsize = 16)
    
    #### Set fig size, borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.15)
    
    #### Set save path
    save_path = "{}/ANN_Mult//traintest//{} {}//".format(data_path,subsurface_data.name.values[0],int(subsurface_data.id.values[0]))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/m {} out {} {} smooth {}.png'.format(save_path,int(train_x.shape[1]),out,ave_window,smooth_window),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w',bbox_inches = 'tight')
    
    #######################################################
    #### Plot cumulative displacement data predictions ####
    #######################################################
    
    #### Get smoothened and resampled displacement data
    subsurface_data_xz = subsurface_data[['ts','xz']].resample(ave_window,label = 'right',closed = 'right',on = 'ts').mean().rolling(smooth_window).mean()
    
    #### Get displacement column
    subsurface_data_xz['disp'] = subsurface_data_xz.xz - subsurface_data_xz.xz.shift(1)
    displacement = subsurface_data_xz.disp.dropna().values

    #### Denormalize quantities
    dn_actual = np.concatenate((train_y,test_y))*(max(displacement) - min(displacement)) + min(displacement)
    dn_pred = np.concatenate((pred_train,pred_test))*(max(displacement) - min(displacement)) + min(displacement)
    
    #### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot measured data points
    ax.plot(subsurface_data.ts.values[start_index:],np.cumsum(dn_actual),label = 'Measured',lw = 3.0)
    
    #### Plot predicted data points
    ax.plot(subsurface_data.ts.values[start_index:],np.cumsum(dn_pred),lw = 1.5,label = 'Predicted')
    
    #### Plot predicted boundary
    ax.axvline(subsurface_data.ts.values[start_index + len(train_y)],ls = '--',color = tableau20[14])
    
    #### Plot legend
    ax.legend(bbox_to_anchor = (1.22,1.0),fontsize = 14)
    
    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 16)
    ax.set_ylabel('Displacement (m)',fontsize = 16)
    
    #### Set fig size, borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.15)
    
    #### Set save path
    save_path = "{}/ANN_Mult//traintestcumsum//{} {}//".format(data_path,subsurface_data.name.values[0],int(subsurface_data.id.values[0]))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/m {} out {} {} smooth {}.png'.format(save_path,int(train_x.shape[1]),out,ave_window,smooth_window),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w',bbox_inches = 'tight')
    
    #### Compute the rmse
    rmse = np.sqrt(np.sum(np.power(test_y - pred_test,2))/float(len(test_y)))
    
    return rmse


def GetTrainTestPlotResultUsePred(subsurface_data,clf,train_x,train_y,test_x,test_y,ave_window = '4H',smooth_window = 12):
    '''
    Gets the superimposed plots of the actual and predicted training and test data via the trained ANN.
    Computes the rmse error of the classifier.
    
    Parameters
    -------------------
    subsurface_data - pd.DataFrame()
        Subsurface data frame
    clf - sklearn classifier
        Trained ANN
    train_x - np.array
        Training feature vectors
    train_y - np.array
        Training label vectors
    test_x - np.array
        Test feature vectors
    test_y - np.array
        Test label vectors
    ave_window - int
        Moving average window
    
    Returns
    -------------------
    test_rmse - root mean square error of the classifier
    '''
    
    #### Get data predictions of ANN
    pred_train = clf.predict(train_x)
    pred_test = clf.predict(test_x)
    
    
    
    #### Get starting indexes for plotting
    start_index = len(subsurface_data) - (len(train_y) + len(test_y))
    
    ############################################
    #### Plot displacement data predictions ####
    ############################################
    
    #### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot measured data points
    ax.plot(subsurface_data.ts.values[start_index:],np.concatenate((train_y,test_y)),label = 'Measured',lw = 3.0)
    
    #### Plot predicted data points
    ax.plot(subsurface_data.ts.values[start_index:],np.concatenate((pred_train,pred_test)),lw = 1.5,label = 'Predicted')
    
    #### Plot predicted boundary
    ax.axvline(subsurface_data.ts.values[start_index + len(train_y)],ls = '--',color = tableau20[14])
    
    #### Plot legend
    ax.legend(bbox_to_anchor = (1.22,1.0),fontsize = 14)
    
    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 16)
    ax.set_ylabel('Displacement (m)',fontsize = 16)
    
    #### Set fig size, borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.15)
    
    #### Set save path
    save_path = "{}/ANN2//traintest//{} {}//".format(data_path,subsurface_data.name.values[0],int(subsurface_data.id.values[0]))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/m {} {} smooth {}.png'.format(save_path,int(train_x.shape[1]),ave_window,smooth_window),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w',bbox_inches = 'tight')
    
    #######################################################
    #### Plot cumulative displacement data predictions ####
    #######################################################
    
    #### Get smoothened and resampled displacement data
    subsurface_data_xz = subsurface_data[['ts','xz']].resample(ave_window,label = 'right',closed = 'right',on = 'ts').mean().rolling(smooth_window).mean()
    
    #### Get displacement column
    subsurface_data_xz['disp'] = subsurface_data_xz.xz - subsurface_data_xz.xz.shift(1)
    displacement = subsurface_data_xz.disp.dropna().values

    #### Denormalize quantities
    dn_actual = np.concatenate((train_y,test_y))*(max(displacement) - min(displacement)) + min(displacement)
    dn_pred = np.concatenate((pred_train,pred_test))*(max(displacement) - min(displacement)) + min(displacement)
    
    #### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot measured data points
    ax.plot(subsurface_data.ts.values[start_index:],np.cumsum(dn_actual),label = 'Measured',lw = 3.0)
    
    #### Plot predicted data points
    ax.plot(subsurface_data.ts.values[start_index:],np.cumsum(dn_pred),lw = 1.5,label = 'Predicted')
    
    #### Plot predicted boundary
    ax.axvline(subsurface_data.ts.values[start_index + len(train_y)],ls = '--',color = tableau20[14])
    
    #### Plot legend
    ax.legend(bbox_to_anchor = (1.22,1.0),fontsize = 14)
    
    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 16)
    ax.set_ylabel('Displacement (m)',fontsize = 16)
    
    #### Set fig size, borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.15)
    
    #### Set save path
    save_path = "{}/ANN2//traintestcumsum//{} {}//".format(data_path,subsurface_data.name.values[0],int(subsurface_data.id.values[0]))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/m {} {} smooth {}.png'.format(save_path,int(train_x.shape[1]),ave_window,smooth_window),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w',bbox_inches = 'tight')
    
    #### Compute the rmse
    rmse = np.sqrt(np.sum(np.power(test_y - pred_test,2))/float(len(test_y)))
    
    return rmse


def PlotRMSE(rmse_csv_file):
    '''
    Reads the rmse results csv file and plots RMSE vs m and Epochs vs m
    
    Parameter
    ------------------
    rmse_csv_file - string
        Filename of the rmse csv file
    
    Returns
    -------------------
    None
        Plotting function only
    '''
    
    #### Read csv file
    results = pd.read_csv(rmse_csv_file)
    
    ########################
    #### Plot RMSE vs m ####
    ########################
    
    #### Produce bar plot
    ax = results.plot.bar(x = ['m'],y = ['RMSE'],color = dyna_colors[1])
    
    #### Get current figure
    fig = plt.gcf()
    
    #### Set y labels and figure title
    ax.set_ylabel('Root Mean Square Error',fontsize = 19)
    ax.set_xlabel('Embedding Dimension',fontsize = 19)
    
    #### Set tick label size
    ax.tick_params(labelsize = 15)
    
    #### Set tick orientation
    plt.xticks(rotation = 0)
    
    #### Remove frame from legend
    ax.legend().set_visible(False)
    
    #### Set fig size
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    
    #### Set save path
    save_path = "{}/ANN//RMSE//".format(data_path)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/{}.png'.format(save_path,rmse_csv_file[:-4]),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w',bbox_inches = 'tight')
    
    #########################
    #### Plot epoch vs m ####
    #########################
    
    #### Produce bar plot
    ax = results.plot.bar(x = ['m'],y = ['epoch'],color = tableau20[2])
    
    #### Get current figure
    fig = plt.gcf()
    
    #### Set y labels and figure title
    ax.set_ylabel('Epochs',fontsize = 19)
    ax.set_xlabel('Embedding Dimension',fontsize = 19)
    
    #### Set tick label size
    ax.tick_params(labelsize = 15)
    
    #### Set tick orientation
    plt.xticks(rotation = 0)
    
    #### Remove frame from legend
    ax.legend().set_visible(False)
    
    #### Set fig size
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    
    #### Set save path
    save_path = "{}/ANN//Epoch//".format(data_path)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/{}.png'.format(save_path,rmse_csv_file[:-4]),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w',bbox_inches = 'tight')
    
    
###############################################################################

if __name__ == '__main__':
    
############################### SINGLE OUTPUT ###############################
    
    subsurface_data = GetSubsurfaceData('2016-08-01', '2017-08-01', 'magta', 15)
    subsurface_data = GetDisplacementPlots(subsurface_data)
    resample = '4H'
    smooth = 12
    train_x, train_y, test_x, test_y = GetFeatureVectors2(subsurface_data, 3,
                                                         train_ratio=0.75,
                                                         ave_window = resample,
                                                         smooth_window = smooth)
    max_clf = GetOptimalCLF2(train_x,train_y,rand_starts = 8)
    rmse = GetTrainTestPlotResult2(subsurface_data, max_clf, train_x,
                                  train_y, test_x, test_y, smooth_window = smooth,
                                  ave_window = resample)

############################### MULTIPLE OUTPUT ###############################

#    subsurface_data = GetSubsurfaceData('2016-08-01', '2017-07-01', 'magta', 15)
#    subsurface_data = GetDisplacementPlots(subsurface_data)
#    resample = '12H'
#    smooth = 12
#    out = 2
#    train_x, train_y, test_x, test_y = GetFeatureVectorsMultOut(subsurface_data,
#                                                         30, out=out,
#                                                         train_ratio=0.75,
#                                                         ave_window = resample,
#                                                         smooth_window = smooth)
#    max_clf = GetOptimalCLF2(train_x,train_y,rand_starts = 8)
#    rmse = GetTrainTestPlotResultMultOut(subsurface_data, max_clf, train_x,
#                                  train_y, test_x, test_y,
#                                  out=out, smooth_window = smooth,
#                                  ave_window = resample)
    
################################ USE PREDICTED ################################

#    subsurface_data = GetSubsurfaceData('2016-08-01', '2017-07-01', 'magta', 15)
#    subsurface_data = GetDisplacementPlots(subsurface_data)
#    resample = '4H'
#    smooth = 12
#    train_x, train_y, test_x, test_y = GetFeatureVectors2(subsurface_data, 3,
#                                                         train_ratio=0.75,
#                                                         ave_window = resample,
#                                                         smooth_window = smooth)
#    max_clf = GetOptimalCLF2(train_x,train_y,rand_starts = 8)
#    rmse = GetTrainTestPlotResultUsePred(subsurface_data, max_clf, train_x,
#                                  train_y, test_x, test_y, smooth_window = smooth,
#                                  ave_window = resample)