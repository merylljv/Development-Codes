# -*- coding: utf-8 -*-
"""
Created on Thu Oct 05 11:02:48 2017

@author: Win8
"""

import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta
from scipy.interpolate import UnivariateSpline
from scipy.signal import gaussian
from scipy.ndimage import filters
from sqlalchemy import create_engine
from sklearn import metrics

### Include Analysis folder of updews-pycodes (HARD CODED!!)

path = os.path.abspath("C:\Users\Win8\Documents\Dynaslope\Data Analysis\updews-pycodes\Analysis")
if not path in sys.path:
    sys.path.insert(1,path)
del path 

import querySenslopeDb as q

#### Global parameters

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)   

data_path = os.path.dirname(os.path.realpath(__file__))

####

def moving_average(series,sigma = 3):
    b = gaussian(39,sigma)
    average = filters.convolve1d(series,b/b.sum())
    var = filters.convolve1d(np.power(series-average,2),b/b.sum())
    return average,var

def students_t(confidence_level,n):
    #### Computes for the critical t-value from two-tailed student's t-distribution 
    #### given the desired confidence level
    gamma = 1 - confidence_level
    return stats.t.ppf(1-gamma/2.,n)

def nonrepeat_colors(ax,NUM_COLORS,color='gist_rainbow'):
    cm = plt.get_cmap(color)
    ax.set_color_cycle([cm(1.*(NUM_COLORS-i-1)/NUM_COLORS) for i in range(NUM_COLORS)[::-1]])
    return ax


def GetAllMarkerData():
    """
    Function to obtain all marker data from the database
    
    Parameters
    ------------------
    None
    
    Returns
    -------------------
    all_marker_data: pd.DataFrame()
    All marker data from the database
    """
    
    #### Write query
    query = "SELECT timestamp, site_id, crack_id, meas FROM gndmeas WHERE timestamp >= '2014-01-01' ORDER by timestamp desc"
    
    #### Get data from database
    df = q.GetDBDataFrame(query)
    
    return df

def ComputeKinematicParameters(marker_data):
    """
    Computes the displacement, time interval, and velocity from the given marker data

    
    Parameters
    ------------------
    marker_data: pd.DataFrame()
        Marker measurement data frame for specific site and crack with the following columns [timestamp,site_id,crack_id,meas]
        
    Returns
    -------------------
    marker_data: pd.DataFrame()
        Marker measurement data with computed values added columns [displacement,time_interval,velocity]
        
    """
    
    #### Sort values per timestamp ascending
    marker_data = marker_data.sort_values('timestamp',ascending = True)
    
    #### Compute for displacement
    marker_data['displacement'] = np.abs(marker_data['meas'] - marker_data['meas'].shift())
    
    #### Compute for the time interval in hours
    marker_data['time_interval'] = (marker_data['timestamp'] - marker_data['timestamp'].shift())/np.timedelta64(1,'h')
    
    #### Compute for velocity in cm/hr
    marker_data['velocity'] = marker_data['displacement']/marker_data['time_interval']
    
    return marker_data

def GenerateKinematicsData():
    """
    Generates the surficial ground marker kinematics, computes for displacement, velocity and time interval

    
    Parameters
    ------------------
    None
        
    Returns
    -------------------
    marker_kinematics: pd.DataFrame() with additional columns ['displacement','velocity','time_interval']
        Generates the marker kinematics dataframe
        
    """
    
    #### Get marker data from database
    marker_data = GetAllMarkerData()
    
    #### Upper case the site codes
    marker_data['site_id'] = map(lambda x:str(x).upper(),marker_data['site_id'])
    
    #### Title case the marker names
    marker_data['crack_id'] = map(lambda x:str(x).title(),marker_data['crack_id'])
    
    #### Group marker data by site and marker name
    marker_data_grouped = marker_data.groupby(['site_id','crack_id'],as_index = False)
    
    #### Compute for the kinematic parameters of the marker data
    marker_data = marker_data_grouped.apply(ComputeKinematicParameters).reset_index()[['timestamp','site_id','crack_id','meas','displacement','time_interval','velocity']]
    
    #### Fill all nan values as zero
    marker_data.fillna(0,inplace = True)
    
    #### Filter values with timestamp less than year 2014
    marker_data = marker_data[marker_data.timestamp >= '2014-01-01']
    
    return marker_data

def SuccededWithinTime(marker_data,time_within):
    """
    Returns the marker data with succeding measurement made within the specified time

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame() with additional columns ['displacement','velocity','time_interval']
        Marker data with computed kinematics grouped by per site and marker name
        
    Returns
    -------------------
    marker_data_with_succeded_acceleration
        Generates the marker kinematics csv file on the current folder
        
    """
    marker_data_within = marker_data[np.logical_and(np.logical_and(marker_data.shift(-1).time_interval <= time_within,marker_data.shift(-1).site_id == marker_data.site_id),marker_data.shift(-1).crack_id == marker_data.crack_id)]
    
    return marker_data_within
    
def PrecededAcceleration(marker_data):
    """
    Returns the marker data preceding an increase in velocity with non zero displacement

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame() with additional columns ['displacement','velocity','time_interval']
        Marker data with computed kinematics grouped by per site and marker name
        
    Returns
    -------------------
    marker_data_with_succeded_acceleration
        Generates the marker kinematics csv file on the current folder
        
    """
    
    #### Get marker data that preceded an increase in velocity
    marker_data_preceding_acceleration = marker_data[marker_data.velocity.values <= marker_data.shift(-1).velocity.fillna(0).values]
    
    #### Get only non-zero velocity values    
    marker_data_preceding_acceleration = marker_data_preceding_acceleration[marker_data_preceding_acceleration.velocity != 0]
        
    return marker_data_preceding_acceleration

def PrecededAccelerationWithTime(marker_data,time_within):
    """
    Returns the marker data preceding an increase in velocity with non zero displacement with the next measurement done within the specified time

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame() with additional columns ['displacement','velocity','time_interval']
        Marker data with computed kinematics grouped by per site and marker name
    time_within: float
        Time in hours in which the next measurement is made
        
    Returns
    -------------------
    marker_data_with_succeded_acceleration
        Generates the marker kinematics csv file on the current folder
        
    """
    #### Get marker data that preceded an increase in velocity
    marker_data_preceding_acceleration = marker_data[np.logical_and(marker_data.velocity.values <= marker_data.shift(-1).velocity.fillna(0).values,marker_data.shift(-1).time_interval <= time_within)]
    
    #### Get only non-zero velocity values    
    marker_data_preceding_acceleration = marker_data_preceding_acceleration[marker_data_preceding_acceleration.velocity != 0]
        
    return marker_data_preceding_acceleration
    
def PrecededDecelerationWithTime(marker_data,time_within):
    """
    Returns the marker data preceding a decrease in velocity with non zero displacement with the next measurement done within the specified time

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame() with additional columns ['displacement','velocity','time_interval']
        Marker data with computed kinematics grouped by per site and marker name
    time_within: float
        Time in hours in which the next measurement is made
        
    Returns
    -------------------
    marker_data_with_succeded_acceleration
        Generates the marker kinematics csv file on the current folder
        
    """
    #### Get marker data that preceded a decrease in velocity
    marker_data_preceding_acceleration = marker_data[np.logical_and(marker_data.velocity.values > marker_data.shift(-1).velocity.fillna(0).values,marker_data.shift(-1).time_interval <= time_within)]
    
    #### Get only non-zero velocity values    
    marker_data_preceding_acceleration = marker_data_preceding_acceleration[marker_data_preceding_acceleration.velocity != 0]
    
    return marker_data_preceding_acceleration

def MarkTrueConditions(marker_kinematics,time_within):
    """
    Returns the marker kinematics data with corresponding true condition based on the definition of L2 success

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame() with additional columns ['displacement','velocity','time_interval']
        Marker data with computed kinematics grouped by per site and marker name
    time_within: float
        Time in hours in which the next measurement is made
        
    Returns
    -------------------
    marker_kinematics_with_condition: pd.DataFrame() with additional columns ['condition'], 1 for positive, -1 for negative
        
    """
    #### Positives are the marker data that preceded an acceleration with succeding data taken within the time specified
    marker_kinematics['condition'] = np.logical_and(marker_kinematics.velocity.values <= marker_kinematics.shift(-1).velocity.fillna(0).values,marker_kinematics.shift(-1).time_interval <= time_within)*2 - 1
    marker_kinematics_with_condition = marker_kinematics
    
    return marker_kinematics_with_condition
    
def MarkVelocityThresholdPredictions(marker_kinematics_with_condition,velocity_threshold):
    """
    Returns the marker kinematics data with corresponding true condition based on the definition of L2 success

    
    Parameters
    ------------------
    marker_kinematics_with_condition: pd.DataFrame() 
        Computed kinematic data marked with corresponding condition
    velocity_threshold: float
        Threshold velocity in cm/hr, used as predicting the actual condition
    time_within: float
        Time in hours in which the next measurement is made

        
    Returns
    -------------------
    marker_kinematics_with_prediction: pd.DataFrame() with additional column ['prediction'], 1 for positive, -1 for negative
        
    """
    #### Positive predictions are those that exceed the velocity threhsold
    marker_kinematics_with_condition['prediction'] = np.array(marker_kinematics_with_condition.velocity.values >= velocity_threshold)*2 - 1
    
    marker_kinematics_with_prediction = marker_kinematics_with_condition
    
    return marker_kinematics_with_prediction

#def MarkOOAThresholdPredictions(marker_kinematics_with_condition,confidence_level,time_within):
#    

def ComputePredictionRates(marker_kinematics_with_prediction):
    """
    Returns the marker kinematics data with corresponding true condition based on the definition of L2 success

    
    Parameters
    ------------------
    marker_kinematics_with_prediction

        
    Returns
    -------------------
    prediction_rates: dict with additional keys ['TPR','FNR','FPR',TNR']
        
    """
    
    #### Get classifier dataframe, 2 = True Positive, 0 = False Negative, 1 = False positive, -1 = True negative
    classifier = 0.5*(marker_kinematics_with_prediction.condition.values + 1) + marker_kinematics_with_prediction.prediction.values
    
    #### Get total conditional positive
    conditional_positive = float(len(marker_kinematics_with_prediction[marker_kinematics_with_prediction.condition.values == 1]))
    
    #### Get total conditional negative
    conditional_negative = float(len(marker_kinematics_with_prediction[marker_kinematics_with_prediction.condition.values == -1]))
    
    #### Get number of true positives, false positives, false negatives, and true negatives
    tp = float(len(classifier[classifier == 2]))
    fn = float(len(classifier[classifier == 0]))
    fp = float(len(classifier[classifier == 1]))
    tn = float(len(classifier[classifier == -1]))
    
    #### Get corresponding rates
    tpr = tp/conditional_positive
    fpr = fp/conditional_negative
    fnr = fn/conditional_positive
    tnr = tn/conditional_negative
    
    #### Print results
    print "Conditional Positive: {}".format(conditional_positive)
    print "Conditional Negative: {}".format(conditional_negative)
    print "-----------------------------------\n"
    print "True Positive Rate: {}".format(tpr)
    print "True Negative Rate: {}".format(tnr)
    print "False Positive Rate: {}".format(fpr)
    print "False Negative Rate: {}\n\n".format(fnr)
    
    return {'tpr':tpr,'fpr':fpr,'fnr':fnr,'tnr':tnr}

def PlotROC(marker_kinematics,time_within,velocity_threshold_range):
    """
    Plots the ROC using the given velocity threshold range with the kinematics data frame and specified time interval

    
    Parameters
    ------------------


        
    Returns
    -------------------

        
    """
    
    #### Get marker_condition dataframe
    
    #### Group based on site_id and crack_id
    marker_kinematics_grouped = marker_kinematics.groupby(['site_id','crack_id'],as_index = False)
    marker_kinematics_with_condition = marker_kinematics_grouped.apply(MarkTrueConditions,time_within).reset_index()
    
    #### Initialize the results container
    true_positive_rates = []
    false_positive_rates = []
    
    #### Iterate all the velocity threshold range
    for velocity_threshold in velocity_threshold_range:
        
        #### Get marker kinematics with prediction data frame
        marker_kinematics_with_prediction = MarkVelocityThresholdPredictions(marker_kinematics_with_condition,velocity_threshold)
        
        #### Get only values with succeding measurement made within 72 hours 
        marker_kinematics_with_prediction = SuccededWithinTime(marker_kinematics_with_prediction,time_within)
        
        #### Get only non-zero values
        marker_kinematics_with_prediction = marker_kinematics_with_prediction[marker_kinematics_with_prediction.displacement != 0]
        
        #### Print current threshold
        print "Threshold: {}".format(round(velocity_threshold,4))
        
        #### Get prediction rates
        prediction_rates = ComputePredictionRates(marker_kinematics_with_prediction)
        
        #### Store result to container
        true_positive_rates.append(prediction_rates['tpr'])
        false_positive_rates.append(prediction_rates['fpr'])
    
    #### Plot results
    
    #### Initialize figure, open the subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot the false positive rate vs true positive rate
    ax.plot(false_positive_rates,true_positive_rates,label = 'Velocity Threshold')
    
    #### Plot the guess line
    ax.plot(np.arange(0,1,0.0001),np.arange(0,1,0.0001),'--',color = tableau20[6],label = 'Random Guess')
    
    #### Plot the legend
    plt.legend(loc = 'upper right')
    
    #### Set axis labels
    ax.set_xlabel('False Positive Rates',fontsize = 14)
    ax.set_ylabel('True Positive Rates',fontsize = 14)
    
    #### Set figure label
    fig.suptitle('ROC Plot for Velocity Thresholds',fontsize = 15)
    
    #### Set figsize
    fig.set_figheight(8)
    fig.set_figwidth(8)
    
    #### Set aspect as equal
    ax.set_aspect('equal')
    
    #### Save fig
    plt.savefig('ROC Velocity Threshold Min {} Max {} N {}.png'.format(round(min(velocity_threshold_range),4),round(max(velocity_threshold_range),4),len(velocity_threshold_range)),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')


def ComputeSplineVelAccel(marker_data,num_pts):
    """
    Computes and saves the spline velocity and acceleration from the marker kinematics data grouped by site and marker name

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame()
        Data frame containing the marker data with computed velocities displacement and time interval

    Returns
    -------------------
    marker_data_with_spline: pd.DataFrame() with additional columns ['sp_velocity','sp_acceleration']
        Marker kinematics data frame with computed spline velocity and acceleration
        
    """
    #### Initialize results containers
    sp_velocity = np.array([])
    sp_acceleration = np.array([])
    
    #### Change index to timestamp
    marker_data.set_index(['timestamp'],inplace = True)
    
    #### Iterate values of the data frame
    for i in range(len(marker_data)):
        
        #### Splice data frame
        cur_data_frame = marker_data.ix[:len(marker_data)-i]
        
        #### Get number of points to be used in spline computation
        cur_data_frame = cur_data_frame.tail(num_pts)
        
        #### Get time elapsed based on time_interval
        cur_data_frame['time'] = cur_data_frame['time_interval'].cumsum()
        
        print cur_data_frame
        
        #### Get time and displacement parameters for spline computation
        time = cur_data_frame.time.values
        meas = cur_data_frame.meas.values
            
            #### Commence interpolation
        try:
            #### Take the gaussian average of data points and its variance
            _,var = moving_average(meas)
            sp = UnivariateSpline(time,meas,w=1/np.sqrt(var))
            
            #### Spline interpolation values    
            vel_int = sp.derivative(n=1)(time[-1])
            acc_int = sp.derivative(n=2)(time[-1])

        except:
            print "Interpolation error {} {}".format(marker_data.site_id.values[0],marker_data.crack_id.values[0])
            vel_int = 0
            acc_int = 0
        
        #### Insert results to array
        sp_velocity = np.insert(sp_velocity,0,abs(vel_int))
        sp_acceleration = np.insert(sp_acceleration,0,abs(acc_int))
    
    marker_data['sp_velocity'] = sp_velocity
    marker_data['sp_acceleration'] = sp_acceleration
    
    return marker_data
        

def MarkerSplineComputations(marker_kinematics,num_pts = 10):
    """
    Computes and saves the spline velocity and acceleration from the marker kinematics data

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame()
        Data frame containing the marker data with computed velocities displacement and time interval
    num_pts: int (default value = 10)
        Number of points to consider in spline computation
        
    Returns
    -------------------
    marker_kinematics_with_spline_computations: pd.DataFrame() with additional columns ['sp_velocity','sp_acceleration']
        Marker kinematics data frame with computed spline velocity and acceleration
        
    """
    #### Group marker data by site and marker name
    marker_kinematics_grouped = marker_kinematics.groupby(['site_id','crack_id'],as_index = False)
    
    #### Compute for the kinematic parameters of the marker data
    marker_kinematics_with_spline_computation = marker_kinematics_grouped.apply(ComputeSplineVelAccel,num_pts).reset_index()[['timestamp','site_id','crack_id','meas','displacement','time_interval','velocity','sp_velocity','sp_acceleration']]
    
    return marker_kinematics_with_spline_computation
    
def ComputeSplinePredictionRates(marker_kinematics_with_spline_computation,confidence_level):
    """
    Computes and saves the spline velocity and acceleration from the marker kinematics data

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame()
        Data frame containing the marker data with computed velocities displacement and time interval
    num_pts: int (default value = 10)
        Number of points to consider in spline computation
        
    Returns
    -------------------
    marker_kinematics_with_spline_predictions: pd.DataFrame() with additional columns ['prediction']
        Marker kinematics data frame with computed spline velocity and acceleration
        
    """
    
    #### Get spline velocity and acceleration
    sp_velocity = marker_kinematics_with_spline_computation.sp_velocity.values
    sp_acceleration = marker_kinematics_with_spline_computation.sp_acceleration.values
    
    #### Generate all constants
    slope = 1.49905955613175
    intercept = -3.00263765777028
    var_v_log = 215.515369339559
    v_log_mean = 2.232839766
    sum_res_square = 49.8880017417971
    n = 30.    
    t_crit = students_t(confidence_level,n-2)
    
    #### Compute delta of confidence interval
    confidence_width = t_crit*np.sqrt(1/(n-2)*sum_res_square*(1. + 1/n + (np.log(sp_velocity) - v_log_mean)**2/var_v_log))
    
    #### Compute threshold line acceleration
    log_acceleration_threshold = slope * np.log(sp_velocity) + intercept
    
    #### Compute upper and lower acceleration
    acceleration_upper_bound = np.exp(log_acceleration_threshold + confidence_width)
    acceleration_lower_bound = np.exp(log_acceleration_threshold - confidence_width)
    
    #### Compute for the exceedances to threshold
    marker_kinematics_with_spline_computation['prediction'] = np.logical_and(sp_acceleration <= acceleration_upper_bound,sp_acceleration >= acceleration_lower_bound)*2 - 1
    
    return marker_kinematics_with_spline_computation

def PlotROCSplinePrediction(marker_kinematics,time_within,confidence_threshold_range):
    """
    Plots the ROC using the given confidence threshold range with the kinematics data frame and specified time interval

    
    Parameters
    ------------------


        
    Returns
    -------------------

        
    """
    
    #### Get marker_condition dataframe
    
    #### Group based on site_id and crack_id
    marker_kinematics_grouped = marker_kinematics.groupby(['site_id','crack_id'],as_index = False)
    marker_kinematics_with_condition = marker_kinematics_grouped.apply(MarkTrueConditions,time_within).reset_index()
    
#    #### Get spline computations
#    marker_kinematics_with_condition = MarkerSplineComputations(marker_kinematics_with_condition)
    
    #### Initialize the results container
    true_positive_rates = []
    false_positive_rates = []
    
    #### Iterate all the velocity threshold range
    for confidence_level in confidence_threshold_range:
        
        #### Get marker kinematics with prediction data frame
        marker_kinematics_with_prediction = ComputeSplinePredictionRates(marker_kinematics_with_condition,confidence_level)
        
        #### Get only values with succeding measurement made within 72 hours 
        marker_kinematics_with_prediction = SuccededWithinTime(marker_kinematics_with_prediction,time_within)
        
        #### Get only non-zero values
        marker_kinematics_with_prediction = marker_kinematics_with_prediction[marker_kinematics_with_prediction.displacement != 0]
        
        #### Print current threshold
        print "Threshold: {}".format(round(confidence_level,4))
        
        #### Get prediction rates
        prediction_rates = ComputePredictionRates(marker_kinematics_with_prediction)
        
        #### Store result to container
        true_positive_rates.append(prediction_rates['tpr'])
        false_positive_rates.append(prediction_rates['fpr'])
    
    #### Plot results
    
    #### Initialize figure, open the subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot the false positive rate vs true positive rate
    ax.plot(false_positive_rates,true_positive_rates,label = 'OOA Filter')
    
    #### Plot the guess line
    ax.plot(np.arange(0,1,0.0001),np.arange(0,1,0.0001),'--',color = tableau20[6],label = 'Random Guess')
    
    #### Plot the legend
    plt.legend(loc = 'upper right')
    
    #### Set axis labels
    ax.set_xlabel('False Positive Rates',fontsize = 14)
    ax.set_ylabel('True Positive Rates',fontsize = 14)
    
    #### Set figure label
    fig.suptitle('ROC Plot for OOA Filter',fontsize = 15)
    
    #### Set figsize
    fig.set_figheight(8)
    fig.set_figwidth(8)
    
    #### Set aspect as equal
    ax.set_aspect('equal')
    
    #### Save fig
    plt.savefig('ROC OOA Threshold Min {} Max {} N {}.png'.format(round(min(confidence_threshold_range),4),round(max(confidence_threshold_range),4),len(confidence_threshold_range)),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
def PlotROCArbitraryParameter(marker_kinematics,time_within,parameter_name):
    """
    Plots the ROC of a given parameter in the marker_kinematics data

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame() 
        Marker kinematics data frame
    time_within: float
        Time in hours in which the next measurement is made
    parameter_name: string
        name of the parameter whose ROC is to be plotted
        
        
    Returns
    -------------------
    None: Plots ROC and saves to ROC folder
        
    """
    
    #### Get marker_condition dataframe
    
    #### Group based on site_id and crack_id
    marker_kinematics_grouped = marker_kinematics.groupby(['site_id','crack_id'],as_index = False)
    marker_kinematics_with_condition = marker_kinematics_grouped.apply(MarkTrueConditions,time_within).reset_index()
    
    #### Get only values taken within specified time_within
    marker_kinematics_with_condition = SuccededWithinTime(marker_kinematics_with_condition,time_within)
    
    #### Get only non-zero displacement values
    marker_kinematics_with_condition = marker_kinematics_with_condition[marker_kinematics_with_condition.displacement != 0]
    
    #### Get false positive rates and true positive rates and corresponding thresholds
    fpr, tpr, thresholds = metrics.roc_curve(marker_kinematics_with_condition.condition.values,marker_kinematics_with_condition[parameter_name].values)
    
    #### Plot results
    
    #### Initialize figure, open subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot fpr vs tpr
    ax.plot(fpr,tpr,label = '{} Threshold'.format(parameter_name.title()))
    
    #### Plot random guess line
    ax.plot(np.arange(0,1,0.0001),np.arange(0,1,0.0001),'--',color = tableau20[6],label = 'Random Guess')
    
    #### Plot the legend
    plt.legend(loc = 'upper right')
    
    #### Set axis labels
    ax.set_xlabel('False Positive Rates',fontsize = 14)
    ax.set_ylabel('True Positive Rates',fontsize = 14)
    
    #### Set figure label
    fig.suptitle('ROC Plot for {} Threshold'.format(parameter_name.title()),fontsize = 15)
    
    #### Set figsize
    fig.set_figheight(8)
    fig.set_figwidth(8)
    
    #### Set aspect as equal
    ax.set_aspect('equal')
    
    #### Save fig
    plt.savefig('{}\ROC\{} Threshold.png'.format(data_path,parameter_name.title()),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
def PlotROCSplinePredictionWithParameterThreshold(marker_kinematics,time_within,confidence_threshold_range,parameter_name,parameter_threshold_range):
    """
    Plots the ROC using the given confidence threshold range with the kinematics data frame and specified time interval combined with a parameter threshold cut off

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame()
        Marker kinematics with spline computation (use MarkerSplineComputations first)
    time_within: float
        Time in hours in which the next measurement should be made
    confidence_threshold_range: np.array
        Value range for the confidence level threshold
    parameter_name: string
        Column in the marker_kinematics data frame to be use as threshold
    parameter_threshold_range: np.array
        Value range for the parameter threshold
        
    Returns
    -------------------
    Plots the ROC of OOA Spline Filter with cut-off parameter threshold
        
    """
    
    #### Get marker_condition dataframe
    
    #### Group based on site_id and crack_id
    marker_kinematics_grouped = marker_kinematics.groupby(['site_id','crack_id'],as_index = False)
    marker_kinematics_with_condition = marker_kinematics_grouped.apply(MarkTrueConditions,time_within).reset_index()
    
#    #### Get spline computations
#    marker_kinematics_with_condition = MarkerSplineComputations(marker_kinematics_with_condition)

    
    #### Initialize figure, open the subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Set non-repeating colors
    ax = nonrepeat_colors(ax,len(parameter_threshold_range),color = 'plasma')
    
    #### Plot the guess line
    ax.plot(np.arange(0,1,0.0001),np.arange(0,1,0.0001),'--',color = tableau20[6],label = 'Random Guess')

    
    #### Iterate all dispalcement threshold range
    for parameter_threshold in parameter_threshold_range:
    
        #### Initialize the results container
        true_positive_rates = []
        false_positive_rates = []
    
        #### Iterate all the OOA threshold range
        for confidence_level in confidence_threshold_range:
            
            #### Get marker kinematics with prediction data frame
            marker_kinematics_with_prediction = ComputeSplinePredictionRates(marker_kinematics_with_condition,confidence_level)
            
            #### Get only values with succeding measurement made within 72 hours 
            marker_kinematics_with_prediction = SuccededWithinTime(marker_kinematics_with_prediction,time_within)
            
            #### Get only non-zero values
            marker_kinematics_with_prediction = marker_kinematics_with_prediction[marker_kinematics_with_prediction.displacement != 0]
            
            #### Cut off parameter threshold
            marker_kinematics_with_prediction = marker_kinematics_with_prediction[marker_kinematics_with_prediction[parameter_name] >= parameter_threshold]
            
            #### Print current thresholds
            print "OOA Threshold: {}".format(round(confidence_level,4))
            print "{} Threshold: {}".format(parameter_name.title(),round(parameter_threshold,4))
            
            #### Get prediction rates
            prediction_rates = ComputePredictionRates(marker_kinematics_with_prediction)
            
            #### Store result to container
            true_positive_rates.append(prediction_rates['tpr'])
            false_positive_rates.append(prediction_rates['fpr'])
    
        #### Plot results
        
        #### Plot the false positive rate vs true positive rate
        ax.plot(false_positive_rates,true_positive_rates)
        
    
    #### Set axis labels
    ax.set_xlabel('False Positive Rates',fontsize = 14)
    ax.set_ylabel('True Positive Rates',fontsize = 14)
    
    #### Set figure label
    fig.suptitle('ROC Plot for OOA Filter With {} Cut Off'.format(parameter_name.title()),fontsize = 15)
    
    #### Set figsize
    fig.set_figheight(8)
    fig.set_figwidth(8)
    
    #### Set aspect as equal
    ax.set_aspect('equal')
    
    #### Save fig
    plt.savefig('ROC OOA Threshold Min {} Max {} N {} With {} Min {} Max {} N {}.png'.format(round(min(confidence_threshold_range),4),round(max(confidence_threshold_range),4),len(confidence_threshold_range),parameter_name.title(),round(min(parameter_threshold_range),4),round(max(parameter_threshold_range),4),len(parameter_threshold_range)),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')

def GetPerSiteMarkerDataCount(marker_kinematics,time_within):
    """
    Counts the number of marker data, event data, and positive events per site

    
    Parameters
    ------------------
    marker_kinematics_with_condition: pd.DataFrame()
        Marker kinematics data frame
    time_within: float
        Time in hours in which the next measurement should be made
        
    Returns
    -------------------
    marker_data_count: pd.DataFrame()
        Data rame containing the number of available marker data, event data, and positive events per site
        
    """
    
    #### Initialize results container
    marker_data_count = pd.DataFrame()
    
    #### Group based on site_id and crack_id
    marker_kinematics_grouped = marker_kinematics.groupby(['site_id','crack_id'],as_index = False)
    
    #### Get marker_condition dataframe
    marker_kinematics_with_condition = marker_kinematics_grouped.apply(MarkTrueConditions,time_within).reset_index()
    
    #### Get total marker data available per site store to results container
    marker_data_count['Marker Data'] = marker_kinematics_with_condition.groupby('site_id').agg('count').timestamp

    #### Get event data avaiable per site
    marker_succeded = SuccededWithinTime(marker_kinematics_with_condition,72)
    marker_succeded = marker_succeded[marker_succeded.displacement != 0]
    
    #### Store results
    marker_data_count['Event Data'] = marker_succeded.groupby('site_id').agg('count').timestamp

    #### Get positive events
    marker_data_count['Positives'] = marker_succeded[marker_succeded.condition == 1].groupby('site_id').agg('count').timestamp

    #### Sort by marker data count
    marker_data_count.sort_values('Marker Data',ascending = False,inplace = True)

    #### Fill na values to zero
    marker_data_count.fillna(0,inplace = True)
    
    #### Remove unknown sites
    marker_data_count = marker_data_count[marker_data_count.index != 'BOS']
    marker_data_count = marker_data_count[marker_data_count.index != 'KAN']
    marker_data_count = marker_data_count[marker_data_count.index != 'PHI']
    marker_data_count = marker_data_count[marker_data_count.index != 'EVE']
    marker_data_count = marker_data_count[marker_data_count.index != 'BTO']
    marker_data_count = marker_data_count[marker_data_count.index != '']
                     
    return marker_data_count

def MarkerDataCountPlots(marker_data_count):
    """
    Plots all marker data count bar plots from the marker data count data frame

    
    Parameters
    ------------------
    marker_data_count: pd.DataFrame()
        (Use GetPerSiteMarkerDataCount first)Data rame containing the number of available marker data, event data, and positive events per site
    Returns
    -------------------
    marker_data_count: pd.DataFrame()
        Data rame containing the number of available marker data, event data, and positive events per site
        
    """
    #### Set save path
    save_path = "{}/Data Count".format(data_path)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #################################
    #### Plot Marker Data Count #####
    #################################
    
    ax = marker_data_count.plot.barh(y = ['Marker Data'],stacked = True)
    
    #### Get current figure
    fig = plt.gcf()
    
    #### Set y labels and figure title
    ax.set_ylabel('Site Code',fontsize = 14)
    ax.set_xlabel('Count',fontsize = 14)
    fig.suptitle('Marker Data Count',fontsize = 15)
    
    #### Set tick label size
    ax.tick_params(labelsize = 8)
    
    #### Remove frame from legend
    ax.legend().get_frame().set_visible(False)
    
    #### Set fig size
    fig.set_figheight(7.5)
    fig.set_figwidth(13)
    
    #### Save fig
    plt.savefig('{}//Marker Data Count.png'.format(save_path),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    ### Close fig
    plt.close()
    
    ####################################################    
    #### Plot Marker Data Count with Label #############
    ####################################################
    
    ax = marker_data_count.plot.barh(y = ['Marker Data'],stacked = True)
    
    #### Annotate bars with data value
    for p in ax.patches:
        ax.annotate(str(int(p.get_width())),(p.get_width() + max(marker_data_count['Marker Data'].values)*0.0025, p.get_y() * 1.000),fontsize = 8)    
        
    #### Get current figure
    fig = plt.gcf()
    
    #### Set y labels and figure title
    ax.set_ylabel('Site Code',fontsize = 14)
    ax.set_xlabel('Count',fontsize = 14)
    fig.suptitle('Marker Data Count',fontsize = 15)
    
    #### Set tick label size
    ax.tick_params(labelsize = 8)
    
    #### Remove frame from legend
    ax.legend().get_frame().set_visible(False)
    
    #### Set fig size
    fig.set_figheight(7.5)
    fig.set_figwidth(13)
    
    #### Save fig
    plt.savefig('{}//Marker Data Count With Label.png'.format(save_path),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    #### Close fig
    plt.close()
    
    ##########################################
    #### Plot Marker Data with Event Data ####
    ##########################################
    
    #### Define adjusted marker data container
    marker_data_count_adjusted = marker_data_count[['Marker Data','Event Data','Positives']]
    
    #### Adjust marker_data_count to accomodate event data
    marker_data_count_adjusted['Marker Data'] = marker_data_count_adjusted['Marker Data'] - marker_data_count_adjusted['Event Data']
    
    #### Plot results
    ax = marker_data_count_adjusted.plot.barh(y = ['Marker Data','Event Data'],stacked = True)
    
    #### Get current figure
    fig = plt.gcf()
    
    #### Set y labels and figure title
    ax.set_ylabel('Site Code',fontsize = 14)
    ax.set_xlabel('Count',fontsize = 14)
    fig.suptitle('Marker Data Count',fontsize = 15)
    
    #### Set tick label size
    ax.tick_params(labelsize = 8)
    
    #### Remove frame from legend
    ax.legend().get_frame().set_visible(False)
    
    #### Set fig size
    fig.set_figheight(7.5)
    fig.set_figwidth(13)
    
    #### Save fig
    plt.savefig('{}//Marker Data Event Data Count.png'.format(save_path),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    ### Close fig
    plt.close()
    
    ###############################
    #### Plot Event Data Count ####
    ###############################
                         
    #### Sort values according to event data
    event_data_count = marker_data_count_adjusted.sort_values('Event Data',ascending = False)[['Event Data','Positives']]
    
    #### Plot results
    ax = event_data_count.plot.barh(y = ['Event Data'],stacked = True,color = tableau20[2])
    
    #### Get current figure
    fig = plt.gcf()
    
    #### Set y labels and figure title
    ax.set_ylabel('Site Code',fontsize = 14)
    ax.set_xlabel('Count',fontsize = 14)
    fig.suptitle('Event Data Count',fontsize = 15)
    
    #### Set tick label size
    ax.tick_params(labelsize = 8)
    
    #### Remove frame from legend
    ax.legend().get_frame().set_visible(False)
    
    #### Set fig size
    fig.set_figheight(7.5)
    fig.set_figwidth(13)
    
    #### Save fig
    plt.savefig('{}//Event Data Count.png'.format(save_path),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    ### Close fig
    plt.close()
    
    ##########################################
    #### Plot Event Data Count With Label ####
    ##########################################
    
    #### Sort values according to event data
    event_data_count = marker_data_count_adjusted.sort_values('Event Data',ascending = False)[['Event Data','Positives']]
    
    #### Plot results
    ax = event_data_count.plot.barh(y = ['Event Data'],stacked = True,color = tableau20[2])
    
    #### Annotate values
    for p in ax.patches:
        ax.annotate(str(int(p.get_width())),(p.get_width() +  max(event_data_count['Event Data'].values)*0.0025, p.get_y() * 1.000),fontsize = 8)    
    
    #### Get current figure
    fig = plt.gcf()
    
    #### Set y labels and figure title
    ax.set_ylabel('Site Code',fontsize = 14)
    ax.set_xlabel('Count',fontsize = 14)
    fig.suptitle('Event Data Count',fontsize = 15)
    
    #### Set tick label size
    ax.tick_params(labelsize = 8)
    
    #### Remove frame from legend
    ax.legend().get_frame().set_visible(False)
    
    #### Set fig size
    fig.set_figheight(7.5)
    fig.set_figwidth(13)
    
    #### Save fig
    plt.savefig('{}//Event Data Count With Label.png'.format(save_path),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    ### Close fig
    plt.close()
    
    ##############################################
    #### Plot Event Data Count With Positives ####
    ##############################################
    
    #### Define adjusted event data container
    event_data_count_adjusted = event_data_count[['Event Data','Positives']]
    
    #### Adjust event_data_count to accomodate positives
    event_data_count_adjusted['Event Data'] = event_data_count_adjusted['Event Data'] - event_data_count_adjusted['Positives']
    
    #### Plot results
    ax = event_data_count_adjusted.plot.barh(y = ['Event Data','Positives'],stacked = True,color = (tableau20[2],tableau20[4]))
    
    #### Get current figure
    fig = plt.gcf()
    
    #### Set y labels and figure title
    ax.set_ylabel('Site Code',fontsize = 14)
    ax.set_xlabel('Count',fontsize = 14)
    fig.suptitle('Event Data Count',fontsize = 15)
    
    #### Set tick label size
    ax.tick_params(labelsize = 8)
    
    #### Remove frame from legend
    ax.legend().get_frame().set_visible(False)
    
    #### Set fig size
    fig.set_figheight(7.5)
    fig.set_figwidth(13)
    
    #### Save fig
    plt.savefig('{}//Event Data Positives Count.png'.format(save_path),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    ### Close fig
    plt.close()
    
    ###################################################################
    #### Plot Event Data Count With Positives And Percentage label ####
    ###################################################################
    
    #### Calculate Percentages
    event_data_count['Percentage'] = np.round(event_data_count_adjusted['Positives']/(event_data_count_adjusted['Event Data'] + event_data_count_adjusted['Positives'])*100,2)
    
    #### Fillna values to zero
    event_data_count.fillna(0,inplace = True)
    
    #### Plot results
    ax = event_data_count_adjusted.plot.barh(y = ['Event Data','Positives'],stacked = True,color = (tableau20[2],tableau20[4]))
    
    #### Get current figure
    fig = plt.gcf()
    
    #### Annotate percentage values
    for y,(percentage,width) in enumerate(event_data_count[['Percentage','Event Data']].values):
        ax.annotate('{}%'.format(percentage),(width + max(event_data_count['Event Data'].values)* 0.0025, (y-0.25) * 1.000),fontsize = 7)    
        
    #### Set y labels and figure title
    ax.set_ylabel('Site Code',fontsize = 14)
    ax.set_xlabel('Count',fontsize = 14)
    fig.suptitle('Event Data Count',fontsize = 15)
    
    #### Set tick label size
    ax.tick_params(labelsize = 8)
    
    #### Remove frame from legend
    ax.legend().get_frame().set_visible(False)
    
    #### Set fig size
    fig.set_figheight(7.5)
    fig.set_figwidth(13)
    
    #### Save fig
    plt.savefig('{}//Event Data Positives Count With Percentage Label.png'.format(save_path),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    ### Close fig
    plt.close()












    



                                               
                                               