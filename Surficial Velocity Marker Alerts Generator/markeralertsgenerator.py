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
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from scipy import stats
import matplotlib.dates as md
from datetime import datetime, timedelta
from scipy.interpolate import UnivariateSpline
from scipy.signal import gaussian
from scipy.ndimage import filters
from sqlalchemy import create_engine
from sklearn import metrics
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from scipy.spatial import ConvexHull

### Include Analysis folder of updews-pycodes (HARD CODED!!)

path = os.path.abspath("D:\Leo\Dynaslope\Data Analysis\updews-pycodes\Analysis")
if not path in sys.path:
    sys.path.insert(1,path)
del path 

import querySenslopeDb as q

#### HARD CODED Results
time_bins = [(0,11),(11,33),(33,63),(63,81),(81,111),(111,154),(154,184),(184,720)]
parameters = ['displacement','velocity','sp_velocity','sp_acceleration']

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

#### Color keys
color_key = {'sp_velocity':tableau20[0],'sp_acceleration':tableau20[2],'velocity':tableau20[4],'displacement':tableau20[10]}

#### Title Keys
title_key = {'sp_velocity':'Spline Velocity','sp_acceleration':'Spline Acceleration','velocity':'Velocity','displacement':'Displacement'}

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
    ax.set_color_cycle([cm(1.*(NUM_COLORS-i-1)/NUM_COLORS) for i in np.array(range(NUM_COLORS))[::-1]])
    return ax

def offset_cut_off(optimal_thresholds):
    """
    Offsets cut off values for plotting
    
    Parameters
    -----------------
    optimal_thresholds - pd.DataFrame()
        Optimal thresholds data frame
    
    Returns
    -----------------
    optimal_threshlds - pd.DataFrame() with offset_cut_off
        cut off values with unique offset
    """
    
    #### Get only relevant values
    cut_off_values = optimal_thresholds[['cut_off','parameter','time_bin','cut_off_parameter']]
    
    #### Iterate for all time bin
    for time_bin in np.unique(cut_off_values.time_bin.values):
        
        #### Get current df
        cur_df = cut_off_values[cut_off_values.time_bin == time_bin]
        
        #### Get unique number counts
        counts = cur_df.cut_off.value_counts()
        
        #### Hanlde counts = 2
        for twin_value in counts[counts == 2].index:
            cut_off_values.loc[np.logical_and(cut_off_values.time_bin == time_bin,cut_off_values.cut_off == twin_value),'cut_off'] = cut_off_values.loc[np.logical_and(cut_off_values.time_bin == time_bin,cut_off_values.cut_off == twin_value),'cut_off'] + np.array([-0.01,0.01])
        
        #### Handle counts = 3
        if len(counts[counts == 3]) != 0:
            triplet_value = counts[counts == 3].index[0]
            cut_off_values.loc[np.logical_and(cut_off_values.time_bin == time_bin,cut_off_values.cut_off == triplet_value),'cut_off'] = cut_off_values.loc[np.logical_and(cut_off_values.time_bin == time_bin,cut_off_values.cut_off == triplet_value),'cut_off'] + np.array([-0.02,0.00,0.02])

        #### Handle counts = 4
        if len(counts[counts == 4]) != 0:
            quadriplet_value = counts[counts == 4].index[0]
            cut_off_values.loc[np.logical_and(cut_off_values.time_bin == time_bin,cut_off_values.cut_off == quadriplet_value),'cut_off'] = cut_off_values.loc[np.logical_and(cut_off_values.time_bin == time_bin,cut_off_values.cut_off == quadriplet_value),'cut_off'] + np.array([-0.03,-0.01,0.01,0.03])
        
    return cut_off_values

def GetAllMarkerData(mode = 'MySQL'):
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
    
    if mode == 'MySQL':
        #### Write query
        query = "SELECT timestamp, site_id, crack_id, meas FROM gndmeas WHERE timestamp >= '2014-01-01' ORDER by timestamp desc"
        
        #### Get data from database
        df = q.GetDBDataFrame(query)
    elif mode == 'csv':
        df = pd.read_csv('gndmeas.csv')
    
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

def GenerateKinematicsData(mode = 'MySQL'):
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
    marker_data = GetAllMarkerData(mode)
    
    #### Convert timestamp row into timestamp
    marker_data['timestamp'] = pd.to_datetime(marker_data['timestamp'])
    
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

def MarkCutOffPredictions(marker_kinematics_with_condition,parameter_name,cut_off_value):
    """
    Returns the marker kinematics data with corresponding cut off conditions based on cut off value

    
    Parameters
    ------------------
    marker_kinematics_with_condition: pd.DataFrame() 
        Computed kinematic data marked with corresponding condition
    parameter_name: string
        Name of cut off parameter
    cut_off_value: float
        Cut off value for the parameter

        
    Returns
    -------------------
    marker_kinematics_with_cut_off: pd.DataFrame() with additional column ['cut off'], 1 for positive, -1 for negative
        
    """
    
    #### Positive predictions are those that exceeded the cut off threshold
    marker_kinematics_with_condition['cutoff'] = np.array(marker_kinematics_with_condition[parameter_name].values >= cut_off_value)*2 - 1
    
    return marker_kinematics_with_condition

def CombinePredictionAndCutOff(marker_kinematics_with_condition):
    """
    Returns the marker kinematics data with combined prediction from cutoff and parameter

    
    Parameters
    ------------------
    marker_kinematics_with_condition: pd.DataFrame() 
        Computed kinematic data marked with corresponding condition
        
    Returns
    -------------------
    marker_kinematics_with_cut_off: pd.DataFrame() with combined prediction column ['prediction'], 1 for positive, -1 for negative
        
    """
    
    marker_kinematics_with_condition['prediction'] = 0.5 * (marker_kinematics_with_condition['prediction'].values + 1) * (marker_kinematics_with_condition['cutoff'].values + 1) - 1
    
    return marker_kinematics_with_condition

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
    
    #### Marker spline fill na
    marker_kinematics_with_spline_computation = marker_kinematics_with_spline_computation.fillna(0)
    
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

def PlotROCSplinePrediction(marker_kinematics,time_within,confidence_threshold_range,time_bin = None):
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
        
        #### Get only displacement values with disp <= 100 cm
        marker_kinematics_with_prediction = marker_kinematics_with_prediction[marker_kinematics_with_prediction.displacement <= 100]
            
        #### Get only values at specified time bin
        if time_bin:
            marker_kinematics_with_prediction = marker_kinematics_with_prediction[np.logical_and(marker_kinematics_with_prediction.time_interval >= time_bin[0],marker_kinematics_with_prediction.time_interval <= time_bin[-1])]
        
        #### Print current threshold
        print "Threshold: {}".format(round(confidence_level,4))
        
        #### Get prediction rates
        prediction_rates = ComputePredictionRates(marker_kinematics_with_prediction)
        
        #### Store result to container
        true_positive_rates.append(prediction_rates['tpr'])
        false_positive_rates.append(prediction_rates['fpr'])
    
    #### Compute for AUC ROC
    auc = np.trapz(true_positive_rates,false_positive_rates)
    
    #### Record results to data frame
    roc_results = pd.DataFrame({'tpr':true_positive_rates,'fpr':false_positive_rates,'threshold':confidence_threshold_range})
    
    #### Compute for Youden Index
    roc_results['yi'] = roc_results.tpr - roc_results.fpr
        
    #### Get maximum Youden Index
    max_YI = max(roc_results.yi)
    
    #### Get threshold with maximum YI
    optimized_threshold = roc_results[roc_results.yi == max_YI].threshold.values[-1]
    
    #### Get fpr and tpr of maximum YI
    optimized_fpr = roc_results[roc_results.threshold == optimized_threshold].fpr.values[0]
    optimized_tpr = roc_results[roc_results.threshold == optimized_threshold].tpr.values[0]
    
    
    #### Plot results
    
    #### Initialize figure, open the subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot the guess line
    random_guess_line, = ax.plot(np.arange(0,1,0.0001),np.arange(0,1,0.0001),'--',color = tableau20[6],label = 'Random Guess')

    #### Plot the false positive rate vs true positive rate
    roc_line, = ax.plot(false_positive_rates,true_positive_rates,label = 'OOA Filter')
        
    #### Plot Max YI line
    ax.plot(np.ones(1000)*optimized_fpr,np.linspace(optimized_fpr,optimized_tpr,1000),'--',color = tableau20[16])
    max_yi_line, = ax.plot([optimized_fpr],[optimized_tpr],'o--',color = tableau20[16],label = 'Max YI',markersize = 4)
    
    #### Print AUC ROC result
    ax.text(0.975,0.025,'Area under the ROC = {}\nMax YI = {}, at {:.2e}'.format(round(auc,4),round(max_YI,4),optimized_threshold),transform = ax.transAxes,ha = 'right',fontsize = 12)
    
    #### Print number of data points and time bin
    if time_bin:
        ax.text(0.975,0.025,'Time Bin {} to {}\nPos {} Neg {}'.format(time_bin[0],time_bin[-1],len(marker_kinematics_with_prediction[marker_kinematics_with_prediction.condition == 1]),len(marker_kinematics_with_prediction[marker_kinematics_with_prediction.condition == -1])),transform = fig.transFigure,ha = 'right',fontsize = 10)
    
    #### Plot the legend
    plt.legend(handles = [roc_line,random_guess_line,max_yi_line],loc = 'upper right')
    
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
    
    #### Set save path
    if time_bin:
        save_path = "{}\ROC\\Time Bins\\{} to {}".format(data_path,time_bin[0],time_bin[1])
    else:
        save_path = "{}\ROC\\OOA".format(data_path)
    
    if not os.path.exists(save_path+'\\'):
        os.makedirs(save_path+'\\')    
        
    #### Save fig
    plt.savefig('{}\\ROC OOA Threshold Min {} Max {} N {}.png'.format(save_path,round(min(confidence_threshold_range),4),round(max(confidence_threshold_range),4),len(confidence_threshold_range)),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
def PlotROCArbitraryParameter(marker_kinematics,time_within,parameter_name,time_bin = None,cut_off_parameter_name = None,cut_off_value = None):
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
    
    #### Get only displacement values < 100 cm
    marker_kinematics_with_condition = marker_kinematics_with_condition[marker_kinematics_with_condition.displacement <= 100]
    
    #### Get only specified time bin
    if time_bin:
        marker_kinematics_with_condition = marker_kinematics_with_condition[np.logical_and(marker_kinematics_with_condition.time_interval >= time_bin[0],marker_kinematics_with_condition.time_interval <= time_bin[1])]
    
    #### Cut off filter
    if cut_off_parameter_name or cut_off_value:
        marker_kinematics_with_condition = marker_kinematics_with_condition[marker_kinematics_with_condition[cut_off_parameter_name] >= cut_off_value]

    #### Get false positive rates and true positive rates and corresponding thresholds
    fpr, tpr, thresholds = metrics.roc_curve(marker_kinematics_with_condition.condition.values,marker_kinematics_with_condition[parameter_name].values)
    
    #### Record results to data frame
    roc_results = pd.DataFrame({'tpr':tpr,'fpr':fpr,'threshold':thresholds})
    
    #### Compute for Youden Index
    roc_results['yi'] = tpr - fpr

    #### Get maximum Youden Index
    max_YI = max(roc_results.yi)
    
    #### Get threshold with maximum YI
    optimized_threshold = roc_results[roc_results.yi == max_YI].threshold.values[-1]
    
    #### Get fpr and tpr of maximum YI
    optimized_fpr = roc_results[roc_results.threshold == optimized_threshold].fpr.values[0]
    optimized_tpr = roc_results[roc_results.threshold == optimized_threshold].tpr.values[0]
    
    #### Get AUC ROC value
    auc = metrics.roc_auc_score(marker_kinematics_with_condition.condition.values,marker_kinematics_with_condition[parameter_name].values)

    #### Plot results
    
    #### Initialize figure, open subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
        
    #### Plot random guess line
    random_guess_line, = ax.plot(np.arange(0,1,0.0001),np.arange(0,1,0.0001),'--',color = tableau20[6],label = 'Random Guess')
    
    #### Plot fpr vs tpr
    roc_line, = ax.plot(fpr,tpr,label = '{} Threshold'.format(parameter_name.title()))
    
    #### Plot Max YI line
    ax.plot(np.ones(1000)*optimized_fpr,np.linspace(optimized_fpr,optimized_tpr,1000),'--',color = tableau20[16])
    max_yi_line, = ax.plot([optimized_fpr],[optimized_tpr],'o--',color = tableau20[16],label = 'Max YI',markersize = 4)
    
    #### Print AUC ROC result
    ax.text(0.975,0.025,'Area under the ROC = {}\nMax YI = {}, at {:.2e}\nTPR {:.1f}% FPR {:.1f}%'.format(round(auc,4),round(max_YI,4),optimized_threshold,round(optimized_tpr,4)*100,round(optimized_fpr,4)*100),transform = ax.transAxes,ha = 'right',fontsize = 10)
    
    if time_bin or cut_off_value:
        #### Print number of data points and time bin
        ax.text(0.975,0.025,'Time Bin {} to {}\nPos {} Neg {}\nCut Off {}'.format(time_bin[0],time_bin[1],len(marker_kinematics_with_condition[marker_kinematics_with_condition.condition == 1]),len(marker_kinematics_with_condition[marker_kinematics_with_condition.condition == -1]),cut_off_value),transform = fig.transFigure,ha = 'right',fontsize = 10)
    
    #### Plot the legend
    plt.legend(handles = [roc_line,random_guess_line,max_yi_line],loc = 'upper right')
    
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
  
    #### Set save path
    if not (time_bin or cut_off_parameter_name):
        save_path = "{}\ROC\\Arbitrary Parameter".format(data_path)
    elif time_bin and cut_off_parameter_name:
        save_path = "{}\ROC\\Time Bins\\Cut Off\\{}\\{} to {}\\Optimal".format(data_path,cut_off_parameter_name,time_bin[0],time_bin[1])
    if not os.path.exists(save_path+'\\'):
        os.makedirs(save_path+'\\')    
    
    #### Save fig
    plt.savefig('{}\{} Threshold.png'.format(save_path,parameter_name.title()),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    if time_bin or cut_off_parameter_name:
        return optimized_threshold

def PlotROCArbitraryParameterWithParameterCutOff(marker_kinematics,time_within,parameter_name,cut_off_parameter_name,cut_off_threshold_range,time_bin = None):
    """
    Plots the ROC of a given parameter in the marker_kinematics data along with another parameter along a cut off threshold

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame() 
        Marker kinematics data frame
    time_within: float
        Time in hours in which the next measurement is made
    parameter_name: string
        name of the parameter whose ROC is to be plotted
    cut_off_parameter_name: string
        name of cut off parameter
    cut_off_threshold_range: np.arange
        threshold range for the cut off parameter
        
        
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
    
    #### Get only displacement with <= 100
    marker_kinematics_with_condition = marker_kinematics_with_condition[marker_kinematics_with_condition.displacement <= 100]
    
    #### Initialize figure, open subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot random guess line
    ax.plot(np.arange(0,1,0.0001),np.arange(0,1,0.0001),'--',color = tableau20[6],label = 'Random Guess')

    #### Set non-repeating colors
    ax = nonrepeat_colors(ax,len(cut_off_threshold_range),color = 'plasma')
    
    #### Initialize auc results container
    auc_scores = []
    
    for cut_off_threshold in cut_off_threshold_range:
        
        #### Cut off parameter threshold
        cur_data = marker_kinematics_with_condition[marker_kinematics_with_condition[cut_off_parameter_name] >= cut_off_threshold]
        
        if time_bin:
            #### Cut off time bin
            cur_data = cur_data[np.logical_and(cur_data.time_interval >= time_bin[0],cur_data.time_interval <= time_bin[1])]
        
        #### Get false positive rates and true positive rates and corresponding thresholds
        fpr, tpr, thresholds = metrics.roc_curve(cur_data.condition.values,cur_data[parameter_name].values)
        
        #### Get AUC
        auc = metrics.roc_auc_score(cur_data.condition.values,cur_data[parameter_name].values)
        
        #### Append to results container
        auc_scores.append(auc)
        
        #### Plot results
        
        #### Plot fpr vs tpr
        ax.plot(fpr,tpr)
    
    #### Record AUC results to data frame
    auc_results = pd.DataFrame({'cut_off':cut_off_threshold_range,'auc':auc_scores})
    
    #### Obtain optimal auc and cut off value
    optimal_auc = max(auc_results.auc.values)
    optimal_cut_off = auc_results[auc_results.auc == optimal_auc].cut_off.values[0]
                
    if time_bin:
        #### Print time bin
        ax.text(0.975,0.025,'Time Bin {} to {}'.format(time_bin[0],time_bin[1]),transform = fig.transFigure,ha = 'right',fontsize = 10)    
    
    #### Set axis labels
    ax.set_xlabel('False Positive Rates',fontsize = 14)
    ax.set_ylabel('True Positive Rates',fontsize = 14)
    
    #### Set figure label
    fig.suptitle('ROC Plot for {} Threshold With {} Cut Off'.format(parameter_name.title(),cut_off_parameter_name.title()),fontsize = 15)
    
    #### Set figsize
    fig.set_figheight(8)
    fig.set_figwidth(8)
    
    #### Set aspect as equal
    ax.set_aspect('equal')
    
    #### Set save path
    if time_bin:
        save_path = "{}\ROC\\Time Bins\\Cut Off\\{}\\{} to {}\\ROC".format(data_path,cut_off_parameter_name,time_bin[0],time_bin[1])
    else:
        save_path = "{}\ROC\\Arbitrary Parameter With Cut Off\\ROC".format(data_path)
    
    #### Create save path if not exists
    if not os.path.exists(save_path+'\\'):
        os.makedirs(save_path+'\\')    
    
    #### Save fig
    plt.savefig('{}\{} Threshold With {} Min {} Max {} N {}.png'.format(save_path,parameter_name.title(),cut_off_parameter_name.title(),round(min(cut_off_threshold_range),4),round(max(cut_off_threshold_range),4),len(cut_off_threshold_range)),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    #### Close fig
    plt.close()
    
    ############################################
    #### Plot AUC vs cut off threshold curve ###
    ############################################
    
    #### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot grid
    ax.grid()
    
    #### Plot results
    ax.scatter(cut_off_threshold_range,auc_scores,c = cut_off_threshold_range, cmap = 'plasma',s = 12,zorder = 3)
    
    if time_bin:
        #### Print time bin
        ax.text(0.975,0.025,'Time Bin {} to {}'.format(time_bin[0],time_bin[1]),transform = fig.transFigure,ha = 'right',fontsize = 10)    
    
    #### Print max AUC
    auc_max_text = 'Max AUC = {}, at {:.2e}'.format(round(optimal_auc,4),optimal_cut_off)
    ax.add_artist(AnchoredText(auc_max_text,prop=dict(size=10), frameon=False,loc = 4))
    
    #### Set axis labels
    ax.set_xlabel('{}'.format(cut_off_parameter_name.title()),fontsize = 14)
    ax.set_ylabel('Area under the ROC',fontsize = 14)
    
    #### Set figure label
    fig.suptitle('AUC for {} vs {} Cut Off'.format(parameter_name.title(),cut_off_parameter_name.title()),fontsize = 15)
    
    #### Set figsize
    fig.set_figheight(6)
    fig.set_figwidth(8)

    #### Set save path
    if time_bin:
        save_path = "{}\ROC\\Time Bins\\Cut Off\\{}\\{} to {}\\AUC".format(data_path,cut_off_parameter_name,time_bin[0],time_bin[1])
    else:
        save_path = "{}\ROC\\Arbitrary Parameter With Cut Off\\AUC".format(data_path)
    if not os.path.exists(save_path+'\\'):
        os.makedirs(save_path+'\\')

    #### Save fig
    plt.savefig('{}\AUC of {} vs {} Min {} Max {} N {}.png'.format(save_path,parameter_name.title(),cut_off_parameter_name.title(),round(min(cut_off_threshold_range),4),round(max(cut_off_threshold_range),4),len(cut_off_threshold_range)),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    #### Close fig
    plt.close()

    
    #################################
    ### Plot ROC of optimal value ###
    #################################
    
    optimal_threshold = PlotROCArbitraryParameter(marker_kinematics,time_within,parameter_name,time_bin,cut_off_parameter_name,optimal_cut_off)
    
    #### Close fig
    plt.close()
    
    if time_bin:
        return optimal_auc, optimal_cut_off, optimal_threshold

def PlotROCSplinePredictionWithParameterThreshold(marker_kinematics,time_within,confidence_threshold_range,parameter_name,parameter_threshold_range,mode = 'combine'):
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
    mode: string
        Mode of cut off threshold, may be 'combine' which uses the threshold as another parameter
        or 'define' cut off threshold is incorporated in the definition of alert hence reducing condition cases
        
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
    
    #### Initialize AUC score container
    auc_scores = []
    
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
            
            if mode == 'combine':
                #### Mark cut off predictions
                marker_kinematics_with_prediction = MarkCutOffPredictions(marker_kinematics_with_prediction,parameter_name,parameter_threshold)
                
                #### Combine cut off predictions and OOA predictions
                marker_kinematics_with_prediction = CombinePredictionAndCutOff(marker_kinematics_with_prediction)
            
            #### Get only values with succeding measurement made within 72 hours 
            marker_kinematics_with_prediction = SuccededWithinTime(marker_kinematics_with_prediction,time_within)
            
            #### Get only non-zero values
            marker_kinematics_with_prediction = marker_kinematics_with_prediction[marker_kinematics_with_prediction.displacement != 0]
            
            if mode == 'define':
                #### Cut off values based on parameter
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
        
        #### Get AUC
        auc = np.trapz(true_positive_rates,x = false_positive_rates)
        
        #### Append auc score to container
        auc_scores.append(auc)
    
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
    
    #### Set save path
    save_path = "{}\ROC\\OOA\\With CutOff\\ROC".format(data_path)
    if not os.path.exists(save_path+'\\'):
        os.makedirs(save_path+'\\')
        
    #### Save fig
    plt.savefig('{}\ROC OOA Threshold Min {} Max {} N {} With {} Min {} Max {} N {} Mode {}.png'.format(save_path,round(min(confidence_threshold_range),4),round(max(confidence_threshold_range),4),len(confidence_threshold_range),parameter_name.title(),round(min(parameter_threshold_range),4),round(max(parameter_threshold_range),4),len(parameter_threshold_range),mode),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    #### Plot AUC vs cut off threshold curve
    
    #### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot grid
    ax.grid()
    
    #### Plot results
    ax.scatter(parameter_threshold_range,auc_scores,c = parameter_threshold_range, cmap = 'plasma',s = 12,zorder = 3)
    
    #### Indicate maximum AUC
    auc_max = round(max(auc_scores),4)
    
    #### Print max AUC
    auc_max_text = 'Max AUC = {}'.format(auc_max)
    ax.add_artist(AnchoredText(auc_max_text,prop=dict(size=10), frameon=False,loc = 4))
    
    #### Set axis labels
    ax.set_xlabel('{}'.format(parameter_name.title()),fontsize = 14)
    ax.set_ylabel('Area under the ROC',fontsize = 14)
    
    #### Set figure label
    fig.suptitle('AUC for OOA vs {} Cut Off'.format(parameter_name.title()),fontsize = 15)
    
    #### Set figsize
    fig.set_figheight(6)
    fig.set_figwidth(8)

    #### Set save path
    save_path = "{}\ROC\\OOA\\With CutOff\\AUC".format(data_path)
    if not os.path.exists(save_path+'\\'):
        os.makedirs(save_path+'\\')

    #### Save fig
    plt.savefig('{}\AUC ROC OOA Threshold Min {} Max {} N {} With {} Min {} Max {} N {} Mode {}.png'.format(save_path,round(min(confidence_threshold_range),4),round(max(confidence_threshold_range),4),len(confidence_threshold_range),parameter_name.title(),round(min(parameter_threshold_range),4),round(max(parameter_threshold_range),4),len(parameter_threshold_range),mode),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')

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

def MarkerDataCountPlots(marker_data_count,time_bins = None):
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
    
    

def PlotROCArbitraryParameterPerSite(marker_kinematics,time_within,parameter_name,site):
    """
    Plots the ROC of a given parameter in the marker_kinematics data for a secified site

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame() 
        Marker kinematics data frame
    time_within: float
        Time in hours in which the next measurement is made
    parameter_name: string
        name of the parameter whose ROC is to be plotted
    site: string
        site_code of the site to be analyzed
        
        
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
    
    #### Get only displacement values < 100 cm
    marker_kinematics_with_condition = marker_kinematics_with_condition[marker_kinematics_with_condition.displacement <= 100]
        
    #### Get only the values for the specified site
    marker_kinematics_with_condition = marker_kinematics_with_condition[marker_kinematics_with_condition.site_id == site]
    
    try:
        #### Get false positive rates and true positive rates and corresponding thresholds
        fpr, tpr, thresholds = metrics.roc_curve(marker_kinematics_with_condition.condition.values,marker_kinematics_with_condition[parameter_name].values)
    except:
        #### Return nan if insufficient data
        fpr = np.array([np.nan])
        tpr = np.array([np.nan])
        thresholds = np.array([np.nan])
    
    #### Record results to data frame
    roc_results = pd.DataFrame({'tpr':tpr,'fpr':fpr,'threshold':thresholds})
    
    #### Compute for Youden Index
    roc_results['yi'] = tpr - fpr

    #### Get maximum Youden Index
    max_YI = max(roc_results.yi)
    
    try:
        #### Get threshold with maximum YI
        optimized_threshold = roc_results[roc_results.yi == max_YI].threshold.values[-1]
        
        #### Get fpr and tpr of maximum YI
        optimized_fpr = roc_results[roc_results.threshold == optimized_threshold].fpr.values[0]
        optimized_tpr = roc_results[roc_results.threshold == optimized_threshold].tpr.values[0]
    except:
        #### Return nan if not enough data
        optimized_fpr = np.nan
        optimized_tpr = np.nan
        optimized_threshold = np.nan
    
    try:
        #### Get AUC ROC value
        auc = metrics.roc_auc_score(marker_kinematics_with_condition.condition.values,marker_kinematics_with_condition[parameter_name].values)
    except:
        #### Return nan if not enough data
        auc = np.nan

    #### Plot results
    
    #### Initialize figure, open subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
        
    #### Plot random guess line
    random_guess_line, = ax.plot(np.arange(0,1,0.0001),np.arange(0,1,0.0001),'--',color = tableau20[6],label = 'Random Guess')
    
    #### Plot fpr vs tpr
    roc_line, = ax.plot(fpr,tpr,label = '{} Threshold'.format(parameter_name.title()))
    
    #### Plot Max YI line
    ax.plot(np.ones(1000)*optimized_fpr,np.linspace(optimized_fpr,optimized_tpr,1000),'--',color = tableau20[16])
    max_yi_line, = ax.plot([optimized_fpr],[optimized_tpr],'o--',color = tableau20[16],label = 'Max YI',markersize = 4)
    
    #### Print AUC ROC result
    ax.text(0.975,0.025,'Area under the ROC = {}\nMax YI = {}, at {:.2e}'.format(round(auc,4),round(max_YI,4),optimized_threshold),transform = ax.transAxes,ha = 'right',fontsize = 12)
        
    #### Print number of data points and time bin
    ax.text(0.975,0.025,'N {}'.format(len(marker_kinematics_with_condition)),transform = fig.transFigure,ha = 'right',fontsize = 10)
    
    #### Plot the legend
    plt.legend(handles = [roc_line,random_guess_line,max_yi_line],loc = 'upper right')
    
    #### Set axis labels
    ax.set_xlabel('False Positive Rates',fontsize = 14)
    ax.set_ylabel('True Positive Rates',fontsize = 14)
    
    #### Set figure label
    fig.suptitle('ROC Plot for {} Threshold for Site {}'.format(parameter_name.title(),site.upper()),fontsize = 15)
    
    #### Set figsize
    fig.set_figheight(8)
    fig.set_figwidth(8)
    
    #### Set aspect as equal
    ax.set_aspect('equal')
    
    #### Set save path
    save_path = "{}\\ROC\\Site\\{}\\Arbitrary Parameter".format(data_path,site)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
        
    #### Save fig
    plt.savefig('{}\\{} Threshold.png'.format(save_path,parameter_name.title()),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    return marker_kinematics_with_condition
 
def PlotROCSplinePredictionPerSite(marker_kinematics,time_within,confidence_threshold_range,site_code):
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
    
    #### Get only data from specified site
    marker_kinematics_with_condition = marker_kinematics_with_condition[marker_kinematics_with_condition.site_id == site_code.upper()]
    
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
    
    #### Set save path
    save_path = "{}\\ROC\\Site\\{}".format(data_path,site_code.upper())
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')   
        
    #### Save fig
    plt.savefig('{}\\ROC OOA Threshold Min {} Max {} N {}.png'.format(save_path,round(min(confidence_threshold_range),4),round(max(confidence_threshold_range),4),len(confidence_threshold_range)),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')

def PlotHistograms(marker_kinematics, parameter_name,time_within,bins = 750):
    """
    Plots the histogram plots of the specified parameter for cases (1) ALL, (2) Non zero displacement, (3) Non zero displacement with precedance of acceleration within prescribed time interval

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame()
        Dataframe containing the computed marker kinematics
    parameter_name: string
        Name of the parameter whose histogram is to be plot
    time_within: float
        Time in hours of the prescribed time_interval from the definition

        
    Returns
    -------------------
    Plots all the histogram for three cases

        
    """
    
    ############################################
    ##### Plot case 1: ALL unfiltered data #####
    ############################################
    
    #### Plot results from data frame filtering data with time interval > 30 days
    fig = plt.figure()
    ax = marker_kinematics[marker_kinematics.time_interval <= 720][parameter_name].hist(bins = bins,zorder = 3)
    
    #### Plot auxillary lines and labels for time interval
    if parameter_name == 'time_interval':
        
        #### Transform axes on xaxis
        trans = ax.get_xaxis_transform()
        
        #### 7 day line
        ax.axvline(x = 168,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.5)
        ax.annotate('7 days',xy = (169,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 4 day line
        ax.axvline(x = 96,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.5)
        ax.annotate('4 days',xy = (97,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 3 day line
        ax.axvline(x = 72,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.5)
        ax.annotate('3 days',xy = (73,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 2 day line
        ax.axvline(x = 48,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.5)
        ax.annotate('2 days',xy = (49,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 1 day line
        ax.axvline(x = 24,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.5)
        ax.annotate('1 day',xy = (25,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 4 hour line
        ax.axvline(x = 4,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.5)
        ax.annotate('4 hours',xy = (5,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
    
    #### Set axis labels
    if parameter_name == 'time_interval':
        ax.set_xlabel('Time Interval (hours)',fontsize = 14)
        ax.set_ylabel('Frequency',fontsize = 14)
    
    #### Set figure label
    if parameter_name == 'time_interval':
        fig.suptitle('Histogram Plot for Time Interval for ALL Marker Data',fontsize = 15)
    
    #### Set fig size
    fig.set_figheight(6.5)
    fig.set_figwidth(13)
    
    #### Set save path
    save_path = "{}\\Histograms\\{}".format(data_path,parameter_name)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')   
        
    #### Save fig
    plt.savefig('{}\\Histogram plot for {} bins {} all marker data.png'.format(save_path,parameter_name,bins),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')

    #######################################################
    ##### Plot case 2: ALL Non zero displacement data #####
    #######################################################
    
    #### Plot results from data frame filtering data with time interval > 30 days
    fig = plt.figure()
    ax = marker_kinematics[np.logical_and(marker_kinematics.time_interval <= 720,marker_kinematics.displacement != 0)][parameter_name].hist(bins = bins,zorder = 3,color = tableau20[8])
    
    #### Plot auxillary lines and labels for time interval
    if parameter_name == 'time_interval':
        
        #### Transform axes on xaxis
        trans = ax.get_xaxis_transform()
        
        #### 7 day line
        ax.axvline(x = 168,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('7 days',xy = (169,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 4 day line
        ax.axvline(x = 96,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('4 days',xy = (97,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 3 day line
        ax.axvline(x = 72,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('3 days',xy = (73,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 2 day line
        ax.axvline(x = 48,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('2 days',xy = (49,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 1 day line
        ax.axvline(x = 24,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('1 day',xy = (25,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 4 hour line
        ax.axvline(x = 4,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('4 hours',xy = (5,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
    
        #### Set xlim
        ax.set_xlim([-35,720])
        
    #### Set axis labels
    if parameter_name == 'time_interval':
        ax.set_xlabel('Time Interval (hours)',fontsize = 14)
        ax.set_ylabel('Frequency',fontsize = 14)
    
    #### Set figure label
    if parameter_name == 'time_interval':
        fig.suptitle('Histogram Plot for Time Interval for Non-Zero Displacement Marker Data',fontsize = 15)
    
    #### Set fig size
    fig.set_figheight(6.5)
    fig.set_figwidth(13)
    
    #### Set fig spacing
    fig.subplots_adjust(right = 0.88)
    
    #### Set save path
    save_path = "{}\\Histograms\\{}".format(data_path,parameter_name)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')   
        
    #### Save fig
    plt.savefig('{}\\Histogram plot for {} bins {} non zero disp marker data.png'.format(save_path,parameter_name,bins),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')

    ##############################################################################
    ##### Plot case 3: ALL Non zero displacement data preceding acceleration #####
    ##############################################################################
    
    #### Get data frame preceding acceleration
    preceded_acceleration = PrecededAccelerationWithTime(marker_kinematics,time_within)
    
    #### Plot results from data frame filtering data with time interval > 30 days
    fig = plt.figure()
    ax = preceded_acceleration[np.logical_and(preceded_acceleration.time_interval <= 720,preceded_acceleration.displacement != 0)][parameter_name].hist(bins = bins,zorder = 3,color = tableau20[4])
    
    #### Plot auxillary lines and labels for time interval
    if parameter_name == 'time_interval':
        
        #### Transform axes on xaxis
        trans = ax.get_xaxis_transform()
        
        #### 7 day line
        ax.axvline(x = 168,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('7 days',xy = (169,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 4 day line
        ax.axvline(x = 96,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('4 days',xy = (97,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 3 day line
        ax.axvline(x = 72,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('3 days',xy = (73,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 2 day line
        ax.axvline(x = 48,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('2 days',xy = (49,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 1 day line
        ax.axvline(x = 24,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('1 day',xy = (25,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 4 hour line
        ax.axvline(x = 4,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('4 hours',xy = (5,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
    
    #### Set axis labels
    if parameter_name == 'time_interval':
        ax.set_xlabel('Time Interval (hours)',fontsize = 14)
        ax.set_ylabel('Frequency',fontsize = 14)
    
    #### Set figure label
    if parameter_name == 'time_interval':
        fig.suptitle('Histogram Plot for Time Interval for Non Zero Marker Data Preceding Acceleration Within {} Hours'.format(time_within),fontsize = 15)
    
    #### Set fig size
    fig.set_figheight(6.5)
    fig.set_figwidth(13)
    
    #### Set save path
    save_path = "{}\\Histograms\\{}".format(data_path,parameter_name)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')   
        
    #### Save fig
    plt.savefig('{}\\Histogram plot for {} bins {} non zero disp marker data prec acc within {}.png'.format(save_path,parameter_name,bins,time_within),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')

    ######################################################################
    ##### Plot case 4: ALL Non zero displacement data with Time Bins #####
    ######################################################################
    
    #### Plot results from data frame filtering data with time interval > 30 days
    fig = plt.figure()
    ax = marker_kinematics[np.logical_and(marker_kinematics.time_interval <= 720,marker_kinematics.displacement != 0)][parameter_name].hist(bins = bins,zorder = 3,color = tableau20[8])
    
    #### Plot auxillary lines and labels for time interval
    if parameter_name == 'time_interval':
        
        #### Transform axes on xaxis
        trans = ax.get_xaxis_transform()
        
        #### 7 day line
        ax.axvline(x = 168,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('7 days',xy = (169,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 4 day line
        ax.axvline(x = 96,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('4 days',xy = (97,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 3 day line
        ax.axvline(x = 72,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('3 days',xy = (73,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 2 day line
        ax.axvline(x = 48,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('2 days',xy = (49,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 1 day line
        ax.axvline(x = 24,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('1 day',xy = (25,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 4 hour line
        ax.axvline(x = 4,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('4 hours',xy = (5,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### Set xlim
        ax.set_xlim([-35,720])
        
        #### Plot time bins
        
        #### Initialize time bin line container
        time_bin_patches = []
        
        #### Hard coded time bins
        time_bins = [(0,11),(11,33),(33,63),(63,81),(81,111),(111,154),(154,184),(184,720)] 
        
        #### Plot all time bins
        for i,time in enumerate(time_bins):
            time_bin_patch = ax.axvspan(time[0],time[1],facecolor = tableau20[(i)*2 + 1], alpha = 0.5,label = '{} to {}'.format(time[0],time[1]))
            time_bin_patches.append(time_bin_patch)
        
        ax.legend(bbox_to_anchor = (1.15,0.999),fontsize = 10)
        
        ax.text(1.060, 1.00,'Time Bins',fontsize = 10,transform = ax.transAxes)
        
    #### Set axis labels
    if parameter_name == 'time_interval':
        ax.set_xlabel('Time Interval (hours)',fontsize = 14)
        ax.set_ylabel('Frequency',fontsize = 14)
    
    #### Set figure label
    if parameter_name == 'time_interval':
        fig.suptitle('Histogram Plot for Time Interval for Non-Zero Displacement Marker Data',fontsize = 15)
    
    #### Set fig size
    fig.set_figheight(6.5)
    fig.set_figwidth(13)
    
    #### Set fig spacing
    fig.subplots_adjust(right = 0.88)
    
    #### Set save path
    save_path = "{}\\Histograms\\{}".format(data_path,parameter_name)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')   
        
    #### Save fig
    plt.savefig('{}\\Histogram plot for {} bins {} non zero disp marker data with time bins.png'.format(save_path,parameter_name,bins),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')


def PlotROCArbitraryParameterPerTimeBin(marker_kinematics,time_within,parameter_name,time_start,time_end):
    """
    Plots the ROC of a given parameter in the marker_kinematics data restricted in a specified time bin

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame() 
        Marker kinematics data frame
    time_within: float
        Time in hours in which the next measurement is made
    parameter_name: string
        name of the parameter whose ROC is to be plotted
    time_start: float
        Time in hours for the start of the bin
    time_end: float
        Time in hours for the end of the bin
        
        
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
    
    #### Get only displacement values < 100 cm
    marker_kinematics_with_condition = marker_kinematics_with_condition[marker_kinematics_with_condition.displacement <= 100]
    
    #### Get only the values for the specified time bin
    marker_kinematics_with_condition = marker_kinematics_with_condition[np.logical_and(marker_kinematics_with_condition.time_interval <= time_end,marker_kinematics_with_condition.time_interval >= time_start)]
    
    #### Get false positive rates and true positive rates and corresponding thresholds
    fpr, tpr, thresholds = metrics.roc_curve(marker_kinematics_with_condition.condition.values,marker_kinematics_with_condition[parameter_name].values)
    
    #### Record results to data frame
    roc_results = pd.DataFrame({'tpr':tpr,'fpr':fpr,'threshold':thresholds})
    
    #### Compute for Youden Index
    roc_results['yi'] = tpr - fpr
        
    #### Get maximum Youden Index
    max_YI = max(roc_results.yi)
    
    #### Get threshold with maximum YI
    optimized_threshold = roc_results[roc_results.yi == max_YI].threshold.values[-1]
    
    #### Get fpr and tpr of maximum YI
    optimized_fpr = roc_results[roc_results.threshold == optimized_threshold].fpr.values[0]
    optimized_tpr = roc_results[roc_results.threshold == optimized_threshold].tpr.values[0]
    
    #### Get AUC ROC value
    auc = metrics.roc_auc_score(marker_kinematics_with_condition.condition.values,marker_kinematics_with_condition[parameter_name].values)

    #### Plot results
    
    #### Initialize figure, open subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
        
    #### Plot random guess line
    random_guess_line, = ax.plot(np.arange(0,1,0.0001),np.arange(0,1,0.0001),'--',color = tableau20[6],label = 'Random Guess')
    
    #### Plot fpr vs tpr
    roc_line, = ax.plot(fpr,tpr,label = '{} Threshold'.format(parameter_name.title()))
    
    #### Plot Max YI line
    ax.plot(np.ones(1000)*optimized_fpr,np.linspace(optimized_fpr,optimized_tpr,1000),'--',color = tableau20[16])
    max_yi_line, = ax.plot([optimized_fpr],[optimized_tpr],'o--',color = tableau20[16],label = 'Max YI',markersize = 4)
    
    #### Print AUC ROC result
    ax.text(0.975,0.025,'Area under the ROC = {}\nMax YI = {}, at {:.2e}'.format(round(auc,4),round(max_YI,4),optimized_threshold),transform = ax.transAxes,ha = 'right',fontsize = 12)
    
    #### Print number of data points and time bin
    ax.text(0.975,0.025,'Time Bin {} to {}\nPos {} Neg {}'.format(time_start,time_end,len(marker_kinematics_with_condition[marker_kinematics_with_condition.condition == 1]),len(marker_kinematics_with_condition[marker_kinematics_with_condition.condition == -1])),transform = fig.transFigure,ha = 'right',fontsize = 10)
    
    #### Plot the legend
    plt.legend(handles = [roc_line,random_guess_line,max_yi_line],loc = 'upper right')
    
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
  
    #### Set save path 1
    save_path = "{}\ROC\\Time Bins\\{} to {}".format(data_path,time_start,time_end)
    if not os.path.exists(save_path+'\\'):
        os.makedirs(save_path+'\\')    
    
    #### Save fig
    plt.savefig('{}\{} Threshold.png'.format(save_path,parameter_name.title()),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    #### Set save path 2
    save_path = "{}\ROC\\Time Bins\\Per Parameter\\{}".format(data_path,parameter_name)
    if not os.path.exists(save_path+'\\'):
        os.makedirs(save_path+'\\')
        
    #### Save fig
    plt.savefig('{}\{} Threshold Time Bin {} to {}.png'.format(save_path,parameter_name,time_start,time_end),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')


def PlotROCArbitraryParameterPerTimeBinAndSite(marker_kinematics,time_within,parameter_name,time_start,time_end,site):
    """
    Plots the ROC of a given parameter in the marker_kinematics data restricted in a specified time bin and site

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame() 
        Marker kinematics data frame
    time_within: float
        Time in hours in which the next measurement is made
    parameter_name: string
        name of the parameter whose ROC is to be plotted
    time_start: float
        Time in hours for the start of the bin
    time_end: float
        Time in hours for the end of the bin
    site: string
        Name of specified site
        
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
    
    #### Get only displacement values < 100 cm
    marker_kinematics_with_condition = marker_kinematics_with_condition[marker_kinematics_with_condition.displacement <= 100]
    
    #### Get only the values for the specified time bin
    marker_kinematics_with_condition = marker_kinematics_with_condition[np.logical_and(marker_kinematics_with_condition.time_interval <= time_end,marker_kinematics_with_condition.time_interval >= time_start)]
    
    #### Get only the values for the specified site
    marker_kinematics_with_condition = marker_kinematics_with_condition[marker_kinematics_with_condition.site_id == site]
    
    try:
        #### Get false positive rates and true positive rates and corresponding thresholds
        fpr, tpr, thresholds = metrics.roc_curve(marker_kinematics_with_condition.condition.values,marker_kinematics_with_condition[parameter_name].values)
    except:
        #### Return nan if insufficient data
        fpr = np.array([np.nan])
        tpr = np.array([np.nan])
        thresholds = np.array([np.nan])
    
    #### Record results to data frame
    roc_results = pd.DataFrame({'tpr':tpr,'fpr':fpr,'threshold':thresholds})
    
    #### Compute for Youden Index
    roc_results['yi'] = tpr - fpr

    #### Get maximum Youden Index
    max_YI = max(roc_results.yi)
    
    try:
        #### Get threshold with maximum YI
        optimized_threshold = roc_results[roc_results.yi == max_YI].threshold.values[-1]
        
        #### Get fpr and tpr of maximum YI
        optimized_fpr = roc_results[roc_results.threshold == optimized_threshold].fpr.values[0]
        optimized_tpr = roc_results[roc_results.threshold == optimized_threshold].tpr.values[0]
    except:
        #### Return nan if not enough data
        optimized_fpr = np.nan
        optimized_tpr = np.nan
        optimized_threshold = np.nan
    
    try:
        #### Get AUC ROC value
        auc = metrics.roc_auc_score(marker_kinematics_with_condition.condition.values,marker_kinematics_with_condition[parameter_name].values)
    except:
        #### Return nan if not enough data
        auc = np.nan

    #### Plot results
    
    #### Initialize figure, open subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
        
    #### Plot random guess line
    random_guess_line, = ax.plot(np.arange(0,1,0.0001),np.arange(0,1,0.0001),'--',color = tableau20[6],label = 'Random Guess')
    
    #### Plot fpr vs tpr
    roc_line, = ax.plot(fpr,tpr,label = '{} Threshold'.format(parameter_name.title()))
    
    #### Plot Max YI line
    ax.plot(np.ones(1000)*optimized_fpr,np.linspace(optimized_fpr,optimized_tpr,1000),'--',color = tableau20[16])
    max_yi_line, = ax.plot([optimized_fpr],[optimized_tpr],'o--',color = tableau20[16],label = 'Max YI',markersize = 4)
    
    #### Print AUC ROC result
    ax.text(0.975,0.025,'Area under the ROC = {}\nMax YI = {}, at {:.2e}'.format(round(auc,4),round(max_YI,4),optimized_threshold),transform = ax.transAxes,ha = 'right',fontsize = 12)
    
    #### Print number of data points and time bin
    ax.text(0.975,0.025,'Time Bin {} to {}\nN {}'.format(time_start,time_end,len(marker_kinematics_with_condition)),transform = fig.transFigure,ha = 'right',fontsize = 10)
    
    #### Plot the legend
    plt.legend(handles = [roc_line,random_guess_line,max_yi_line],loc = 'upper right')
    
    #### Set axis labels
    ax.set_xlabel('False Positive Rates',fontsize = 14)
    ax.set_ylabel('True Positive Rates',fontsize = 14)
    
    #### Set figure label
    fig.suptitle('ROC Plot for {} Threshold for Site {}'.format(parameter_name.title(),site.upper()),fontsize = 15)
    
    #### Set figsize
    fig.set_figheight(8)
    fig.set_figwidth(8)
    
    #### Set aspect as equal
    ax.set_aspect('equal')
  
    #### Set save path
    save_path = "{}\ROC\\Site\\{}\\Time Bins\\{} to {}".format(data_path,site,time_start,time_end)
    if not os.path.exists(save_path+'\\'):
        os.makedirs(save_path+'\\')    
    
    #### Save fig
    plt.savefig('{}\{} Threshold.png'.format(save_path,parameter_name.title()),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')

def PlotHistogramsPerSite(marker_kinematics, parameter_name,time_within,site,bins = 750):
    """
    Plots the histogram plots of the specified parameter for cases (1) ALL, (2) Non zero displacement, (3) Non zero displacement with precedance of acceleration within prescribed time interval for a chosen site

    
    Parameters
    ------------------
    marker_kinematics: pd.DataFrame()
        Dataframe containing the computed marker kinematics
    parameter_name: string
        Name of the parameter whose histogram is to be plot
    time_within: float
        Time in hours of the prescribed time_interval from the definition
    site: string
        Site code of the chosen site
        
    Returns
    -------------------
    Plots all the histogram for three cases

        
    """
    
    #### Select only data from chosen site
    marker_kinematics = marker_kinematics[marker_kinematics.site_id == site]
    
    ############################################
    ##### Plot case 1: ALL unfiltered data #####
    ############################################
    
    #### Plot results from data frame filtering data with time interval > 30 days
    fig = plt.figure()
    ax = marker_kinematics[marker_kinematics.time_interval <= 720][parameter_name].hist(bins = bins,zorder = 3)
    
    #### Plot auxillary lines and labels for time interval
    if parameter_name == 'time_interval':
        
        #### Transform axes on xaxis
        trans = ax.get_xaxis_transform()
        
        #### 7 day line
        ax.axvline(x = 168,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.5)
        ax.annotate('7 days',xy = (169,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 4 day line
        ax.axvline(x = 96,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.5)
        ax.annotate('4 days',xy = (97,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 3 day line
        ax.axvline(x = 72,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.5)
        ax.annotate('3 days',xy = (73,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 2 day line
        ax.axvline(x = 48,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.5)
        ax.annotate('2 days',xy = (49,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 1 day line
        ax.axvline(x = 24,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.5)
        ax.annotate('1 day',xy = (25,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 4 hour line
        ax.axvline(x = 4,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.5)
        ax.annotate('4 hours',xy = (5,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
    
    #### Set axis labels
    if parameter_name == 'time_interval':
        ax.set_xlabel('Time Interval (hours)',fontsize = 14)
        ax.set_ylabel('Frequency',fontsize = 14)
    
    #### Set figure label
    if parameter_name == 'time_interval':
        fig.suptitle('Histogram Plot for Time Interval for ALL Marker Data',fontsize = 15)
    
    #### Set fig size
    fig.set_figheight(6.5)
    fig.set_figwidth(13)
    
    #### Set save path
    save_path = "{}\\Histograms\\Site\\{}\\{}".format(data_path,site,parameter_name)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')   
        
    #### Save fig
    plt.savefig('{}\\Histogram plot for {} bins {} all marker data.png'.format(save_path,parameter_name,bins),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')

    #######################################################
    ##### Plot case 2: ALL Non zero displacement data #####
    #######################################################
    
    #### Plot results from data frame filtering data with time interval > 30 days
    fig = plt.figure()
    ax = marker_kinematics[np.logical_and(marker_kinematics.time_interval <= 720,marker_kinematics.displacement != 0)][parameter_name].hist(bins = bins,zorder = 3,color = tableau20[8])
    
    #### Plot auxillary lines and labels for time interval
    if parameter_name == 'time_interval':
        
        #### Transform axes on xaxis
        trans = ax.get_xaxis_transform()
        
        #### 7 day line
        ax.axvline(x = 168,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('7 days',xy = (169,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 4 day line
        ax.axvline(x = 96,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('4 days',xy = (97,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 3 day line
        ax.axvline(x = 72,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('3 days',xy = (73,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 2 day line
        ax.axvline(x = 48,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('2 days',xy = (49,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 1 day line
        ax.axvline(x = 24,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('1 day',xy = (25,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 4 hour line
        ax.axvline(x = 4,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('4 hours',xy = (5,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
    
    #### Set axis labels
    if parameter_name == 'time_interval':
        ax.set_xlabel('Time Interval (hours)',fontsize = 14)
        ax.set_ylabel('Frequency',fontsize = 14)
    
    #### Set figure label
    if parameter_name == 'time_interval':
        fig.suptitle('Histogram Plot for Time Interval for Non-Zero Displacement Marker Data',fontsize = 15)
    
    #### Set fig size
    fig.set_figheight(6.5)
    fig.set_figwidth(13)
    
    #### Set save path
    save_path = "{}\\Histograms\\Site\\{}\\{}".format(data_path,site,parameter_name)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')   
        
    #### Save fig
    plt.savefig('{}\\Histogram plot for {} bins {} non zero disp marker data.png'.format(save_path,parameter_name,bins),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')

    ##############################################################################
    ##### Plot case 3: ALL Non zero displacement data preceding acceleration #####
    ##############################################################################
    
    #### Get data frame preceding acceleration
    preceded_acceleration = PrecededAccelerationWithTime(marker_kinematics,time_within)
    
    #### Plot results from data frame filtering data with time interval > 30 days
    fig = plt.figure()
    ax = preceded_acceleration[np.logical_and(preceded_acceleration.time_interval <= 720,preceded_acceleration.displacement != 0)][parameter_name].hist(bins = bins,zorder = 3,color = tableau20[4])
    
    #### Plot auxillary lines and labels for time interval
    if parameter_name == 'time_interval':
        
        #### Transform axes on xaxis
        trans = ax.get_xaxis_transform()
        
        #### 7 day line
        ax.axvline(x = 168,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('7 days',xy = (169,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 4 day line
        ax.axvline(x = 96,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('4 days',xy = (97,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 3 day line
        ax.axvline(x = 72,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('3 days',xy = (73,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 2 day line
        ax.axvline(x = 48,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('2 days',xy = (49,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 1 day line
        ax.axvline(x = 24,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('1 day',xy = (25,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
        
        #### 4 hour line
        ax.axvline(x = 4,ls = '--',color = tableau20[6],zorder = 2,lw = 1,alpha = 0.75)
        ax.annotate('4 hours',xy = (5,1.01),xycoords = trans,rotation = 60,fontsize = 8,rotation_mode = 'anchor')
    
    #### Set axis labels
    if parameter_name == 'time_interval':
        ax.set_xlabel('Time Interval (hours)',fontsize = 14)
        ax.set_ylabel('Frequency',fontsize = 14)
    
    #### Set figure label
    if parameter_name == 'time_interval':
        fig.suptitle('Histogram Plot for Time Interval for Non Zero Marker Data Preceding Acceleration Within {} Hours'.format(time_within),fontsize = 15)
    
    #### Set fig size
    fig.set_figheight(6.5)
    fig.set_figwidth(13)
    
    #### Set save path
    save_path = "{}\\Histograms\\Site\\{}\\{}".format(data_path,site,parameter_name)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')   
        
    #### Save fig
    plt.savefig('{}\\Histogram plot for {} bins {} non zero disp marker data prec acc within {}.png'.format(save_path,parameter_name,bins,time_within),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')

def PlotOptimalThresholdResults(time_bins,thresholds,parameter_name,cut_off_parameter = None,mode = None):
    """
    Plots the optimal threshold vs time bins of the result of the AUC and YI analysis of spline velocity and spline acceleration thresholds
    
    Parameters
    ------------------
    time_bins: List
        List containing the defined time bins
    sp_velocity_thresholds: List
        List containing the obtained optimized threshold
        
    Returns
    -------------------
    Plots the optimal threshold vs time bins
    """
    
    #### Initialize figure and axes
    
    if mode == 'cut_off_all':
        fig = plt.gcf()
        ax = fig.gca()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    #### Initialize color, labels, and tick format for parameter
    if parameter_name == 'sp velocity' or parameter_name == 'sp_velocity':
        color = tableau20[4]
        label = 'Velocity (cm/hr)'
        sup_title = 'Spline Velocity'
        ax.get_yaxis().get_major_formatter().set_powerlimits((-2,-4))
    elif parameter_name == 'sp acceleration' or parameter_name == 'sp_acceleration':
        color = tableau20[16]
        label = r'Acceleration (cm/hr$^2$)'
        sup_title = 'Spline Acceleration'
        ax.get_yaxis().get_major_formatter().set_powerlimits((-4,-5))
    elif parameter_name == 'displacement':
        color = tableau20[2]
        label = 'Displacement (cm)'
        sup_title = 'Displacement'
    elif parameter_name == 'velocity':
        color = tableau20[6]
        label = 'Velocity (cm/hr)'
        sup_title = 'Velocity'
    else:
        color = tableau20[0]
    
    #### Iterate for all time bins
    for i,(time_bin,threshold) in enumerate(zip(time_bins,thresholds)):
        
        #### Plot threshold per bin
        ax.plot(np.linspace(time_bin[0],time_bin[1],100),np.ones(100)*threshold,color = color,lw = 2.5)

        if i != 0:
            #### Plot connecting lines
            if mode == 'cut_off_all':
                ax.axvline(time_bin[0],color = 'black',ls = '--',alpha = 0.5/4.,lw = 1.0)
            else:
                ax.axvline(time_bin[0],color = 'black',ls = '--',alpha = 0.5,lw = 1.0)
        
    #### Set xlim
    ax.set_xlim(0,214)
    
    #### Set axis labels
    if mode == 'cut_off' or mode == 'cut_off_all':
        if cut_off_parameter == 'displacement':
            ax.set_ylabel('{} (cm)'.format(cut_off_parameter.title()),fontsize = 14)
    else:
        ax.set_ylabel(label,fontsize = 14)
    ax.set_xlabel('Time Interval (hours)',fontsize = 14)
    
    #### Set fig title
    if mode == 'cut_off':
        fig.suptitle('Optimal {} Cut Off for {} Threshold vs Time Bins'.format(cut_off_parameter.title(),sup_title),fontsize = 15)
    elif mode == 'cut_off_all':
        fig.suptitle('Optimal {} Cut Off vs Time Bins'.format(cut_off_parameter.title(),sup_title),fontsize = 15)
    else:
        fig.suptitle('Optimal {} Threshold vs Time Bins'.format(sup_title),fontsize = 15)
    
    #### Set fig size
    fig.set_figheight(6.5)
    fig.set_figwidth(13)
    
    #### Set save path
    if cut_off_parameter:
        save_path = "{}\\ROC\\Optimal Parameters\\Cut Off\\{}\\Optimal Value\\Parameter".format(data_path,cut_off_parameter)
    else:
        save_path = "{}\\ROC\\Optimal Parameters".format(data_path)
    if mode == 'cut_off' or mode == 'cut_off_all':
        save_path = "{}\\ROC\\Optimal Parameters\\Cut Off\\{}\\Optimal Value\\Cut Off".format(data_path,cut_off_parameter)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')   
        
    #### Save fig
    if mode == 'cut_off_all':
        plt.savefig('{}\\ALL threshold vs time bins'.format(save_path),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    else:
        plt.savefig('{}\\{} threshold vs time bins'.format(save_path,parameter_name),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
def PlotHistogramForThreshold(marker_kinematics,parameter_name,time_within,thresholds,time_bins,bins = 1000,mode = None,cut_off_parameter = None, cut_off_values = None):
    """
    Plots the histogram of specified parameter and compares it with the optimal cut off threshold
    
    Parameters
    ------------------
    marker_kinematics - pd.DataFrame()
        Data frame containing the kimenatic data of markers
    parameter_name - string
        name of the parameter
    time_within - float
        time in hours of the defined time interval between measurements
    thresholds - list
        list containing the threshold values
    time_bins - list
        list containing the time bins
        
    Returns
    -------------------
    Plots the optimal threshold vs time bins
    """
    #### Get marker_condition dataframe
    
    #### Group based on site_id and crack_id
    marker_kinematics_grouped = marker_kinematics.groupby(['site_id','crack_id'],as_index = False)
    marker_kinematics_with_condition = marker_kinematics_grouped.apply(MarkTrueConditions,time_within).reset_index()
    
    #### Get only succeded within specified time
    marker_kinematics_with_condition = SuccededWithinTime(marker_kinematics_with_condition,time_within)
    
    #### Apply preceded functions
    preceded_acceleration = marker_kinematics_with_condition[marker_kinematics_with_condition.condition == 1]
    preceded_deceleration = marker_kinematics_with_condition[marker_kinematics_with_condition.condition == -1]
    
    #### Filter values
    if parameter_name == 'sp_velocity':
        filter_value = 3
    elif parameter_name == 'sp_acceleration':
        filter_value = 0.002
    else:
        filter_value = 100
        
    #### Get data frame preceding acceleration filtered values
    preceded_acceleration = preceded_acceleration[preceded_acceleration[parameter_name].values <= filter_value]
    preceded_deceleration = preceded_deceleration[preceded_deceleration[parameter_name].values <= filter_value]
    
    #### Filter displacment
    preceded_acceleration = preceded_acceleration[preceded_acceleration.displacement > 0][preceded_acceleration.displacement <= 100]
    preceded_deceleration = preceded_deceleration[preceded_deceleration.displacement > 0][preceded_deceleration.displacement <= 100]
    
    #### Initialize xlabels and fig title
    if parameter_name == 'sp_velocity':
        label = 'Velocity (cm/hr)'
        sup_title = 'Spline Velocity'
    elif parameter_name == 'sp_acceleration':
        label = r'Acceleration (cm/hr$^2$)'
        sup_title = 'Spline Acceleration'
    elif parameter_name == 'velocity':
        label = 'Velocity (cm/hr)'
        sup_title = 'Velocity'
    elif parameter_name == 'displacement':
        label = 'Displacement (cm)'
        sup_title = 'Displacement'
    
    #### Plot threshold lines and annotations
    for i,(time_bin,threshold) in enumerate(zip(time_bins,thresholds)):
        
        #### Get current df
        cur_df_deceleration = preceded_deceleration[np.logical_and(preceded_deceleration.time_interval >= time_bin[0],preceded_deceleration.time_interval <= time_bin[1])]
        cur_df_acceleration = preceded_acceleration[np.logical_and(preceded_acceleration.time_interval >= time_bin[0],preceded_acceleration.time_interval <= time_bin[1])]

        #### Initialize figure and axes
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        #### Filter cut off values
        if mode == 'cut_off':
            cur_df_deceleration = cur_df_deceleration[cur_df_deceleration[cut_off_parameter] >= cut_off_values[i]]
            cur_df_acceleration = cur_df_acceleration[cur_df_acceleration[cut_off_parameter] >= cut_off_values[i]]
        elif mode == 'reverse_cut_off':
            cur_df_deceleration = cur_df_deceleration[cur_df_deceleration[cut_off_parameter] < cut_off_values[i]]
            cur_df_acceleration = cur_df_acceleration[cur_df_acceleration[cut_off_parameter] < cut_off_values[i]]

        #### Plot histogram of data within time bin
        cur_df_deceleration[parameter_name].hist(bins = bins,zorder = 3,color = tableau20[6],ax = ax,label = 'Deceleration')
        cur_df_acceleration[parameter_name].hist(bins = bins,zorder = 4,color = tableau20[4],ax = ax,alpha = 0.75,label = 'Acceleration')
        
        #### Plotting threshold line
        ax.axvline(threshold,ls = '--',color = tableau20[16],zorder = 5,alpha = 0.75,label = 'Threshold')
        
        #### Print time bin and number of bins and positives and negatives
        ax.text(0.975,0.025,'{} to {}\n{} Bins\nPos {} Neg {}'.format(time_bin[0],time_bin[1],bins,len(cur_df_acceleration),len(cur_df_deceleration)),transform = fig.transFigure,ha = 'right',fontsize = 10)
        
        #### Set axis labels
        ax.set_xlabel(label,fontsize = 14)
        ax.set_ylabel('Frequency',fontsize = 14)
        
        #### Set fig sup title
        fig.suptitle('Histogram Plot for {}'.format(sup_title),fontsize = 15)
        
        #### Draw legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1],labels[::-1])
        
        #### Set fig size
        fig.set_figheight(6.5)
        fig.set_figwidth(13)
        
        #### Set save path
        if mode == 'cut_off':
            save_path = "{}\\Histograms\\Cut Off\\{}".format(data_path,parameter_name)
        elif mode == 'reverse_cut_off':
            save_path = "{}\\Histograms\\Reverse Cut Off\\{}".format(data_path,parameter_name)
        else:
            save_path = "{}\\Histograms\\Thresholds\\{}".format(data_path,parameter_name)
        if not os.path.exists(save_path+'/'):
            os.makedirs(save_path+'/')   
            
        #### Save fig
        if mode == 'cut_off' or mode == 'reverse_cut_off':
            plt.savefig('{}\\{} bins {} threshold histogram time bin {} to {}'.format(save_path,bins,parameter_name,time_bin[0],time_bin[1]),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
        else:
            plt.savefig('{}\\{} threshold histogram time bin {} to {}'.format(save_path,parameter_name,time_bin[0],time_bin[1]),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
        
        #### Close fig
        plt.close()
        
def PlotSplineVelComputation(marker_kinematics,time_within,time_bin,sp_velocity_threshold,sp_acceleration_threshold,num_pts = 10,mode = 'normal'):
    '''
    Plots the spline computation for every marker in the specified time bin and compares it with the computed threshold
    
    Parameters
    ---------------------------
    marker_kinematics - pd.DataFrame()
        Data frame containing the marker data with kinematic data
    time_within - float
        time in hours of the defined time interval between measurements
    time_bin - array shape(2)
        Specified time bin
    sp_velocity_threshold - float
        Spline velocity in cm/hour of the specified time bin
    sp_acceleration_threshold - float
        Spline acceleration threshold in cm/hour^2 of the specified time bin
    
    Returns
    ---------------------------
    None
    
    '''
    #### Get marker_condition dataframe
    
    #### Group based on site_id and crack_id
    marker_kinematics_grouped = marker_kinematics.groupby(['site_id','crack_id'],as_index = False)
    marker_kinematics_with_condition = marker_kinematics_grouped.apply(MarkTrueConditions,time_within).reset_index()
    
    #### Restrict data frame to specified condition
    
    #### Non zero displacement
    marker_data_to_plot = marker_kinematics_with_condition[marker_kinematics_with_condition.displacement > 0]
    
    #### Displacement filter
    marker_data_to_plot = marker_data_to_plot[marker_data_to_plot.displacement <= 100]
    
    #### Within time bin
    marker_data_to_plot = marker_data_to_plot[np.logical_and(marker_data_to_plot.time_interval >= time_bin[0],marker_data_to_plot.time_interval <= time_bin[1])]
    
    for fig_num,(timestamp, site_id, crack_id) in enumerate(marker_data_to_plot[['timestamp','site_id','crack_id']].values):
        
        #### Skip to specified figure number
#        if fig_num <= 0:
#            continue
        
        #### Get the site and marker data
        marker_data = marker_kinematics_with_condition[np.logical_and(marker_kinematics_with_condition.site_id == site_id,marker_kinematics_with_condition.crack_id == crack_id)]
        
        #### Get the next data
        try:
            next_data = marker_data.ix[marker_data[marker_data.timestamp == timestamp].index + 1]
        except:
            next_data = None
        
        #### Get the last 10 data
        marker_data = marker_data[marker_data.timestamp <= timestamp].tail(num_pts)
        
        #### Get time elapsed based on time_interval
        marker_data['time'] = marker_data['time_interval'].cumsum()
        
        print marker_data
        
        #### Get time and displacement values for spline computation
        time = marker_data.time.values
        meas = marker_data.meas.values
        
        #### Get interpolation time and timestamps
        time_int = np.linspace(time[0],time[-1],1000)
        datetime_int = pd.to_datetime(np.linspace(marker_data.iloc[0].timestamp.value,marker_data.iloc[-1].timestamp.value,1000))
        
        #### Commence interpolation
        try:
            if mode == 'normal':
                #### Take the gaussian average of data points and its variance
                _,var = moving_average(meas)
                sp = UnivariateSpline(time,meas,w=1/np.sqrt(var))
                
                #### Spline interpolation values    
                disp_int = sp(time_int)
                vel_int = sp.derivative(n=1)(time_int)
                acc_int = sp.derivative(n=2)(time_int)
            elif mode == '0.15 sf':
                #### Interpolate without using weighting functions
                sp = UnivariateSpline(time,meas)
                
                #### Set smoothing factor to 0.15
                sp.set_smoothing_factor(0.15)
                
                #### Spline interpolation values
                disp_int = sp(time_int)
                vel_int = sp.derivative(n=1)(time_int)
                acc_int = sp.derivative(n=2)(time_int)
                
            elif mode == 'polyfit 4':
                #### Interpolate using polyfit deg 4
                coeff = np.polyfit(time,meas,4)
                polynomial = np.poly1d(coeff)
                
                #### Interpolation values
                disp_int = polynomial(time_int)
                vel_int = polynomial.deriv(1)(time_int)
                acc_int = polynomial.deriv(2)(time_int)

        except:
            print "Interpolation error {} {}".format(marker_data.site_id.values[0],marker_data.crack_id.values[0])
            disp_int = np.zeros(len(time_int))
            vel_int = np.zeros(len(time_int))
            acc_int = np.zeros(len(time_int))
            
        #### Commence plotting
        fig = plt.figure()
        disp_ax = plt.subplot2grid((2,3),(0,0),2,2)
        vel_ax = plt.subplot2grid((2,3),(0,2))
        acc_ax = plt.subplot2grid((2,3),(1,2),sharex = vel_ax)
        conf_vel_ax = inset_axes(vel_ax,width = "25%",height = "25%",loc = 4)
        conf_acc_ax = inset_axes(acc_ax,width = "25%",height = "25%",loc = 4)
        
        #### Display grid
        disp_ax.grid()
        
        #### Plot meas interpolation
        interpolation_line, = disp_ax.plot(datetime_int,disp_int,color = tableau20[0],label = 'Interpolation')
        
        #### Plot meas data
        disp_ax.plot(np.append(marker_data.timestamp.values,next_data.timestamp),np.append(meas,next_data.meas),'--',color = tableau20[1],alpha = 0.5)
        data_line, = disp_ax.plot(np.append(marker_data.timestamp.values,next_data.timestamp),np.append(meas,next_data.meas),'.',color = tableau20[12],label = 'Data',markersize = 8)
        
        #### Plot velocity interpolation
        vel_ax.plot(time_int,vel_int,color = tableau20[4])
        
        #### Plot velocity threshold
        vel_ax.axhline(sp_velocity_threshold,ls = '--',color = tableau20[16])
        
        #### Plot acceleration interpolation
        acc_ax.plot(time_int,acc_int,color = tableau20[6])
        acc_ax.axhline(sp_acceleration_threshold,ls = '--',color = tableau20[16])
        
        #### Compute for the classification based on threshold and conditions

        vel_class = marker_data.iloc[-1].condition + (vel_int[-1] >= sp_velocity_threshold)*1
        acc_class = marker_data.iloc[-1].condition + (acc_int[-1] >= sp_acceleration_threshold)*1
        
        #### Plot confusion matrix
        for i,conf_ax in enumerate([conf_vel_ax,conf_acc_ax]):
            conf_ax.axvline(0.5,color = 'black',lw = 0.5)
            conf_ax.axhline(0.5,color = 'black',lw = 0.5)
            
            #### Get classification for current class
            if i == 0:
                classification = vel_class
            else:
                classification = acc_class
            
            #### True positives
            if classification == 2:
                conf_ax.axvspan(0,0.5,0.5,1,color = tableau20[4],alpha = 0.5)
                
            #### False positive
            elif classification == 0:
                conf_ax.axvspan(0,0.5,0,0.5,color = tableau20[6],alpha = 0.5)
            
            #### False negative
            elif classification == 1:
                conf_ax.axvspan(0.5,1,0.5,1,color = tableau20[6],alpha = 0.5)
                
            #### True negative
            elif classification == -1:
                conf_ax.axvspan(0.5,1,0,0.5,color = tableau20[4],alpha = 0.5)
        
            #### Set xlim and ylim
            conf_ax.set_xlim([0,1])
            conf_ax.set_ylim([0,1])
            
            #### Print confusion matrix labels
            conf_ax.text(0.25,0.75,'TP',transform = conf_ax.transAxes,fontsize = 8,ha = 'center',va = 'center')
            conf_ax.text(0.75,0.75,'FN',transform = conf_ax.transAxes,fontsize = 8,ha = 'center',va = 'center')
            conf_ax.text(0.75,0.25,'TN',transform = conf_ax.transAxes,fontsize = 8,ha = 'center',va = 'center')
            conf_ax.text(0.25,0.25,'FP',transform = conf_ax.transAxes,fontsize = 8,ha = 'center',va = 'center')
            
            #### Remove ticks
            conf_ax.tick_params('both',which = 'both',bottom = 'off',left = 'off',labelbottom = 'off',labelleft = 'off')
            
            #### Set equal aspect ratio
            conf_ax.set_aspect('equal')
            
            #### Set axes transparency
            conf_ax.patch.set_alpha(0.5)
        
        #### Print time bin
        acc_ax.text(0.975,0.025,'Time Bin {} to {}'.format(time_bin[0],time_bin[1]),transform = fig.transFigure,ha = 'right',fontsize = 10)
        
        #### Set tick parameters
        vel_ax.tick_params(which = 'both',left = 'off',right = 'on',labelright = 'on',bottom = 'off',labelbottom = 'off',labelleft = 'off')
        acc_ax.tick_params(which = 'both',left = 'off',right = 'on',labelright = 'on',labelleft = 'off')
        disp_ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
        
        #### Set legend
        disp_ax.legend(handles = [data_line,interpolation_line],loc = 1)
        
        #### Set axis labels
        disp_ax.set_ylabel('Meas (cm)',fontsize = 14)
        disp_ax.set_xlabel('Datetime',fontsize = 14)
        vel_ax.set_ylabel('Velocity (cm/hr)',fontsize = 14)
        acc_ax.set_ylabel(r'Acceleration (cm/hr$^2$)',fontsize = 14)
        acc_ax.set_xlabel('Time (hr)',fontsize = 14)
        
        #### Set axis label position
        vel_ax.yaxis.set_label_position('right')
        acc_ax.yaxis.set_label_position('right')
        
        #### Set fig sup title
        fig.suptitle('{} Site {} Marker {}'.format(marker_data.iloc[-1].timestamp.strftime("%b %d, %Y"),marker_data.iloc[-1].site_id.upper(),marker_data.iloc[-1].crack_id.title()),fontsize = 15)
        
        #### Set fig size
        fig.set_figwidth(15)
        fig.set_figheight(10)
        
        #### Set fig spacing
        fig.subplots_adjust(top = 0.92,bottom = 0.11,left = 0.08,right = 0.92,hspace = 0.05, wspace = 0.05)
        
        #### Set savefig directory
        if mode == 'normal':
            save_path = "{}\\Interpolations\\Default\\{} to {}".format(data_path,time_bin[0],time_bin[1])
            if not os.path.exists(save_path+'/'):
                os.makedirs(save_path+'/')
        else:
            save_path = "{}\\Interpolations\\{}\\{} to {}".format(data_path,mode,time_bin[0],time_bin[1])
            if not os.path.exists(save_path+'/'):
                os.makedirs(save_path+'/')
            
        #### Save fig
        plt.savefig('{}\\{} {} {} {}'.format(save_path,fig_num +1,marker_data.iloc[-1].timestamp.strftime("%Y-%m-%d_%H-%M"),marker_data.iloc[-1].site_id.upper(),marker_data.iloc[-1].crack_id.title()),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
        
        #### Close fig
        plt.close()

def ObtainOptimalThresholds(marker_kinematics,time_within,parameters,cut_off_parameter,cut_off_parameter_range,time_bins):
    """
    Generate ROC AUC for all parameters and optimal thresholds
    
    Parameters
    ---------------------------
    marker_kinematics - pd.DataFrame()
        Data frame containing the marker data with kinematic data
    time_within - float
        time in hours of the defined time interval between measurements
    parameters - list
        List containing all parameter names
    cut_off_parameter_name - string
        Name of the cut off parameter
    cut_off_parameter_range - list
        Range of the value of the cut off parameter
    time_bins - list
        List of the defined time bins
    
    Returns
    ---------------------------
    optimal_thresholds - pd.DataFrame with columns ['parameter','time_bin','auc','cut_off','optimal_value','cut_off_parameter']
        Optimal thresholds value data frame with auc, cut off, and threshold value
    
    """
    
    
    #### Initialize results container
    optimal_thresholds = pd.DataFrame(columns = ['parameter','time_bin','auc','cut_off','optimal_value','cut_off_parameter'])

    #### Iterate with respect to parameters then time bin
    for time_bin in time_bins:
        
        for parameter in parameters:
            #### Generate ROC AUC plots and collect results
            auc, cut_off, optimal_value = PlotROCArbitraryParameterWithParameterCutOff(marker_kinematics,72,parameter,cut_off_parameter,cut_off_parameter_range,time_bin)
            
            #### Save results to container
            optimal_thresholds = optimal_thresholds.append(pd.DataFrame([{'parameter':parameter,'time_bin':'{} to {}'.format(time_bin[0],time_bin[1]),'auc':auc,'cut_off':cut_off,'optimal_value':optimal_value,'cut_off_parameter':cut_off_parameter}]),ignore_index = True)
            
    return optimal_thresholds
        
        
def OptimalThresholdPlots(marker_kinematics,time_within,optimal_thresholds,bins = 500):
    """
    Generates all ROC plots and optimal threshold plots 
        (i) AUC vs Parameter
        (ii) Parameter vs Time interval
        (iii) cut off vs time interval
        (iv) cut off vs time interval for all
        (v) histogram per parameter
        (vi) histogram of site per parameter
        
    Parameters
    ---------------------------
    marker_kinematics - pd.DataFrame()
        Data frame containing the marker data with kinematic data
    time_within - float
        time in hours of the defined time interval between measurements
    optimal_thresholds - pd.DataFrame()
        Data frame containing the optimal threshold values aucs and cut off values
        
    Returns
    ---------------------------
    None
    """
    
    ############################################
    #### Plot AUC vs Parameter Per Time Bin ####
    ############################################
    
    #### Iterate for all time bins
    for i,time_bin in enumerate(np.unique(optimal_thresholds.time_bin.values)):
        
        #### Get current df
        cur_df = optimal_thresholds[optimal_thresholds.time_bin == time_bin]
        
        #### Initialize figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        #### Plot to current figure
        cur_df.plot.bar(x = ['parameter'],y = ['auc'],color = tableau20[(i)*2],ax = ax,align = 'center',width = 0.40)
        
        #### Rotate x ticks
        plt.xticks(rotation = 'horizontal',fontsize = 12)
        
        #### Set ylim
        ax.set_ylim([0.950*min(optimal_thresholds.auc.values),max(optimal_thresholds.auc.values)*1.05])
        
        #### Set axis labels
        ax.set_ylabel('AUC',fontsize = 14)
        ax.set_xlabel('Parameters',fontsize = 14)
        
        #### Hide legend
        ax.legend().set_visible(False)
        
        #### Set fig sup title
        fig.suptitle('AUC vs Parameter Plot for Time Bin {}'.format(time_bin),fontsize = 16)
        
        #### Set fig size 
        fig.set_figwidth(11)
        fig.set_figheight(8)
        
        #### Set save path
        save_path = "{}\\ROC\\Optimal Parameters\\Cut Off\\{}\\AUC vs Parameter\\".format(data_path,np.unique(optimal_thresholds.cut_off_parameter.values)[0])
        if not os.path.exists(save_path+'/'):
            os.makedirs(save_path+'/')

        #### Save fig
        plt.savefig('{}\\{}'.format(save_path,time_bin),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
        
        ### Close fig
        plt.close()
        
    ############################################
    #### Plot AUC vs Parameter ALL Time Bin ####
    ############################################

    #### Initialize figure and subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Initialize colors container
    colors = {}
    
    #### Set colors
    for i, time_bin in enumerate(np.unique(optimal_thresholds.time_bin.values)):
        colors[time_bin] = tableau20[(i)*2]
    
    #### Plot data
    optimal_thresholds.plot.bar(x = ['parameter'],y = ['auc'],color = [colors[i] for i in optimal_thresholds['time_bin']],ax = ax,align = 'center',width = 0.40)
    
    #### Rotate x ticks
    plt.xticks(rotation = 70,fontsize = 10, ha = 'right')
    
    #### Label containers
    handles = []
    
    #### Set label for each time bin
    for i,container in enumerate(ax.containers[0]):
        if i % 4 == 0:
            container.set_label(optimal_thresholds.time_bin.values[i])
            handles.append(container)
    
    #### Remove legend frame
    ax.legend(handles = handles).get_frame().set_visible(False)
    
    #### Set ylim
    ax.set_ylim([0.950*min(optimal_thresholds.auc.values),max(optimal_thresholds.auc.values)*1.05])
    
    #### Set axis labels
    ax.set_ylabel('AUC',fontsize = 14)
    ax.set_xlabel('Parameters',fontsize = 14)
        
    #### Set fig sup title
    fig.suptitle('AUC vs Parameter'.format(time_bin),fontsize = 16)
    
    #### Set fig size 
    fig.set_figwidth(15)
    fig.set_figheight(10)
    
    #### Set fig spacing
    fig.subplots_adjust(top = 0.92,bottom = 0.17,left = 0.07,right = 0.95)
    
    #### Set save path
    save_path = "{}\\ROC\\Optimal Parameters\\Cut Off\\{}\\AUC vs Parameter\\".format(data_path,np.unique(optimal_thresholds.cut_off_parameter.values)[0])
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')

    #### Save fig
    plt.savefig('{}\\ALL'.format(save_path),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    #### Close fig
    plt.close()
    
    #################################################
    #### Plot Optimal Parameter vs Time Interval ####
    #################################################
    
    #### Iterate for every parameter
    for parameter in np.unique(optimal_thresholds.parameter):

        #### Get current data frame
        cur_df = optimal_thresholds[optimal_thresholds.parameter == parameter]
        
        #### Get time bins
        time_bins = map(lambda x:tuple(map(lambda y:int(y),x.split(' to '))),cur_df.time_bin.values)
        
        #### Plot results
        PlotOptimalThresholdResults(time_bins,cur_df.optimal_value.values,parameter,cur_df.cut_off_parameter.values[0])
        
        #### Close fig
        plt.close()
        
    #############################################################
    #### Plot Optimal Cut Off vs Time Interval Per Parameter ####
    #############################################################

        PlotOptimalThresholdResults(time_bins,cur_df.cut_off.values,parameter,cur_df.cut_off_parameter.values[0],mode = 'cut_off')
        
        #### Close fig
        plt.close()
    
    #################################################################
    #### Plot Optimal Cut Off vs Time Interval For ALL Parameter ####
    #################################################################
    
    #### Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Offset optimal thresholds parameter
    optimal_thersholds_off_set = offset_cut_off(optimal_thresholds)
    
    #### Iterate for every parameter
    for i,parameter in enumerate(np.unique(optimal_thresholds.parameter)):
        
        #### Get current data frame
        cur_df = optimal_thersholds_off_set[optimal_thersholds_off_set.parameter == parameter]
        
        #### Get time bins
        time_bins = map(lambda x:tuple(map(lambda y:int(y),x.split(' to '))),cur_df.time_bin.values)
        
        #### Plot results
        PlotOptimalThresholdResults(time_bins,cur_df.cut_off.values,parameter,cur_df.cut_off_parameter.values[0],mode = 'cut_off_all')
    
    #### Plot legend handles
    velocity_line, = ax.plot([],[],color = tableau20[6], label = 'Velocity')
    displacement_line, = ax.plot([],[],color = tableau20[2], label = 'Displacement')
    sp_velocity_line, = ax.plot([],[],color = tableau20[4], label = 'Spline Velocity')
    sp_acceleration_line, = ax.plot([],[],color = tableau20[16], label = 'Spline Acceleration')
    
    #### Plot legend
    ax.legend(handles = [sp_acceleration_line,sp_velocity_line,displacement_line,velocity_line])
    
    #### Set save path
    save_path = "{}\\ROC\\Optimal Parameters\\Cut Off\\{}\\Optimal Value\\Cut Off".format(data_path,optimal_thersholds_off_set.cut_off_parameter.values[0])
    
    #### Save fig
    plt.savefig('{}\\ALL threshold vs time bins'.format(save_path),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    ###################################################
    #### Plot Histogram for Parameter per Time Bin ####
    ###################################################
    
    #### Iterate for every parameter
    for parameter in np.unique(optimal_thresholds.parameter):

        #### Get current data frame
        cur_df = optimal_thresholds[optimal_thresholds.parameter == parameter]
        
        #### Get time bins
        time_bins = map(lambda x:tuple(map(lambda y:int(y),x.split(' to '))),cur_df.time_bin.values)
        
        #### Plot results
        PlotHistogramForThreshold(marker_kinematics,parameter,time_within,cur_df.optimal_value.values,time_bins,bins = bins,mode = 'cut_off',cut_off_parameter = 'displacement',cut_off_values = cur_df.cut_off.values)    

    #########################################################################
    #### Plot Cut Off Non-excedance Histogram for Parameter per Time Bin ####
    #########################################################################
    
    #### Iterate for every parameter
    for parameter in np.unique(optimal_thresholds.parameter):

        #### Get current data frame
        cur_df = optimal_thresholds[optimal_thresholds.parameter == parameter]
        
        #### Get time bins
        time_bins = map(lambda x:tuple(map(lambda y:int(y),x.split(' to '))),cur_df.time_bin.values)
        
        #### Plot results
        PlotHistogramForThreshold(marker_kinematics,parameter,time_within,cur_df.optimal_value.values,time_bins,bins = bins,mode = 'reverse_cut_off',cut_off_parameter = 'displacement',cut_off_values = cur_df.cut_off.values)    

    #########################################
    #### Plot Per Site Marker Data Count ####
    #########################################
    
    #### Iterate for every parameter
    for parameter in np.unique(optimal_thresholds.parameter):
        
        #### Get current data frame
        cur_df = optimal_thresholds[optimal_thresholds.parameter == parameter]
        
        #### Get time bins
        time_bins = map(lambda x:tuple(map(lambda y:int(y),x.split(' to '))),cur_df.time_bin.values)
        
        #### Iterate for every time bin
        for time_bin in time_bins:
            
            #### Get current optimal thresholds data frame
            cur_df = optimal_thresholds[np.logical_and(optimal_thresholds.parameter == parameter,optimal_thresholds.time_bin == '{} to {}'.format(time_bin[0],time_bin[1]))]
            
            #### Get only values taken within specified time_within
            cur_marker_df = SuccededWithinTime(marker_kinematics,time_within)
            
            #### Get only non-zero displacement values
            cur_marker_df = cur_marker_df[cur_marker_df.displacement != 0]
            
            #### Get only displacement values < 100 cm
            cur_marker_df = cur_marker_df[cur_marker_df.displacement <= 100]

            #### Get only specified time bin
            cur_marker_df = cur_marker_df[np.logical_and(cur_marker_df.time_interval >= time_bin[0],cur_marker_df.time_interval <= time_bin[1])]
            
            #### Cut off filter
            cur_marker_df = cur_marker_df[cur_marker_df[cur_df.cut_off_parameter.values[0]] >= cur_df.cut_off.values[0]]
            
            #### Get marker count df
            marker_count = GetPerSiteMarkerDataCount(cur_marker_df,time_within)
            
            ax = marker_count.plot.barh(y = ['Marker Data'],stacked = True)
            
            #### Get current figure
            fig = plt.gcf()
            
            #### Set y labels and figure title
            ax.set_ylabel('Site Code',fontsize = 14)
            ax.set_xlabel('Count',fontsize = 14)
            fig.suptitle('Marker Data Count for {} Threshold Time Bin {}'.format(parameter,time_bin),fontsize = 15)
            
            #### Set tick label size
            ax.tick_params(labelsize = 8)
            
            #### Remove frame from legend
            ax.legend().get_frame().set_visible(False)
            
            #### Set fig size
            fig.set_figheight(7.5)
            fig.set_figwidth(13)
            
            #### Set save path
            save_path = "{}\\Histograms\\Cut Off\\{}\\Data Count".format(data_path,parameter)
            if not os.path.exists(save_path+'/'):
                os.makedirs(save_path+'/')
            
            #### Save fig
            plt.savefig('{}//Marker Data Count Time Bin {}.png'.format(save_path,time_bin),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
            
            ### Close fig
            plt.close()

def PlotDataCountPerTimeBin(marker_kinematics,time_within,time_bins):
    """
    Plot data count per time bin per parameter
    
    Parameters
    ---------------------------
    marker_kinematics - pd.DataFrame()
        Marker kinematics data frame
    time_witin - float
        Time in hours of the defined time interval between measurements
    time_bins - list
        List of the defined time bins

    Returns
    ---------------------------
    None
    """
    
    #### Iterate for every time bin
    for time_bin in time_bins:
        
        #### Get current marker df
        cur_df = marker_kinematics[np.logical_and(marker_kinematics.time_interval >= time_bin[0],marker_kinematics.time_interval <= time_bin[1])]
        
        #### Displacement filters
        cur_df = cur_df[cur_df.displacement != 0]
        cur_df = cur_df[cur_df.displacement <= 100]
        
        #### Get current marker data count df
        cur_marker_count_df = GetPerSiteMarkerDataCount(cur_df,time_within)
        
        
        #### Plot results
        ax = cur_marker_count_df.plot.barh(y = ['Marker Data'],stacked = True)
        
        #### Get current figure
        fig = plt.gcf()
        
        #### Set y labels and figure title
        ax.set_ylabel('Site Code',fontsize = 14)
        ax.set_xlabel('Count',fontsize = 14)
        fig.suptitle('Marker Data Count Time Bin {} to {}'.format(time_bin[0],time_bin[1]),fontsize = 15)
        
        #### Set tick label size
        ax.tick_params(labelsize = 8)
        
        #### Remove frame from legend
        ax.legend().get_frame().set_visible(False)
        
        #### Set fig size
        fig.set_figheight(7.5)
        fig.set_figwidth(13)
        
        #### Set save path
        save_path = "{}\\Data Count\\Time Bins".format(data_path)
        if not os.path.exists(save_path+'/'):
            os.makedirs(save_path+'/')
        
        #### Save fig
        plt.savefig('{}//Marker Data Count Time Bin {} to {}.png'.format(save_path,time_bin[0],time_bin[1]),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
        
        ### Close fig
        plt.close()
        
def OptimizeThresholds(optimal_thresholds):
    """
    Obtain optimized thresholds per time bin
    
    Parameters
    ---------------------------
    optimal_thresholds - pd.DataFrame()
        Data frame of optimized thresholds per parameter
        
    Returns
    ---------------------------
    optimized_thresholds - pd.DataFrame()
        Optimized thresholds per time bin
    """
    
    #### Initialize results data frame
    optimized_thresholds = pd.DataFrame(columns = optimal_thresholds.columns)
    
    #### Iterate for all time bins
    for time_bin in np.unique(optimal_thresholds.time_bin.values):
        
        #### Get data frame for current time bin
        cur_df = optimal_thresholds[optimal_thresholds.time_bin == time_bin]
        
        #### Append to dataframe the parameter with maximum auc
        optimized_thresholds = optimized_thresholds.append(cur_df[cur_df.auc == max(cur_df.auc.values)]).sort_index()
        
    return optimized_thresholds
    
def ComputeROCPointsTimeBin(marker_spline,time_within,parameter_name,time_start,time_end):
    '''
    Computes the fpr, tpr, and threshold points given the marker spline dataframe and parameter name
    
    Parameters
    ----------------------------
    marker_kinematics - pd.DataFrame() 
        Marker kinematics data frame
    time_within - float
        Time in hours in which the next measurement is made
    parameter_name - string
        name of the parameter whose ROC is to be plotted
    time_start - float
        Time in hours for the start of the bin
    time_end - float
        Time in hours for the end of the bin
    
    Returns
    ----------------------------
    fpr - np.array
        False positive points
    tpr - np.array
        True positive points
    thresholds - np.array
        Threshold points
    pos - int
        Number of positives
    neg - int
        Number of negative points
    '''
    
    #### Get false positive rates and true positive rates and corresponding thresholds
    fpr, tpr, thresholds = metrics.roc_curve(marker_spline.condition.values,marker_spline[parameter_name].values)
    
    #### Get number of positive points
    pos = len(marker_spline[marker_spline.condition == 1])
    
    #### Get number of negative points
    neg = len(marker_spline[marker_spline.condition == -1])
    
    return fpr, tpr, thresholds, pos, neg
    
def PlotMultipleROCConvexHull(marker_spline,time_within,parameter_list,time_bins):
    '''
    Plots the ROC Convex Hull for multiple parameters given the parameter list and time bins
    
    Parameters
    ----------------------------
    marker_kinematics - pd.DataFrame() 
        Marker kinematics data frame
    time_within - float
        Time in hours in which the next measurement is made
    parameter_list - list
        List of strings of the parameter name
    time_bins - list
        List of time bins
    
    Returns
    ----------------------------
    None
    '''
    #### Get marker_condition dataframe
    
    #### Group based on site_id and crack_id
    marker_kinematics_grouped = marker_spline.groupby(['site_id','crack_id'],as_index = False)
    marker_kinematics_with_condition = marker_kinematics_grouped.apply(MarkTrueConditions,time_within).reset_index()

    #### Get only values taken within specified time_within
    marker_kinematics_with_condition = SuccededWithinTime(marker_kinematics_with_condition,time_within)

    #### Get only non-zero displacement values
    marker_kinematics_with_condition = marker_kinematics_with_condition[marker_kinematics_with_condition.displacement != 0]
    
    #### Get only displacement values < 100 cm
    marker_kinematics_with_condition = marker_kinematics_with_condition[marker_kinematics_with_condition.displacement <= 100]
    
    #### Iterate through all time bins
    for time_bin in time_bins:
        
        #### Get only the values for the specified time bin
        cur_df = marker_kinematics_with_condition[np.logical_and(marker_kinematics_with_condition.time_interval <= time_bin[1],marker_kinematics_with_condition.time_interval >= time_bin[0])]

        #### Initialize points list
        points = np.array([[1,0]])
        
        #### Initialize figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
                
        #### Iterate through all parameters
        for parameter in parameter_list:
            
            #### Compute for fpr, tpr and thresholds
            fpr, tpr, thresholds, pos, neg = ComputeROCPointsTimeBin(cur_df,time_within,parameter,time_bin[0],time_bin[1])
            
            #### Plot current ROC curve
            ax.plot(fpr,tpr,'-',color = color_key[parameter],label = parameter)
            
            #### Add to points list
            points = np.append(points,zip(fpr,tpr),0)

        #### Compute for ROC convex hull
        ROCCH = ConvexHull(points).vertices
        
        #### Remove point zero
        ROCCH = np.append(ROCCH[list(ROCCH).index(0):],ROCCH[:list(ROCCH).index(0)])[1:]
        
        #### Convert points to array
        points = np.array(points)
        
        #### Plot Guess line
        ax.plot(np.linspace(0,1,100),np.linspace(0,1,100),'-',color = tableau20[6],label = 'Random Guess',lw = 1,zorder = 1)

        #### Plot Convex Hull
        ax.plot(points[ROCCH,0],points[ROCCH,1],'--',color = tableau20[8], label = 'ROCCH')
        
        #### Print time bin
        ax.text(0.97,0.11,'Time Bin {} to {}\nPos {} Neg {}'.format(time_bin[0],time_bin[1],pos,neg),transform = fig.transFigure,ha = 'right',fontsize = 10)    
        
        #### Plot legend
        ax.legend(bbox_to_anchor = (1.03,1.01),fontsize = 10)
        
        #### Set axis labels
        ax.set_xlabel('False Positive Rates',fontsize = 14)
        ax.set_ylabel('True Positive Rates',fontsize = 14)
        
        #### Set figure label
        fig.suptitle('ROC Convex Hull Plot',fontsize = 15)
        
        #### Set figsize
        fig.set_figheight(8)
        fig.set_figwidth(10)
        
        #### Set fig spacing
        fig.subplots_adjust(top = 0.92,bottom = 0.17,left = 0.07,right = 0.95)

        
        #### Set aspect as equal
        ax.set_aspect('equal')
    
        #### Set save path 1
        save_path = "{}\ROC\\ROCCH\\".format(data_path,time_bin[0],time_bin[1])
        if not os.path.exists(save_path+'\\'):
            os.makedirs(save_path+'\\')    
    
        #### Save fig
        plt.savefig('{}\ROCCH Time Bin {} to {}.png'.format(save_path,time_bin[0],time_bin[1]),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')

def PlotROCThresholdAverage(marker_spline,time_within,parameter,time_bin,num_split,num_threshold,return_to_bin = False,alpha = 0.05):
    '''
    Splits the given data frame to several sets then performs threshold averaging for the given parameter and time bin
    Then plots the corresponding ROC threshold average per specified step
    
    Parameters
    ----------------------------
    marker_kinematics - pd.DataFrame() 
        Marker kinematics data frame
    time_within - float
        Time in hours in which the next measurement is made
    parameter - string
        Name of parameter
    time_bin - tuple
        Tuple of the time bin
    num_split - int
        Number of samples to be made from original df
    num_threshold - int
        Number of threshold confidence interval computation
    return_to_bin - boleean
        Set to True if sample is returned to the population after picking
    alpha - float
        
    
    Returns
    --------------------------
    None - plotting function only
    
    '''
    #### Get marker_condition dataframe
    
    #### Group based on site_id and crack_id
    marker_kinematics_grouped = marker_spline.groupby(['site_id','crack_id'],as_index = False)
    marker_kinematics_with_condition = marker_kinematics_grouped.apply(MarkTrueConditions,time_within).reset_index()

    #### Get only values taken within specified time_within
    marker_kinematics_with_condition = SuccededWithinTime(marker_kinematics_with_condition,time_within)

    #### Get only non-zero displacement values
    marker_kinematics_with_condition = marker_kinematics_with_condition[marker_kinematics_with_condition.displacement != 0]
    
    #### Get only displacement values < 100 cm
    marker_kinematics_with_condition = marker_kinematics_with_condition[marker_kinematics_with_condition.displacement <= 100]

    #### Get only the values for the specified time bin
    marker_kinematics_with_condition = marker_kinematics_with_condition[np.logical_and(marker_kinematics_with_condition.time_interval <= time_bin[1],marker_kinematics_with_condition.time_interval >= time_bin[0])]
    
    #### Define sample size
    sample_size = int(len(marker_kinematics_with_condition)/num_split)

    #### Initialize the index array
    index_array = np.zeros((num_split,int(len(marker_kinematics_with_condition)/num_split)))
    indices = marker_kinematics_with_condition.index
    
    ### Initialize fpr, tpr, and threshold array
    fpr_list = []
    tpr_list = []
    threshold_list = []
    
    #### Initialize multiple ROC figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Generate random sampling
    for i in range(num_split):
        
        #### Get random sample from indices
        sample = np.random.choice(indices,sample_size)

        #### Store current sample to index array
        index_array[i] = sample
        
        if not return_to_bin:
            #### Remove the current sample to indices
            indices = np.array(list(set(indices)-set(sample)))
        
        #### Get current df
        current_df = marker_kinematics_with_condition.ix[sample]

        #### Compute current fpr, tpr and thresholds
        fpr, tpr, threshold, pos, neg = ComputeROCPointsTimeBin(current_df,time_within,parameter,time_bin[0],time_bin[1])
        
        #### Store results
        fpr_list.append([fpr])
        tpr_list.append([tpr])
        threshold_list.append([threshold])
        
        #### Plot current result
        ax.plot(fpr,tpr,'-')
    
    #### Plot Guess line
    ax.plot(np.linspace(0,1,100),np.linspace(0,1,100),'-',color = tableau20[6],label = 'Random Guess',lw = 1,zorder = 1)
    
    #### Print number of data points and time bin
    ax.text(0.975,0.025,'Time Bin {} to {}\nN {}\nSplit {}'.format(time_bin[0],time_bin[1],pos+neg,num_split),transform = fig.transFigure,ha = 'right',fontsize = 10)
        
    #### Set axis labels
    ax.set_xlabel('False Positive Rates',fontsize = 14)
    ax.set_ylabel('True Positive Rates',fontsize = 14)
    
    #### Set figure label
    fig.suptitle('Sampled ROC Plot for {} Threshold'.format(title_key[parameter]),fontsize = 15)
    
    #### Set figsize
    fig.set_figheight(8)
    fig.set_figwidth(8)
    
    #### Set aspect as equal
    ax.set_aspect('equal')
  
    #### Set save path 1
    save_path = "{}\ROC\ROC Averaging\\Time Bins\\{} to {}".format(data_path,time_bin[0],time_bin[1])
    if not os.path.exists(save_path+'\\'):
        os.makedirs(save_path+'\\')    

    #### Save fig
    plt.savefig('{}\{} Threshold {} N {} ret.png'.format(save_path,parameter,num_split,return_to_bin*1),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')

    #### Compute fpr, tpr, threshold, pos, neg for all samples
    fpr, tpr, threshold, pos, neg = ComputeROCPointsTimeBin(marker_kinematics_with_condition,time_within,parameter,time_bin[0],time_bin[1])
    
    #### Set fpr_points to plot based on given
    fpr_points = np.linspace(0,1.,num_threshold)

    ##### Start vertical averaging
    
    #### Initialize results container
    vert_delta = []
    
    #### Iterate to each fpr_points
    for fpr_point in fpr_points:
        
        #### Initialize tpr points
        tpr_points = []
        
        #### Get index of the cur fpr point in each array of fpr_list
        for cur_fpr_list,cur_tpr_list in zip(fpr_list,tpr_list):

            #### Get tpr point
            tpr_points.append(cur_tpr_list[0][cur_fpr_list[0] >= fpr_point][0])
            
        #### Compute confidence interval
        
        ### Set tpr points as array
        tpr_points = np.array(tpr_points)
        
        ### Get t crit
        t_crit = stats.t.ppf(1-alpha,num_split - 1)

        ### Get sample standard dev
        s = np.sqrt(1/float(num_split - 1)*np.sum(np.power(tpr_points - np.mean(tpr_points),2)))
        
        #### Get uncertainty
        delta = t_crit*s/np.sqrt(num_split)
        
        #### Save to results container
        vert_delta.append(delta)
        
    #### Get fpr and tpr points to plot for error bars
    
    #### Initialize results container
    tpr_plot = []
    fpr_plot = []
    
    #### Iterate to each fpr_points
    for fpr_point in fpr_points:
        
        #### Get nearest point in fpr and tpr values
        fpr_plot.append(fpr[fpr >= fpr_point][0])
        tpr_plot.append(tpr[fpr >= fpr_point][0])
    
    #### Initialize plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Plot average ROC
    ax.plot(fpr,tpr,'-',color = tableau20[0],label = 'Average')
    
    #### Plot error bars
    ax.errorbar(fpr_plot,tpr_plot,yerr = vert_delta,ecolor = tableau20[0],fmt = 'none',capthick = 1,elinewidth = 1,capsize = 4)

    #### Plot guess line
    ax.plot(np.linspace(0,1,100),np.linspace(0,1,100),'-',color = tableau20[6],label = 'Random Guess',lw = 1,zorder = 1)

    #### Set axis labels
    ax.set_xlabel('False Positive Rates',fontsize = 14)
    ax.set_ylabel('True Positive Rates',fontsize = 14)
    
    #### Draw legend
    ax.legend()
    
    #### Set figure label
    fig.suptitle('Average ROC Plot for {} Threshold'.format(title_key[parameter]),fontsize = 15)
    
    #### Set figsize
    fig.set_figheight(8)
    fig.set_figwidth(8)
    
    #### Set aspect as equal
    ax.set_aspect('equal')
  
    #### Set save path 1
    save_path = "{}\ROC\ROC Averaging\\Time Bins\\{} to {}\\Vertical Average\\".format(data_path,time_bin[0],time_bin[1])
    if not os.path.exists(save_path+'\\'):
        os.makedirs(save_path+'\\')    

    #### Save fig
    plt.savefig('{}\{} Threshold {} N {} ret.png'.format(save_path,parameter,num_split,return_to_bin*1),dpi = 320,facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
#    ####### Start confidence interval computation
#    
#    #### Get minimum from all maximum thresholds in threshold list
#    
#    #### Initialize minimum threshold
#    min_threshold = 1e100
#    
#    #### Iterate through every array in threshold list
#    for thresholds in threshold_list:
#        print np.max(thresholds)
#        #### Change current minimum
#        if np.max(thresholds) <= min_threshold:
#            min_threshold = np.max(thresholds)
#    
#    #### Get threshold array
#    thresholds = np.logspace(-10,np.log10(min_threshold),num_threshold)
#    print thresholds
#    
#    print fpr_list
#    print tpr_list
#    print threshold_list
#    #### Compute for fpr error and tpr error per threshold
#    
#    #### Iterate through all specified thresholds
#    for threshold in thresholds:
#
#        #### Initialize fpr and tpr points
#        fpr_points = []
#        tpr_points = []
#        
#        
#        #### Iterate through all list in threshold list
#        for i,cur_list in enumerate(threshold_list):
#            print threshold
#            #### Get fpr and tpr points
#            fpr_points.append(np.array(fpr_list[i][0])[cur_list[0] <= threshold][0])
#            tpr_points.append(np.array(tpr_list[i][0])[cur_list[0] <= threshold][0])
#        print tpr_points
#        print fpr_points
    
                       
                       















    
    
    
    
                                               