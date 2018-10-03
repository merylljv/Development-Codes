##### IMPORTANT matplotlib declarations must always be FIRST to make sure that matplotlib works with cron-based automation
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
from mpl_toolkits.mplot3d import Axes3D
plt.ion()

import os
import pandas as pd
import numpy as np
from datetime import date, time, datetime, timedelta
from scipy import stats
import sys

from scipy.interpolate import UnivariateSpline
from scipy.signal import gaussian
from scipy.ndimage import filters
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib.legend_handler import HandlerLine2D
from matplotlib import transforms
#### Include Analysis folder of updews-pycodes (HARD CODED!!)
path = os.path.abspath("D:\Meryll\updews-pycodes\Analysis")
if not path in sys.path:
    sys.path.insert(1,path)
del path 

import RealtimePlotter as rp
import ColumnPlotter as cp
import rtwindow as rtw
import querySenslopeDb as q
import genproc as gp
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from operator import is_not
from functools import partial
from PIL import Image



#### Global Parameters
data_path = os.path.dirname(os.path.realpath(__file__)) + ' 2'

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

cm = plt.get_cmap('plasma')
plasma = cm.colors


#### Hard Coded Parameters:
### Use 30 days for colpos and disp analysis or 3 days for velocity analysis
vel_event_window = pd.Timedelta(3,'D')
event_window = pd.Timedelta(30,'D')
colpos_interval_days = 3. ### in days
colpos_interval_hours = colpos_interval_days*24 ### in hours
window = 3 ### in days
transparency = 0.5 ### Set transparency for colpos plots
window = pd.Timedelta(window,'D')
threshold_file = 'threshold.csv'
threshold_type = 'on onset'

def remove_overlap(ranges):
    result = []
    current_start = pd.to_datetime(-1)
    current_stop = pd.to_datetime(-1 )

    for start, stop in sorted(ranges):
        start = pd.to_datetime(start)
        stop = pd.to_datetime(stop)
        if start > current_stop:
            # this segment starts after the last segment stops
            # just add a new segment
            result.append( (start, stop) )
            current_start, current_stop = start, stop
        else:
            # segments overlap, replace
            result[-1] = (current_start, stop)
            # current_start already guaranteed to be lower
            current_stop = max(current_stop, stop)

    return result

def moving_average(series,sigma = 3):
    b = gaussian(39,sigma)
    average = filters.convolve1d(series,b/b.sum())
    var = filters.convolve1d(np.power(series-average,2),b/b.sum())
    return average,var

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
    x_array = np.array(x_array)
    if not x_array.all():
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

def GoodnessOfSplineFit(x,y,sp):
    mean = np.mean(y)
    n = float(len(y))
    SS_tot = np.sum(np.power(y-mean,2))
    SS_res = np.sum(np.power(y-sp(x),2))
    coef_determination = 1 - SS_res/SS_tot
    RMSE = np.sqrt(SS_res/n)
    return SS_res,coef_determination,RMSE    

def CheckIfEventIsValid(candidate_triggers):
    query = "SELECT * FROM senslopedb.site_level_alert WHERE source = 'public' AND alert != 'A0' AND alert != 'A1' ORDER BY updateTS desc"
    public_alerts = q.GetDBDataFrame(query)
    query2 = "SELECT * FROM smsalerts WHERE (alertstat = 'invalid' or remarks LIKE '%invalid%') AND alertmsg LIKE '%sensor%'"
    invalid_sensor_alerts = q.GetDBDataFrame(query2)
    for ts_set,alertmsg in invalid_sensor_alerts[['ts_set','alertmsg']].values:
        site = alertmsg[alertmsg.index(':A')-3:alertmsg.index(':A')]

        ts_start = np.array(public_alerts[public_alerts.site == site].timestamp.values)
        ts_end = np.array(public_alerts[public_alerts.site == site].updateTS.values)
        

        ts_start_check = map(lambda x:(pd.to_datetime(x) - ts_set)/np.timedelta64(1,'D'), ts_start)
        ts_end_check = map(lambda x:(pd.to_datetime(x) - ts_set)/np.timedelta64(1,'D'), ts_end)
        
        
        checker = np.array(ts_start_check) * np.array(ts_end_check)
        try:
            index_invalid = list(checker).index(checker[checker < 0])
            mask = np.logical_not((candidate_triggers.site == site) & (candidate_triggers.timestamp >= ts_start[index_invalid]) & (candidate_triggers.timestamp <= ts_end[index_invalid]))
            candidate_triggers = candidate_triggers[mask]
        except:
            pass
        
    return candidate_triggers

def GetNodeLevelAlert(event_timestamp,sensor_column):
    query = "SELECT * FROM senslopedb.node_level_alert WHERE timestamp >= '{}' AND timestamp <= '{}' AND site = '{}'".format(event_timestamp[0].strftime('%Y-%m-%d %H:%M:%S'),event_timestamp[1].strftime('%Y-%m-%d %H:%M:%S'),sensor_column)
    return q.GetDBDataFrame(query)

def GetVelDF(disp):
    vel_df = disp.loc[:,['ts','id','vel_xz','vel_xy']]
    vel_df['vel'] = 0
    vel_df.loc[:,['vel']] = np.sqrt(np.power(vel_df.vel_xz,2) + np.power(vel_df.vel_xy,2)) * 100
    return vel_df[['ts','id','vel']].reset_index(drop = True).set_index('ts')
    
def compute_depth(colposdf):
    colposdf = colposdf.drop_duplicates()
    colposdf = colposdf.sort_values(['id'],ascending = False)
    cum_sum_yz = colposdf[['yz']].cumsum()
    colposdf['depth'] = cum_sum_yz.yz.values
    return np.round(colposdf,4)

def compute_cumdisp(colposdf):
    colposdf = colposdf.drop_duplicates()
    colposdf = colposdf.sort_values(['id'],ascending = False)
    cum_sum_xz = colposdf[['xz']].cumsum()
    cum_sum_xy = colposdf[['xy']].cumsum()
    colposdf['cum_xz'] = cum_sum_xz.xz.values
    colposdf['cum_xy'] = cum_sum_xy.xy.values
    return np.round(colposdf,4)

def set_ts_to_now(ts):
    if pd.to_datetime(ts) >= datetime.now():
        return datetime.now()
    else:
        return ts

def SignalToNoiseRatio(disp_id):
    mean_xy = np.mean(np.abs(disp_id.xy.values))
    mean_xz = np.mean(np.abs(disp_id.xz.values))
    std_xy = np.std(np.abs(disp_id.xy.values))
    std_xz = np.std(np.abs(disp_id.xz.values))
    
    snr_xz = mean_xz/std_xz
    snr_xy = mean_xy/std_xy
    return snr_xz, snr_xy

def site_events_to_plot(candidate_triggers):
    df = pd.DataFrame()    
    timestamp_start = candidate_triggers['timestamp_start'].values
    timestamp_end = candidate_triggers['timestamp_end'].values
    timestamp_end = map(lambda x:set_ts_to_now(x),timestamp_end)
    timestamps = remove_overlap(zip(timestamp_start,timestamp_end))
    df['timestamp'] = timestamps
    df['site'] = candidate_triggers.site.values[0]
    return df

def sensor_columns_to_plot(site_events):
    for timestamp,site in site_events[['timestamp','site']].values:
        query = "SELECT DISTINCT site, id FROM senslopedb.node_level_alert WHERE timestamp >= '{}' AND timestamp <= '{}' AND site LIKE '{}%'".format(timestamp[0].strftime("%Y-%m-%d %H:%M:%S"),timestamp[1].strftime("%Y-%m-%d %H:%M:%S"),site)
        sensor_columns = q.GetDBDataFrame(query)
        site_events.loc[(site_events.timestamp == timestamp)&(site_events.site == site),['sensor_columns']] = ', '.join(np.unique(sensor_columns['site'].values))
        nodes = ''
        for column in np.unique(sensor_columns['site'].values):
            nodes = nodes + '[' + ','.join(map(lambda x:str(x),np.unique(sensor_columns[sensor_columns.site == column].id.values))) + ']'
            if column != np.unique(sensor_columns['site'].values)[-1]:
                nodes = nodes + ', '
        site_events.loc[(site_events.timestamp == timestamp)&(site_events.site == site),['nodes']] = nodes
        
    return site_events

def set_zero_disp(colposdf):
    initial_ts = min(colposdf['ts'].values)
    colposdf['xz'] = colposdf['xz'].values - colposdf[colposdf.ts == initial_ts]['xz'].values[0]
    colposdf['xy'] = colposdf['xy'].values - colposdf[colposdf.ts == initial_ts]['xy'].values[0]
    
    return colposdf

def SubsurfaceValidEvents(vel = False):
    #### This function gets all valid L2/L3 subsurface events then outputs the appropriate start and end timestamp for each site and relevant sensor columns
    
    #### Obtain from site level alerts all L2/L3 candidate triggers
    query = "SELECT * FROM senslopedb.site_level_alert WHERE source = 'internal' AND alert LIKE '%s%' AND alert not LIKE '%s0%' AND alert not LIKE '%ND%' ORDER BY updateTS desc"
    candidate_triggers = q.GetDBDataFrame(query)
    
    #### Filter invalid alerts
    candidate_triggers = CheckIfEventIsValid(candidate_triggers)
    
    #### Set initial and final plotting timestamp depending on mode
    if not(vel):
        candidate_triggers['timestamp_start'] = map(lambda x:x - event_window,candidate_triggers['timestamp'])
        candidate_triggers['timestamp_end'] = map(lambda x:x + event_window,candidate_triggers['updateTS'])
    else:
        candidate_triggers['timestamp_start'] = map(lambda x:x - vel_event_window - window,candidate_triggers['timestamp'])
        candidate_triggers['timestamp_end'] = map(lambda x:x + vel_event_window,candidate_triggers['updateTS'])

    
    #### Remove overlaps and merge timestamps per site
    candidate_triggers_group = candidate_triggers.groupby(['site'],as_index = False)
    site_events = candidate_triggers_group.apply(site_events_to_plot)
    
    #### Determine columns to plot and nodes to check
    site_events['sensor_columns'] = None
    site_events['nodes'] = None
    return sensor_columns_to_plot(site_events).reset_index()[['timestamp','site','sensor_columns','nodes']]

def SubsurfaceValidEvents2(vel = False):
    '''
    Gets all valid L2/L3 subsurface events from site_level_alert database while checking its validity from gsm ack
    
    Parameters
    ----------------------
    None
    
    Optional Parameters
    ----------------------
    vel - boolean
        specify if velocity event window will be used
    
    Returns
    ----------------------
        subsurface_valid_events - pd.DataFrame()
            data frame of validated L2/L3 events with corresponding window specified by the window parameter
    '''
    #### Obtain from site level alerts all L2/L3 candidate triggers
    query = "SELECT * FROM senslopedb.site_level_alert WHERE source = 'sensor' and alert != 'L0' and alert != 'ND' ORDER BY updateTS desc"
    candidate_triggers = q.GetDBDataFrame(query)
    
    #### Filter invalid alerts
    candidate_triggers = CheckIfEventIsValid(candidate_triggers)
    
    #### Set initial and final plotting timestamp depending on mode
    if not(vel):
        candidate_triggers['timestamp_start'] = map(lambda x:x - event_window,candidate_triggers['timestamp'])
        candidate_triggers['timestamp_end'] = map(lambda x:x + event_window,candidate_triggers['updateTS'])
    else:
        candidate_triggers['timestamp_start'] = map(lambda x:x - vel_event_window - window,candidate_triggers['timestamp'])
        candidate_triggers['timestamp_end'] = map(lambda x:x + vel_event_window,candidate_triggers['updateTS'])

    
    #### Remove overlaps and merge timestamps per site
    candidate_triggers_group = candidate_triggers.groupby(['site'],as_index = False)
    site_events = candidate_triggers_group.apply(site_events_to_plot)
    
    #### Determine columns to plot and nodes to check
    site_events['sensor_columns'] = None
    site_events['nodes'] = None
    return sensor_columns_to_plot(site_events).reset_index()[['timestamp','site','sensor_columns','nodes']]

def GetDispAndColPosDataFrame(event_timestamp,sensor_column,compute_vel = False):
    #### Get all required parameters from realtime plotter
    col = q.GetSensorList(sensor_column)
    window, config = rtw.getwindow(pd.to_datetime(event_timestamp[-1]))
    window.start = pd.to_datetime(event_timestamp[0])
    window.offsetstart = window.start - timedelta(days=(config.io.num_roll_window_ops*window.numpts-1)/48.)
    config.io.col_pos_interval = str(int(colpos_interval_hours)) + 'h'
    config.io.num_col_pos = int((window.end - window.start).days/colpos_interval_days + 1)
    monitoring = gp.genproc(col[0], window, config, config.io.column_fix,comp_vel = compute_vel)

    #### Get colname, num nodes and segment length
    num_nodes = monitoring.colprops.nos
    seg_len = monitoring.colprops.seglen
    
    if compute_vel:
        monitoring_vel = monitoring.disp_vel.reset_index()[['ts', 'id', 'depth', 'xz', 'xy', 'vel_xz', 'vel_xy']]
    else:
        monitoring_vel = monitoring.disp_vel.reset_index()[['ts', 'id', 'depth', 'xz', 'xy']]
    monitoring_vel = monitoring_vel.loc[(monitoring_vel.ts >= window.start)&(monitoring_vel.ts <= window.end)]
    
    colposdf = cp.compute_colpos(window, config, monitoring_vel, num_nodes, seg_len)
    
    #### Recomputing depth
    colposdf['yz'] = np.sqrt(seg_len**2 - np.power(colposdf['xz'],2) - np.power(colposdf['xy'],2))
    colposdf_ts = colposdf.groupby('ts',as_index = False)
    colposdf = colposdf_ts.apply(compute_depth)
    colposdf['depth'] = map(lambda x:x - max(colposdf.depth.values),colposdf['depth'])
    
    return monitoring_vel.reset_index(drop = True),colposdf[['ts','id','depth','xy','xz']].reset_index(drop = True),sensor_column

def GetDispDataFrame(event_timestamp,sensor_column,compute_vel = False):
    #### Get all required parameters from realtime plotter
    col = q.GetSensorList(sensor_column)
    window, config = rtw.getwindow(pd.to_datetime(event_timestamp[-1]))
    window.start = pd.to_datetime(event_timestamp[0])
    window.offsetstart = window.start - timedelta(days=(config.io.num_roll_window_ops*window.numpts-1)/48.)
    config.io.col_pos_interval = str(int(colpos_interval_hours)) + 'h'
    config.io.num_col_pos = int((window.end - window.start).days/colpos_interval_days + 1)
    monitoring = gp.genproc(col[0], window, config, config.io.column_fix,comp_vel = compute_vel)

    #### Get colname, num nodes and segment length
#    num_nodes = monitoring.colprops.nos
#    seg_len = monitoring.colprops.seglen
    
    if compute_vel:
        monitoring_vel = monitoring.disp_vel.reset_index()[['ts', 'id', 'depth', 'xz', 'xy', 'vel_xz', 'vel_xy']]
    else:
        monitoring_vel = monitoring.disp_vel.reset_index()[['ts', 'id', 'depth', 'xz', 'xy']]
    monitoring_vel = monitoring_vel.loc[(monitoring_vel.ts >= window.start)&(monitoring_vel.ts <= window.end)]
    
#    colposdf = cp.compute_colpos(window, config, monitoring_vel, num_nodes, seg_len)
#    
#    #### Recomputing depth
#    colposdf['yz'] = np.sqrt(seg_len**2 - np.power(colposdf['xz'],2) - np.power(colposdf['xy'],2))
#    colposdf_ts = colposdf.groupby('ts',as_index = False)
#    colposdf = colposdf_ts.apply(compute_depth)
#    colposdf['depth'] = map(lambda x:x - max(colposdf.depth.values),colposdf['depth'])
    
    return monitoring_vel.reset_index(drop = True),sensor_column


def ComputeCumShear(disp_ts):
    cumsheardf = pd.DataFrame(columns = ['ts','cumshear','cum_xz','cum_xy'])
    cumsheardf.ix['ts'] = disp_ts.ts.values[0]
    
    cum_xz = np.sum(disp_ts.xz.values)
    cum_xy = np.sum(disp_ts.xy.values)
    sum_cum = np.sqrt(np.square(cum_xz) + np.square(cum_xy))
    cumsheardf['cumshear'] = sum_cum
    cumsheardf['cum_xz'] = cum_xz
    cumsheardf['cum_xy'] = cum_xy
    
    return cumsheardf     

def PlotIncrementalDisplacement(colposdf,sensor_column,zeroed = False,zoomin=False):
    #### Set figure and subplots
    fig = plt.figure()
    ax_xz = fig.add_subplot(121)
    ax_xy = fig.add_subplot(122, sharey = ax_xz)
    
    #### Ensure non repeating colors and color scheme = plasma
    ax_xz=cp.nonrepeat_colors(ax_xz,len(set(colposdf.ts.values)),color='plasma')
    ax_xy=cp.nonrepeat_colors(ax_xy,len(set(colposdf.ts.values)),color='plasma')
    
    #### Set grid
    ax_xz.grid()
    ax_xy.grid()
    
    if zeroed == True:
        colposdf_id = colposdf.groupby('id',as_index = False)
        colposdf = colposdf_id.apply(set_zero_disp)
        zeroed = 'zeroed '
    else:
        zeroed = ''
    
    for ts in np.unique(colposdf.ts.values):
        #### Get df for a specific timestamp
        ts = pd.to_datetime(ts)
        cur_df = colposdf[colposdf.ts == ts]
        
        #### Obtain values to plot
        cur_depth = cur_df['depth'].values
        cur_xz = cur_df['xz'].values * 1000
        cur_xy = cur_df['xy'].values * 1000
        
        #### Plot values write label to xz plot only
        cur_plot = ax_xz.plot(cur_xz,cur_depth,'.-',lw = 1.25,markersize = 10,label = ts.strftime("%d %b '%y %H:%M"))
        ax_xy.plot(cur_xy,cur_depth,'.-',lw = 1.25,markersize = 10)
        
        
        #### Contain all plots in 'plots' variable
        try:
            plots = plots + cur_plot
        except:
            plots = cur_plot
    
    #### Set fontsize and rotate ticks for x axis
    for tick in ax_xz.xaxis.get_minor_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
        
    for tick in ax_xy.xaxis.get_minor_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
   
    for tick in ax_xz.xaxis.get_major_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
        
    for tick in ax_xy.xaxis.get_major_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
    
    #### Plot the legends and labels
    labels = [l.get_label() for l in plots]
    fig.legend(plots,labels,loc = 'center right',fontsize = 12)
    if zeroed == 'zeroed ':
        fig.suptitle("Incremental Displacement Plot for {} (zeroed)".format(sensor_column.upper()),fontsize = 15)
    else:
        fig.suptitle("Incremental Displacement Plot for {}".format(sensor_column.upper()),fontsize = 15)
        
    ax_xz.set_ylabel('Depth (meters)',fontsize = 14)
    ax_xz.set_xlabel('Displacement (mm)\ndownslope direction',fontsize = 14)
    ax_xy.set_xlabel('Displacement (mm)\nacross slope direction',fontsize = 14)
    
    #### Set xlim and ylim
    depth_range = abs(max(colposdf.depth.values) - min(colposdf.depth.values))
    xz_range = abs(max(colposdf.xz.values)- min(colposdf.xz.values))
    xy_range = abs(max(colposdf.xy.values)- min(colposdf.xy.values))
    
    ax_xz.set_ylim([min(colposdf.depth.values)-0.05*depth_range,max(colposdf.depth.values)+0.05*depth_range])
    
    if zoomin:
        ax_xz.set_xlim(np.array([min(colposdf.xz.values)-0.05*xz_range,max(colposdf.xz.values)+0.05*xz_range])*1000)
        ax_xy.set_xlim(np.array([min(colposdf.xy.values)-0.05*xy_range,max(colposdf.xy.values)+0.05*xy_range])*1000)
        zoomin = 'zoomin '
    else:
    #### Set automatic adjustement of x-axis limits
        if xz_range > xy_range:
            total_range = xz_range*1.1
            ax_xz.set_xlim(np.array([min(colposdf.xz.values)-0.05*xz_range,max(colposdf.xz.values)+0.05*xz_range])*1000)
            
            xy_center = 0.5*(max(colposdf.xy.values) + min(colposdf.xy.values))
            ax_xy.set_xlim(np.array([xy_center-total_range*0.5,xy_center + total_range * 0.5])*1000)
        else:
            total_range = xy_range*1.1
            ax_xy.set_xlim(np.array([min(colposdf.xy.values)-0.05*xy_range,max(colposdf.xy.values)+0.05*xy_range])*1000)
            
            xz_center = 0.5*(max(colposdf.xz.values) + min(colposdf.xz.values))
            ax_xz.set_xlim(np.array([xz_center-total_range*0.5,xz_center + total_range * 0.5])*1000)
        zoomin = ''
    
    #### Set fig size, borders and spacing
    fig.set_figheight(9)
    fig.set_figwidth(11.5)
    fig.subplots_adjust(right = 0.795,top = 0.925,left = 0.100)
    
    #### Set save path
    save_path = "{}/{}/Event {} to {}/IncDisp".format(data_path,sensor_column,pd.to_datetime(min(colposdf.ts.values)).strftime("%d %b %y"),pd.to_datetime(max(colposdf.ts.values)).strftime("%d %b %y"))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/IncDisp {}{}{} {} to {}.png'.format(save_path,zeroed,zoomin,sensor_column,pd.to_datetime(min(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M")),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')

def PlotCumulativeDisplacementPlot(colposdf,sensor_column,zeroed = False):
    #### Set figure and subplots
    fig = plt.figure()
    ax_xz = fig.add_subplot(121)
    ax_xy = fig.add_subplot(122,sharey = ax_xz)
    
    #### Ensure non repeating colors and color scheme = plasma
    ax_xz=cp.nonrepeat_colors(ax_xz,len(set(colposdf.ts.values)),color='plasma')
    ax_xy=cp.nonrepeat_colors(ax_xy,len(set(colposdf.ts.values)),color='plasma')
    
    #### Set grid
    ax_xz.grid()
    ax_xy.grid()
    
    if zeroed == True:
        colposdf_id = colposdf.groupby('id',as_index = False)
        colposdf = colposdf_id.apply(set_zero_disp)
        zeroed = 'zeroed '
    else:
        zeroed = ''
        
    #### Compute for cumulative displacement
    colposdf_ts = colposdf.groupby('ts',as_index = False)
    colposdf = colposdf_ts.apply(compute_cumdisp)
    
    for ts in np.unique(colposdf.ts.values):
        #### Get df for a specific timestamp
        ts = pd.to_datetime(ts)
        cur_df = colposdf[colposdf.ts == ts]
        
        #### Obtain values to plot
        cur_depth = cur_df['depth'].values
        cur_xz = cur_df['cum_xz'].values * 1000
        cur_xy = cur_df['cum_xy'].values * 1000
        
        #### Plot values write label to xz plot only
        cur_plot = ax_xz.plot(cur_xz,cur_depth,'.-',lw = 1.25,markersize = 10,label = ts.strftime("%d %b '%y %H:%M"),alpha = transparency)
        ax_xy.plot(cur_xy,cur_depth,'.-',lw = 1.25,markersize = 10,alpha = transparency)
        
        
        #### Contain all plots in 'plots' variable
        try:
            plots = plots + cur_plot
        except:
            plots = cur_plot
    
    #### Set fontsize and rotate ticks for x axis
    for tick in ax_xz.xaxis.get_minor_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
        
    for tick in ax_xy.xaxis.get_minor_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
   
    for tick in ax_xz.xaxis.get_major_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
        
    for tick in ax_xy.xaxis.get_major_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
    
    #### Plot the legends and labels
    labels = [l.get_label() for l in plots]
    fig.legend(plots,labels,loc = 'center right',fontsize = 12)
    if zeroed == 'zeroed ':
        fig.suptitle("Cumulative Deflection Plot for {} (zeroed)".format(sensor_column.upper()),fontsize = 15)
    else:
        fig.suptitle("Cumulative Deflection Plot for {}".format(sensor_column.upper()),fontsize = 15)
        
    ax_xz.set_ylabel('Depth (meters)',fontsize = 14)
    ax_xz.set_xlabel('Cumulative Deflection (mm)\ndownslope direction',fontsize = 14)
    ax_xy.set_xlabel('Cumulative Deflection (mm)\nacross slope direction',fontsize = 14)
    
    #### Set xlim and ylim
    depth_range = abs(max(colposdf.depth.values) - min(colposdf.depth.values))
    cum_xz_range = abs(max(colposdf.cum_xz.values)- min(colposdf.cum_xz.values))
    cum_xy_range = abs(max(colposdf.cum_xy.values)- min(colposdf.cum_xy.values))
    
    ax_xz.set_ylim([min(colposdf.depth.values)-0.05*depth_range,max(colposdf.depth.values)+0.05*depth_range])
    
    #### Set automatic adjustement of x-axis limits
    if cum_xz_range > cum_xy_range:
        total_range = cum_xz_range*1.1
        ax_xz.set_xlim(np.array([min(colposdf.cum_xz.values)-0.05*cum_xz_range,max(colposdf.cum_xz.values)+0.05*cum_xz_range])*1000)
        
        cum_xy_center = 0.5*(max(colposdf.cum_xy.values) + min(colposdf.cum_xy.values))
        ax_xy.set_xlim(np.array([cum_xy_center-total_range*0.5,cum_xy_center + total_range * 0.5])*1000)
    else:
        total_range = cum_xy_range*1.1
        ax_xy.set_xlim(np.array([min(colposdf.cum_xy.values)-0.05*cum_xy_range,max(colposdf.cum_xy.values)+0.05*cum_xy_range])*1000)
        
        cum_xz_center = 0.5*(max(colposdf.cum_xz.values) + min(colposdf.cum_xz.values))
        ax_xz.set_xlim(np.array([cum_xz_center-total_range*0.5,cum_xz_center + total_range * 0.5])*1000)
    

    #### Set fig size, borders and spacing
    fig.set_figheight(9)
    fig.set_figwidth(11.5)
    fig.subplots_adjust(right = 0.795,top = 0.925,left = 0.100)
    
    #### Set save path
    save_path = "{}/{}/Event {} to {}/ColPos".format(data_path,sensor_column,pd.to_datetime(min(colposdf.ts.values)).strftime("%d %b %y"),pd.to_datetime(max(colposdf.ts.values)).strftime("%d %b %y"))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/CumDef {}{} {} to {}.png'.format(save_path,zeroed,sensor_column,pd.to_datetime(min(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M")),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')

def PlotALL(colposdf,sensor_name):
    PlotIncrementalDisplacement(colposdf,sensor_name)
    PlotIncrementalDisplacement(colposdf,sensor_name,zoomin = True)
    PlotIncrementalDisplacement(colposdf,sensor_name,zeroed = True)
    PlotIncrementalDisplacement(colposdf,sensor_name,zeroed = True,zoomin = True)
    PlotCumulativeDisplacementPlot(colposdf,sensor_name,zeroed = False)
    PlotCumulativeDisplacementPlot(colposdf,sensor_name,zeroed = True)

    
def PlotALLIdentifiedEvents():
    events_list = SubsurfaceValidEvents2()[28:]
    for timestamp in events_list.timestamp.values:
        sensor_columns = events_list[events_list.timestamp == timestamp].sensor_columns.values[0].split(', ')
        for sensor in sensor_columns:
            print "Getting data from event {} to {} for site {} sensor column {}".format(timestamp[0].strftime("%d %b '%y %H:%S"),timestamp[1].strftime("%d %b '%y %H:%S"),events_list[events_list.timestamp == timestamp].site.values[0].upper(),sensor.upper())
            disp,colposdf,name = GetDispAndColPosDataFrame(timestamp,sensor)
            print "Plotting data..."
            PlotALL(colposdf,name)
            for i in range(6):
                plt.close()
            print "...Done!"

def PlotCumShearDisplacementSingle(disp,name,nodes):
    #### Select only relevant nodes
    mask = np.zeros(len(disp.id.values))
    for values in nodes:
        mask = np.logical_or(mask,disp.id.values == values)
    disp = disp[mask]
    
    #### Set initial displacements to zero
    disp_id = disp.groupby('id',as_index = False)
    disp = disp_id.apply(set_zero_disp)    
    
    #### Compute Shear Displacements
    disp_ts = disp.groupby('ts',as_index = True)
    cumsheardf = disp_ts.apply(ComputeCumShear).reset_index(drop = True)
    
    #### Plot computed values
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cumsheardf.ts.values,cumsheardf.cumshear.values*100,color = tableau20[0],lw=2,label = "Nodes {}".format(' '.join(map(lambda x:str(x),nodes))))
    ax.grid()

    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter('%m/%d/%y'))
    
    #### Rotate x axis ticks
    for tick in ax.xaxis.get_minor_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
   
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
    
    #### Set xlim and ylim
    y_range = max(cumsheardf.cumshear.values*100) - min(cumsheardf.cumshear.values*100)
    ax.set_ylim(min(cumsheardf.cumshear.values*100) - 0.05*y_range,max(cumsheardf.cumshear.values*100)+0.05*y_range)
    
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 14)
    ax.set_ylabel('Cumulative Shear Displacement (cm)',fontsize = 14)
    fig.suptitle("Displacement vs. Time Plot for Site {} Sensor {}".format(name[:3].upper(),name.upper()),fontsize = 15)
    legend = ax.legend(loc = 'upper left',fontsize = 12)
    legend.get_frame().set_linewidth(0)   
    legend.get_frame().set_alpha(0)
    
    #### Set fig size, borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.15)
    
    #### Set save path
    save_path = "{}/{}/Event {} to {}/CumShearDisplacement".format(data_path,name,pd.to_datetime(min(disp.ts.values)).strftime("%d %b %y"),pd.to_datetime(max(disp.ts.values)).strftime("%d %b %y"))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/DispvTime {} {} to {} {}.png'.format(save_path,name,pd.to_datetime(min(disp.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(disp.ts.values)).strftime("%Y-%m-%d_%H-%M"),'.'.join(map(lambda x:str(x),nodes))),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    #### Set major axis label for interactive mode
    ax.xaxis.set_major_formatter(md.DateFormatter('%m/%d/%y %H:%M'))
    
    
def PlotCumShearDisplacementMultiple(disp,name,nodes):

    #### Set initial displacements to zero
    disp_id = disp.groupby('id',as_index = False)
    disp = disp_id.apply(set_zero_disp)    
        
    #### Set figure parameters    
    fig = plt.figure()    
    ax = fig.add_subplot(111)
    ax.grid()
    plot_num = 1
    min_y = 10000
    max_y = -1
    for set_nodes in nodes:
        #### Select only relevant data
        mask = np.zeros(len(disp.id.values))
        for values in set_nodes:
            mask = np.logical_or(mask,disp.id.values == values)
        cur_data = disp[mask]
        
        #### Compute Shear Displacements
        cur_ts = cur_data.groupby('ts',as_index = True)
        cur_cum_shear = cur_ts.apply(ComputeCumShear).reset_index(drop = True)
        
        #### Plot current values
        ax.plot(cur_cum_shear.ts.values,cur_cum_shear.cumshear.values*100,lw = 2,color = tableau20[(plot_num-1)*2%19],label = "Nodes "+' '.join(map(lambda x:str(x),set_nodes)))
        
        #### Set maximum values of y (for adjustment of ylim)        
        if min_y > min(cur_cum_shear.cumshear.values*100):
            min_y = min(cur_cum_shear.cumshear.values*100)
        if max_y < max(cur_cum_shear.cumshear.values*100):
            max_y = max(cur_cum_shear.cumshear.values*100)
        
        #### Increment plot num
        plot_num += 1
    
    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter('%m/%d/%y'))
    
    #### Rotate x axis ticks
    for tick in ax.xaxis.get_minor_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
   
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
        
    #### Set xlim and ylim
    y_range = max_y - min_y
    ax.set_ylim(min_y - 0.05*y_range,max_y+0.05*y_range)
    
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 14)
    ax.set_ylabel('Cumulative Shear Displacement (cm)',fontsize = 14)
    fig.suptitle("Displacement vs. Time Plot for Site {} Sensor {}".format(name[:3].upper(),name.upper()),fontsize = 15)
    legend = ax.legend(loc = 'upper left',fontsize = 12)
    legend.get_frame().set_linewidth(0)   
    legend.get_frame().set_alpha(0)
    
    
    #### Set fig size, borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.15)
    
    #### Set save path
    save_path = "{}/{}/Event {} to {}/CumShearDisplacement".format(data_path,name,pd.to_datetime(min(disp.ts.values)).strftime("%d %b %y"),pd.to_datetime(max(disp.ts.values)).strftime("%d %b %y"))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Set filename tag
    tag = ''    
    for node in nodes:
        if len(node)>1:
            tag = tag + "{}to{}.".format(node[0],node[-1])
        else:
            tag = tag + "{}.".format(node[0])
            
    
    #### Save figure
    plt.savefig('{}/DispvTime Multiple {} {} to {} {}.png'.format(save_path,name,pd.to_datetime(min(disp.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(disp.ts.values)).strftime("%Y-%m-%d_%H-%M"),tag),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    #### Set major axis label for interactive mode
    ax.xaxis.set_major_formatter(md.DateFormatter('%m/%d/%y %H:%M'))

def PlotCumShearMutipleSites(disp_list,nodes_list,name_list):
    #### Set figure parameters    
    fig = plt.figure()    
    ax = fig.add_subplot(111)
    ax.grid()
    plot_num = 1
    min_y = 10000
    max_y = -1
    save_label = ''
    
    #### Plot all displacements to the same axis
    for i in range(len(disp_list)):
        disp = disp_list[i]
        nodes = nodes_list[i]
        
        #### Update save label
        save_label = save_label + name_list[i]
        save_label = ' ' + ' '.join(map(lambda x: str(x),nodes))
        
        #### Set initial displacements to zero
        disp_id = disp.groupby('id',as_index = False)
        disp = disp_id.apply(set_zero_disp)
        
        #### Select only relevant nodes
        mask = np.zeros(len(disp.id.values))
        for values in nodes:
            mask = np.logical_or(mask,disp.id.values == values)
        disp = disp[mask]
        
        #### Compute Shear Displacements
        disp_ts = disp.groupby('ts',as_index = True)
        cumsheardf = disp_ts.apply(ComputeCumShear).reset_index(drop = True)
        
        #### Plot computed values
        ax.plot(cumsheardf.ts.values,cumsheardf.cumshear.values*100,color = tableau20[(plot_num-1)*2],lw=2,label = "{} {}".format(name_list[i].upper(),' '.join(map(lambda x:str(x),nodes))))
        
        #### Set maximum values of y (for adjustment of ylim)        
        if min_y > min(cumsheardf.cumshear.values*100):
            min_y = min(cumsheardf.cumshear.values*100)
        if max_y < max(cumsheardf.cumshear.values*100):
            max_y = max(cumsheardf.cumshear.values*100)
        
        #### Increment plot num
        plot_num += 1
    
    #### Set datetime format for x axis
    ax.xaxis.set_major_formatter(md.DateFormatter('%m/%d/%y'))
    
    #### Rotate x axis ticks
    for tick in ax.xaxis.get_minor_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
   
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
        
    #### Set xlim and ylim
    y_range = max_y - min_y
    ax.set_ylim(min_y - 0.05*y_range,max_y+0.05*y_range)
    
    
    #### Set axis labels and legend
    ax.set_xlabel('Date',fontsize = 14)
    ax.set_ylabel('Cumulative Shear Displacement (cm)',fontsize = 14)
    fig.suptitle("Displacement vs. Time Plot for Site {} Sensors {}".format(name_list[0][:3].upper(),' '.join(name_list).upper()),fontsize = 15)
    legend = ax.legend(loc = 'upper left',fontsize = 12)
    legend.get_frame().set_linewidth(0)   
    legend.get_frame().set_alpha(0)
    
    
    #### Set fig size, borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.15)

    #### Set save path
    save_path = "{}".format(data_path)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
        
    
    #### Save figure
    plt.savefig('{}/DispvTime Multiple Sites {} {} to {}.png'.format(save_path,save_label,pd.to_datetime(min(disp.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(disp.ts.values)).strftime("%Y-%m-%d_%H-%M")),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    #### Set major axis label for interactive mode
    ax.xaxis.set_major_formatter(md.DateFormatter('%m/%d/%y %H:%M'))
    
def GetCumShearDF(disp,nodes):
    #### Select only relevant nodes
    mask = np.zeros(len(disp.id.values))
    for values in nodes:
        mask = np.logical_or(mask,disp.id.values == values)
    disp = disp[mask]
    
    #### Set initial displacements to zero
    disp_id = disp.groupby('id',as_index = False)
    disp = disp_id.apply(set_zero_disp)    
    
    #### Compute Shear Displacements
    disp_ts = disp.groupby('ts',as_index = True)
    return disp_ts.apply(ComputeCumShear).reset_index(drop = True).set_index('ts')
    

def PlotInterpolation(disp,nodes,name,window):
    #### Get cumshear df
    cumsheardf = GetCumShearDF(disp,nodes)
    
    #### Compute for time delta values
    cumsheardf['time'] = map(lambda x: x / np.timedelta64(1,'D'),cumsheardf.index - cumsheardf.index[0])
    
    #### Convert displacement to centimeters
    cumsheardf['cumshear'] = cumsheardf.cumshear.apply(lambda x:x*100)
    
    
    #### Get last timestamp in df
    last_ts = cumsheardf.index[-1]        
    
    #### Set figure number    
    fig_num = 1
    
    #### Set bounds of slice df according to window
    for ts_start in cumsheardf[cumsheardf.index <= last_ts - window].index:

#        #### Resume run
#        if fig_num <= 5752:
#            print "Skipping frame {:04d}".format(fig_num)
#            fig_num+=1
#            continue
        
        #### Set end ts according to window        
        ts_end = ts_start + window
        
        #### Slice df
        slicedf = cumsheardf[ts_start:ts_end]
        
        #### Get time and displacement values
        time_delta = slicedf.time.values
        disp = slicedf.cumshear.values
        
        #### Commence interpolation
        try:
            #### Take the gaussian average of data points and its variance
            _,var = moving_average(disp)
            sp = UnivariateSpline(time_delta,disp,w=1/np.sqrt(var))
            
            #### Use 10000 points for interpolation
            time_int = np.linspace(time_delta[0],time_delta[-1],10000)
            
            #### Spline interpolation values    
            disp_int = sp(time_int)
            vel_int = sp.derivative(n=1)(time_int)
            acc_int = sp.derivative(n=2)(time_int)
            
            #### Compute for goodness of fit values
            SS_res, r2, RMSE = GoodnessOfSplineFit(time_delta,disp,sp)            
            
        except:
            print "Interpolation Error {}".format(pd.to_datetime(str(time_delta[-1])).strftime("%m/%d/%Y %H:%M"))
            disp_int = np.ones(len(time_int))*np.nan
            vel_int = np.ones(len(time_int))*np.nan
            acc_int = np.ones(len(time_int))*np.nan
        
        #### Commence plotting

        #### Initialize figure parameters
        fig = plt.figure()
        int_ax = fig.add_subplot(211)
        vel_ax = fig.add_subplot(223)
        acc_ax = fig.add_subplot(224)
        
        #### Plot Grid
        int_ax.grid()
        vel_ax.grid()
        acc_ax.grid()
        
        #### Compute corresponding datetime array
        datetime_array = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_delta)        
        datetime_int = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_int)
        #### Plot computed values
        
        ### Plot data and interpolated values
        int_ax.plot(datetime_array,disp,'.',color = tableau20[0],label = 'Data')
        int_ax.plot(datetime_int,disp_int,'-',color = tableau20[12],label = 'Interpolation')
        
        ### Create inset axes
        inset_ax = inset_axes(int_ax,width = "14%",height = "35%",loc = 3)
        
        ### Plot current range to the inset axes
        inset_ax.plot(cumsheardf.index,cumsheardf.cumshear.values)
        inset_ax.axvspan(ts_start,ts_end,facecolor = tableau20[2],alpha = 0.5)
        
        ### Hide ticks and labels for inset plot
        inset_ax.tick_params(top = 'off',left = 'off',bottom = 'off',right = 'off',labelleft = 'off',labelbottom = 'off')        
        
        ### Set transparency for inset plot
        inset_ax.patch.set_alpha(0.5)        
        
        ### Plot velocity and acceleration
        vel_ax.plot(time_int,vel_int,'-',color = tableau20[4])
        acc_ax.plot(time_int,acc_int,'-',color = tableau20[6])

        #### Set datetime format for x axis
        int_ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))

        #### Set ylim for plots
        disp_max = max(np.concatenate((disp,disp_int)))
        disp_min = min(np.concatenate((disp,disp_int)))
        disp_range = abs(disp_max - disp_min)
        vel_range = abs(max(vel_int) - min(vel_int))
        acc_range = abs(max(acc_int) - min(acc_int))
        
        int_ax.set_ylim(disp_min - disp_range*0.05,disp_max + disp_range *0.05)
        vel_ax.set_ylim(min(vel_int) - vel_range*0.05,max(vel_int) + vel_range*0.05)
        acc_ax.set_ylim(min(acc_int) - acc_range*0.05,max(acc_int) + acc_range*0.05)
        
        #### Incorporate Anchored Texts
        int_at = AnchoredText("SSR = {}\n$r^2$ = {}\n RMSE = {}".format(np.round(SS_res,4),np.round(r2,4),np.round(RMSE,4)),prop=dict(size=10), frameon=True,loc = 4)        
        vel_at = AnchoredText("v = {}".format(np.round(vel_int[-1],4)),prop=dict(size=10), frameon=True,loc = 4)
        acc_at = AnchoredText("a = {}".format(np.round(acc_int[-1],4)),prop=dict(size=10), frameon=True,loc = 4)        
        
        int_at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        vel_at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        acc_at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        
        int_at.patch.set_alpha(0.5)
        vel_at.patch.set_alpha(0.5)        
        acc_at.patch.set_alpha(0.5)
        
        int_ax.add_artist(int_at)
        vel_ax.add_artist(vel_at)
        acc_ax.add_artist(acc_at)
        
        #### Incorporate frame number in the figure
        plt.figtext(1-0.005,0.005,str(fig_num),ha = 'right',va='bottom',fontsize = 8)
        
        #### Plot legend for interpolation graph
        int_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5,fontsize = 12)
        
        #### Set fig title
        fig.suptitle('Data Interpolation Plot for Site {} Sensor {}'.format(name[:3].upper(),name.upper()),fontsize = 15)        
        
        #### Set axis labels
        int_ax.set_ylabel('Displacement (cm)',fontsize = 14)
        vel_ax.set_xlabel('Velocity (cm/day)',fontsize = 14)
        acc_ax.set_xlabel('Acceleration (cm/day$^2$)',fontsize = 14)
        
        ### Hide x-axis tick labels of velocity and acceleration
        vel_ax.tick_params(axis = 'x',labelbottom = 'off')
        acc_ax.tick_params(axis = 'x',labelbottom = 'off')
        
        #### Set fig size borders and spacing
        fig.set_figheight(7.5*1.25)
        fig.set_figwidth(15)
        fig.subplots_adjust(right = 0.96,top = 0.94,left = 0.050,bottom = 0.05,hspace = 0.20, wspace = 0.12)
        
        #### Set save path
        save_path = "{}/{}/Event {} to {}/Data Interpolation".format(data_path,name,cumsheardf.index[0].strftime("%d %b %y"),last_ts.strftime("%d %b %y"))
        if not os.path.exists(save_path+'/'):
            os.makedirs(save_path+'/')    
        
        #### Save figure
        plt.savefig('{}/node{}.{:04d}.png'.format(save_path,str(nodes[0])+'to'+str(nodes[-1]),fig_num),
                dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
        
        #### Close figure
        plt.close()
        
        #### Increment figure number
        fig_num += 1

def PlotThresholds(threshold_file,confidence = 0.95,interval = 'confidence'):
    #### Obtain data frame from file
    threshold_df = pd.read_csv(threshold_file,index_col = 0)         

    #### Initialize figure parameters
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()    
    all_v = np.linspace(min(threshold_df.velocity.values),max(threshold_df.velocity.values),10000)
    plot_num = 1
    marker_type = ['s','^','+','x','d']    
    
    #### Loop for all threshold type
    for threshold_type in reversed(np.unique(threshold_df.type.values)):
        
        #### Obtain critical values        
        v = threshold_df[threshold_df.type == threshold_type].velocity.values
        a = threshold_df[threshold_df.type == threshold_type].acceleration.values         
        
        #### Obtain the logarithm of the critical values
        log_v = np.log(v)
        log_a = np.log(a)
        
        #### Compute the parameters of linear regression and confidence interval
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_v,log_a)
        delta = uncertainty(log_v,log_a,slope,intercept,confidence,np.log(all_v),interval)
        
        #### Compute the threshold line and confidence interval values
        a_threshold = np.exp(slope*np.log(all_v) + intercept)
        a_threshold_upper = np.exp(np.log(a_threshold) + delta)
        a_threshold_lower = np.exp(np.log(a_threshold) - delta)
        
        #### Plot critical values
        ax.plot(v,a,'.',marker = marker_type[plot_num - 1],color = tableau20[(plot_num -1)*2])        
        
        #### Plot all computed values
        ax.plot(all_v,a_threshold,'-',color = tableau20[(plot_num -1)*2],label = threshold_type.title(),lw=1.5)
        ax.plot(all_v,a_threshold_upper,'--',color = tableau20[(plot_num -1)*2])
        ax.plot(all_v,a_threshold_lower,'--',color = tableau20[(plot_num - 1)*2])
        
        ### Plot threshold envelope
        ax.fill_between(all_v,a_threshold_lower,a_threshold_upper,color = tableau20[(plot_num - 1)*2],alpha = 0.2)
        
        #### Increment plot number        
        plot_num += 1
        
    #### Compute parameters of linear regression for all values
    v = threshold_df.velocity.values
    a = threshold_df.acceleration.values
    log_v = np.log(v)
    log_a = np.log(a)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_v,log_a)
    delta = uncertainty(log_v,log_a,slope,intercept,confidence,np.log(all_v),interval)
    a_threshold = np.exp(slope*np.log(all_v) + intercept)
    a_threshold_upper = np.exp(np.log(a_threshold) + delta)
    a_threshold_lower = np.exp(np.log(a_threshold) - delta)
    ax.plot(all_v,a_threshold,'-',color = tableau20[(plot_num -1)*2],label = 'ALL',lw=1.5)
    ax.plot(all_v,a_threshold_upper,'--',color = tableau20[(plot_num -1)*2])
    ax.plot(all_v,a_threshold_lower,'--',color = tableau20[(plot_num - 1)*2])
    ax.fill_between(all_v,a_threshold_lower,a_threshold_upper,color = tableau20[(plot_num - 1)*2],alpha = 0.2)
        
    #### Set x and y labels and scales
    ax.set_xscale('log')        
    ax.set_yscale('log')
    ax.set_xlabel('Velocity (cm/day)',fontsize = 14)
    ax.set_ylabel('Acceleration(cm/day$^2$)',fontsize = 14)
 
    #### Set xlim and ylim
    v_max = max(threshold_df.velocity.values)
    v_min = min(threshold_df.velocity.values)
    a_max = max(threshold_df.acceleration.values)
    a_min = min(threshold_df.acceleration.values)
    ax.set_xlim(v_min,v_max)
    ax.set_ylim(a_min,a_max)
   
    #### Plot labels and figure title
    ax.legend(loc = 'upper left', fancybox = True, framealpha = 0.5)
    fig.suptitle('Velocity vs. Acceleration Threshold Line for Subsurface Movement',fontsize = 15)
    
    #### Write anchored text of threshold type
    threshold_type_at = AnchoredText("{}% {} Interval".format(round(confidence*100,2),interval.title()),prop=dict(size=10), frameon=False,loc = 3)        
    ax.add_artist(threshold_type_at)
    
    #### Set fig size borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.09,bottom = 0.09)
    
    #### Set save path
    save_path = "{}".format(data_path)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    now = pd.to_datetime(datetime.now())
    
    #### Save figure
    plt.savefig('{}/Velocity vs Acceleration {} {} Threshold Line {}.png'.format(save_path,round(confidence*100,2),interval.title(),now.strftime("%Y-%m-%d_%H-%M")),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')

def PlotThresholdsForPaper(threshold_file,confidence = 0.95,interval = 'confidence'):
    #### Obtain data frame from file
    threshold_df = pd.read_csv(threshold_file,index_col = 0)         

    #### Initialize figure parameters
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()    
    v_range = max(threshold_df.velocity.values) - min(threshold_df.velocity.values)
    all_v = np.linspace(np.exp(np.log(min(threshold_df.velocity.values)) - np.log(v_range)*0.05),np.exp(np.log(max(threshold_df.velocity.values))+np.log(v_range)*0.05),10000)
    plot_num = 1
    marker_type = ['o','s','^','+','x','d']    
    h_map = {}
    #### Loop for all threshold type
    for threshold_type in reversed(np.unique(threshold_df.type.values)):
        
        #### Obtain critical values        
        v = threshold_df[threshold_df.type == threshold_type].velocity.values
        a = threshold_df[threshold_df.type == threshold_type].acceleration.values         
        
        #### Obtain the logarithm of the critical values
#        log_v = np.log(v)
#        log_a = np.log(a)
#        
#        #### Compute the parameters of linear regression and confidence interval
#        slope, intercept, r_value, p_value, std_err = stats.linregress(log_v,log_a)
#        delta = uncertainty(log_v,log_a,slope,intercept,confidence,np.log(all_v),interval)
#        
#        #### Compute the threshold line and confidence interval values
#        a_threshold = np.exp(slope*np.log(all_v) + intercept)
#        a_threshold_upper = np.exp(np.log(a_threshold) + delta)
#        a_threshold_lower = np.exp(np.log(a_threshold) - delta)
        
        #### Plot critical values
        data, = ax.plot(v,a,'.',marker = marker_type[plot_num - 1],color = tableau20[(plot_num -1)*2],label = 'Event Data')        
        
        #### Set handler map
        h_map[data] = HandlerLine2D(numpoints = 1)
        #### Plot all computed values
#        ax.plot(all_v,a_threshold,'-',color = tableau20[(plot_num -1)*2],label = threshold_type.title(),lw=1.5)
#        ax.plot(all_v,a_threshold_upper,'--',color = tableau20[(plot_num -1)*2])
#        ax.plot(all_v,a_threshold_lower,'--',color = tableau20[(plot_num - 1)*2])
        
        ### Plot threshold envelope
#        ax.fill_between(all_v,a_threshold_lower,a_threshold_upper,color = tableau20[(plot_num - 1)*2],alpha = 0.2)
        
        #### Increment plot number        
        plot_num += 1
        
    #### Compute parameters of linear regression for all values
    v = threshold_df.velocity.values
    a = threshold_df.acceleration.values
    log_v = np.log(v)
    log_a = np.log(a)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_v,log_a)
    delta = uncertainty(log_v,log_a,slope,intercept,confidence,np.log(all_v),interval)
    a_threshold = np.exp(slope*np.log(all_v) + intercept)
    a_threshold_upper = np.exp(np.log(a_threshold) + delta)
    a_threshold_lower = np.exp(np.log(a_threshold) - delta)
    ax.plot(all_v,a_threshold,'-',color = tableau20[(plot_num -1)*2],label = 'Threshold Line',lw=1.5)
    ax.plot(all_v,a_threshold_upper,'--',color = tableau20[(plot_num -1)*2])
    ax.plot(all_v,a_threshold_lower,'--',color = tableau20[(plot_num - 1)*2])
    ax.fill_between(all_v,a_threshold_lower,a_threshold_upper,color = tableau20[(plot_num - 1)*2],alpha = 0.2)
        
    #### Set x and y labels and scales
    ax.set_xscale('log')        
    ax.set_yscale('log')
    ax.set_xlabel('Velocity (cm/day)',fontsize = 14)
    ax.set_ylabel('Acceleration(cm/day$^2$)',fontsize = 14)
 
    #### Set xlim and ylim
    v_max = max(all_v)
    v_min = min(all_v)
    a_max = max(a_threshold_upper)
    a_min = min(a_threshold_lower)
    ax.set_xlim(v_min,v_max)
    ax.set_ylim(a_min,a_max)
   
    #### Plot labels and figure title
    ax.legend(loc = 'upper left', fancybox = True, framealpha = 0.5,handler_map = h_map)
#    fig.suptitle('Velocity vs. Acceleration Threshold Line for Subsurface Movement',fontsize = 15)
    
    #### Write anchored text of threshold type
    threshold_type_at = AnchoredText("{}% {} Interval".format(round(confidence*100,2),interval.title()),prop=dict(size=10), frameon=False,loc = 4)        
    ax.add_artist(threshold_type_at)
    
    #### Set fig size borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.09,bottom = 0.09)
    
    #### Set save path
    save_path = "{}".format(data_path)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    now = pd.to_datetime(datetime.now())
    
    #### Save figure
    plt.savefig('{}/Velocity vs Acceleration {} {} Threshold Line {} For Paper.png'.format(save_path,round(confidence*100,2),interval.title(),now.strftime("%Y-%m-%d_%H-%M")),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')

def PlotThresholdLinePerSite(threshold_file,confidence = 0.95,interval = 'confidence'):
    #### Obtain data frame from file
    threshold_df = pd.read_csv(threshold_file,index_col = 0)         

    #### Initialize figure parameters
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()    
    all_v = np.linspace(min(threshold_df.velocity.values),max(threshold_df.velocity.values),10000)
    plot_num = 1
    threshold_lines = []
    sensor_markers = []
    marker_type = ['s','^','o','D','*','H']    
    
    #### Plot Threshold Lines
    for threshold_type in reversed(np.unique(threshold_df.type.values)):

        #### Obtain critical values        
        v = threshold_df[threshold_df.type == threshold_type].velocity.values
        a = threshold_df[threshold_df.type == threshold_type].acceleration.values         
        
        #### Obtain the logarithm of the critical values
        log_v = np.log(v)
        log_a = np.log(a)
        
        #### Compute the parameters of linear regression and confidence interval
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_v,log_a)
        delta = uncertainty(log_v,log_a,slope,intercept,confidence,np.log(all_v),interval = interval)
        
        #### Compute the threshold line and confidence interval values
        a_threshold = np.exp(slope*np.log(all_v) + intercept)
        a_threshold_upper = np.exp(np.log(a_threshold) + delta)
        a_threshold_lower = np.exp(np.log(a_threshold) - delta)
                
        #### Plot all computed values
        threshold_line = ax.plot(all_v,a_threshold,'-',color = tableau20[(plot_num -1)*2],label = threshold_type.title(),lw=1.5)
        ax.plot(all_v,a_threshold_upper,'--',color = tableau20[(plot_num -1)*2])
        ax.plot(all_v,a_threshold_lower,'--',color = tableau20[(plot_num - 1)*2])
        
        #### Append threshold line to container
        threshold_lines = threshold_lines + threshold_line
        
        ### Plot threshold envelope
        ax.fill_between(all_v,a_threshold_lower,a_threshold_upper,color = tableau20[(plot_num - 1)*2],alpha = 0.2)
        
        #### Increment plot number        
        plot_num += 1

    #### Compute parameters of linear regression for all values
    v = threshold_df.velocity.values
    a = threshold_df.acceleration.values
    log_v = np.log(v)
    log_a = np.log(a)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_v,log_a)
    delta = uncertainty(log_v,log_a,slope,intercept,confidence,np.log(all_v),interval)
    a_threshold = np.exp(slope*np.log(all_v) + intercept)
    a_threshold_upper = np.exp(np.log(a_threshold) + delta)
    a_threshold_lower = np.exp(np.log(a_threshold) - delta)
    threshold_line = ax.plot(all_v,a_threshold,'-',color = tableau20[(plot_num -1)*2],label = 'ALL',lw=1.5)
    ax.plot(all_v,a_threshold_upper,'--',color = tableau20[(plot_num -1)*2])
    ax.plot(all_v,a_threshold_lower,'--',color = tableau20[(plot_num - 1)*2])
    ax.fill_between(all_v,a_threshold_lower,a_threshold_upper,color = tableau20[(plot_num - 1)*2],alpha = 0.2)
    threshold_lines = threshold_lines + threshold_line
    plot_num += 1
        
    #### Plot data point per site
    for sensor in np.unique(threshold_df.sensor.values):
        #### Obtain critical velocities and acceleration
        v = threshold_df[threshold_df.sensor == sensor].velocity.values
        a = threshold_df[threshold_df.sensor == sensor].acceleration.values
        
        #### Plot corresponding values        
        ## Adjust marker sizes according to shape
        if marker_type[plot_num%6] == '*' or marker_type[plot_num%6] == 'H':
            size = 8.5
        elif marker_type[plot_num%6] == 'o':
            size = 7.5
        elif marker_type[plot_num%6] == 'D':
            size = 5.5
        else:
            size = 6.5
        sensor_marker = ax.plot(v,a,'.',marker = marker_type[plot_num%6],color = tableau20[(plot_num-1)*2-16],label = sensor,markersize = size)
        
        #### Append sensor marker to container
        sensor_markers = sensor_markers + sensor_marker
        plot_num += 1
        
    #### Set x and y labels and scales
    ax.set_xscale('log')        
    ax.set_yscale('log')
    ax.set_xlabel('Velocity (cm/day)',fontsize = 14)
    ax.set_ylabel('Acceleration(cm/day$^2$)',fontsize = 14)
 
    #### Set xlim and ylim
    v_max = max(threshold_df.velocity.values)
    v_min = min(threshold_df.velocity.values)
    a_max = max(threshold_df.acceleration.values)
    a_min = min(threshold_df.acceleration.values)
    ax.set_xlim(v_min,v_max)
    ax.set_ylim(a_min,a_max)
    
    #### Set figure title
    fig.suptitle('Velocity vs. Acceleration Threshold Line for Subsurface Movement',fontsize = 15)
    
    #### Set legends
    line_labels = [l.get_label() for l in threshold_lines]
    marker_labels = [l.get_label() for l in sensor_markers]
    line_legends = ax.legend(threshold_lines,line_labels,loc = 'upper left',fancybox = True, framealpha = 0.5)
    ax.legend(sensor_markers,marker_labels,loc = 'lower right',fancybox = True, framealpha = 0.5,numpoints = 1)
    ax.add_artist(line_legends)
    
    #### Write anchored text of threshold type
    threshold_type_at = AnchoredText("{}% {} Interval".format(round(confidence*100,2),interval.title()),prop=dict(size=10), frameon=False,loc = 3)        
    ax.add_artist(threshold_type_at)
    
    #### Set fig size borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.09,bottom = 0.09)
    
    #### Set save path
    save_path = "{}".format(data_path)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Get datetime now to add to filename label
    now = pd.to_datetime(datetime.now())
    
    #### Save figure
    plt.savefig('{}/Velocity vs Acceleration {} {} Threshold Line Per Site {}.png'.format(save_path,round(confidence*100,2),interval,now.strftime("%Y-%m-%d_%H-%M")),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
def PlotVelocityComparison(disp,name,nodes,window):
    #### Get cumshear df
    cumsheardf = GetCumShearDF(disp,nodes)
    
    #### Compute for time delta values
    cumsheardf['time'] = map(lambda x: x / np.timedelta64(1,'D'),cumsheardf.index - cumsheardf.index[0])
    
    #### Convert displacement to centimeters
    cumsheardf['cumshear'] = cumsheardf.cumshear.apply(lambda x:x*100)
    
    #### Initialize velocity computation data
    cumsheardf['spline'] = None
    cumsheardf['ols'] = None
    cumsheardf['back'] = None
    cumsheardf['center'] = None
    cumsheardf['average'] = None
    cumsheardf['spline_delay'] = None
    
    #### Get last timestamp in df
    last_ts = cumsheardf.index[-1]        
    
    #### Set figure number    
    fig_num = 1
    
    #### Set bounds of slice df according to window
    for ts_start in cumsheardf[cumsheardf.index <= last_ts - window].index:

        #### Resume run
#        if fig_num != 3:
#            print "Skipping frame {:04d}".format(fig_num)
#            fig_num+=1
#            continue
        
        #### Print current progress
        print "Plotting 'Velocity Comparison' figure number {:04d}".format(fig_num)        
        
        #### Set end ts according to window        
        ts_end = ts_start + window
        
        #### Slice df
        slicedf = cumsheardf[ts_start:ts_end]
        
        #### Get time and displacement values
        time_delta = slicedf.time.values
        disp = slicedf.cumshear.values
        
        #### Commence interpolation
        try:
            #### Take the gaussian average of data points and its variance
            _,var = moving_average(disp)
            sp = UnivariateSpline(time_delta,disp,w=1/np.sqrt(var))
            
            #### Use 10000 points for interpolation
            time_int = np.linspace(time_delta[0],time_delta[-1],10000)
            
            #### Spline interpolation values    
            disp_int = sp(time_int)
            vel_int = sp.derivative(n=1)(time_int)
            
            #### Compute for goodness of fit values
            SS_res, r2, RMSE = GoodnessOfSplineFit(time_delta,disp,sp)            
            
        except:
            print "Interpolation Error {}".format(pd.to_datetime(str(time_delta[-1])).strftime("%m/%d/%Y %H:%M"))
            disp_int = np.ones(len(time_int))*np.nan
            vel_int = np.ones(len(time_int))*np.nan
        
        #### Perform velocity computations using different methods
        
        #### Method 1: From spline interpolation
        vel_spline = vel_int[-1]
        
        #### Method 2: From ordinary least squares
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_delta[-7:],disp[-7:])
        vel_ols = slope
        
        #### Method 3: From backward difference formula (sixth order)
        vel_back = np.sum(disp[-4:] *np.array([-1/3.,3/2.,-3.,11/6.]))/ (time_delta[-1] - time_delta[-2])
        
        #### Method 4: From central difference formula (sixth order)
        vel_center = np.sum(disp[-7:] *np.array([-1/60.,3/20.,-3/4.,0.,3/4.,-3/20.,1/60.])) / (time_delta[-1] - time_delta[-2])
        
        #### Method 5: From average velocity
        vel_average = (disp[-1] - disp[-7]) / (time_delta[-1] - time_delta[-7])
        
        #### Method 6: Spline Delay
        vel_spline_delay = sp.derivative(n=1)(time_delta[-4])
        
        #### Write results to the data frame
        cumsheardf.loc[ts_end,['spline','ols','back','center','average','spline_delay']] = vel_spline, vel_ols, vel_back, vel_center, vel_average, vel_spline_delay
        slicedf.loc[ts_end,['spline','ols','back','center','average','spline_delay']] = vel_spline, vel_ols, vel_back, vel_center, vel_average, vel_spline_delay
        
        #### Commence plotting
        #### Initialize figure parameters
        fig = plt.figure()
        disp_ax = fig.add_subplot(211)
        vel_ax = fig.add_subplot(212,sharex = disp_ax)
        
        #### Plot Grid
        disp_ax.grid()
        vel_ax.grid()
        
        #### Compute corresponding datetime array
        datetime_array = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_delta)        
        datetime_int = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_int)
        
        #### Plot computed values
        
        ### Plot data and interpolated values
        disp_ax.plot(datetime_array[-72:],disp[-72:],'.',color = tableau20[0],label = 'Data')
        disp_ax.plot(datetime_int,disp_int,'-',color = tableau20[12],label = 'Interpolation')
        
        #### Plot velocity values
        vel_ax.plot(datetime_array[-72:],slicedf.back.values[-72:],'-',color = tableau20[6],label = 'Back',lw = 1.75)
        vel_ax.plot(datetime_array[-72:],slicedf.center.values[-72:],'-',color = tableau20[8],label = 'Center',lw = 1.75)
        vel_ax.plot(datetime_array[-72:],slicedf.average.values[-72:],'-',color = tableau20[18],label = 'Average',lw = 1.75)
        vel_ax.plot(datetime_array[-72:],slicedf.spline_delay.values[-72:],'-',color = tableau20[10],label = 'Spline (Delay)',lw = 1.75)
        vel_ax.plot(datetime_array[-72:],slicedf.spline.values[-72:],'-',color = tableau20[2],label = 'Spline', lw = 1.75)
        vel_ax.plot(datetime_array[-72:],slicedf.ols.values[-72:],'-',color = tableau20[4],label = 'OLS', lw = 1.75)

        #### Set datetime format for x axis
        disp_ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
        vel_ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
        
        #### Set xlim for plots
        disp_ax.set_xlim(datetime_array[-72],ts_end)        
        
        #### Set ylim for plots
        disp_max = max(np.concatenate((disp[-72:],disp_int[-72:])))
        disp_min = min(np.concatenate((disp[-72:],disp_int[-72:])))
        disp_range = abs(disp_max - disp_min)
        all_v = np.concatenate((slicedf.spline.values[-72:],slicedf.ols.values[-72:],slicedf.back.values[-72:],slicedf.center.values[-72:],slicedf.average.values[-72:]))
        vel_max = max(all_v)
        vel_min = min(all_v)
        try:
            vel_range = abs(vel_max - vel_min)
            vel_ax.set_ylim(vel_min - vel_range*0.05,vel_max + vel_range*0.05)
        except:
            vel_range = 0
        
        disp_ax.set_ylim(disp_min - disp_range*0.05,disp_max + disp_range *0.05)
        
        ### Create inset axes
        inset_ax = inset_axes(disp_ax,width = "14%",height = "20%",loc = 3)
        
        ### Plot current range to the inset axes
        inset_ax.plot(cumsheardf.index,cumsheardf.cumshear.values)
        inset_ax.axvspan(datetime_array[-72],ts_end,facecolor = tableau20[2],alpha = 0.5)
        
        ### Hide ticks and labels for inset plot
        inset_ax.tick_params(top = 'off',left = 'off',bottom = 'off',right = 'off',labelleft = 'off',labelbottom = 'off')        
        
        ### Set transparency for inset plot
        inset_ax.patch.set_alpha(0.5)     
        
        #### Incorporate Anchored Texts
        disp_at = AnchoredText("SSR = {}\n$r^2$ = {}\n RMSE = {}".format(np.round(SS_res,4),np.round(r2,4),np.round(RMSE,4)),prop=dict(size=10), frameon=True,loc = 4)        
        disp_at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        disp_at.patch.set_alpha(0.5)
        
        disp_ax.add_artist(disp_at)
        
        #### Incorporate frame number in the figure
        plt.figtext(1-0.005,0.005,str(fig_num),ha = 'right',va='bottom',fontsize = 8)
        
        #### Plot legend
        disp_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5,fontsize = 12)
        vel_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5, fontsize = 12)
        
        #### Set fig title
        fig.suptitle('Velocity Comparison Plot for Site {} Sensor {}'.format(name[:3].upper(),name.upper()),fontsize = 15)        
        
        #### Set axis labels
        disp_ax.set_ylabel('Displacement (cm)',fontsize = 14)
        vel_ax.set_ylabel('Velocity (cm/day)',fontsize = 14)
        vel_ax.set_xlabel('Date',fontsize = 14)
        
        ### Hide x-axis tick labels of displacement
        disp_ax.tick_params(axis = 'x',labelbottom = 'off')
        
        #### Set fig size borders and spacing
        fig.set_figheight(7.5*1.25)
        fig.set_figwidth(15*0.70)
        fig.subplots_adjust(right = 0.96,top = 0.93,left = 0.085,bottom = 0.07,hspace = 0.11, wspace = 0.20)
        
        #### Set save path
        save_path = "{}/{}/Event {} to {}/Velocity Comparison".format(data_path,name,cumsheardf.index[0].strftime("%d %b %y"),last_ts.strftime("%d %b %y"))
        if not os.path.exists(save_path+'/'):
            os.makedirs(save_path+'/')    
        
        #### Save figure
        plt.savefig('{}/node{}.{:04d}.png'.format(save_path,str(nodes[0])+'to'+str(nodes[-1]),fig_num),
                dpi=240, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
        
        #### Close figure
        plt.close()
        
        #### Increment figure number
        fig_num += 1
        
def PlotVelocityInterpretation(disp,name,nodes,window):
    #### Get cumshear df
    cumsheardf = GetCumShearDF(disp,nodes)
    
    #### Compute for time delta values
    cumsheardf['time'] = map(lambda x: x / np.timedelta64(1,'D'),cumsheardf.index - cumsheardf.index[0])
    
    #### Convert displacement to centimeters
    cumsheardf['cumshear'] = cumsheardf.cumshear.apply(lambda x:x*100)
    
    #### Initialize velocity computation data
    cumsheardf['spline'] = None
    cumsheardf['ols'] = None
    cumsheardf['back'] = None
    cumsheardf['center'] = None
    cumsheardf['average'] = None
    cumsheardf['spline_delay'] = None
    
    #### Get last timestamp in df
    last_ts = cumsheardf.index[-1]        
    
    #### Set figure number    
    fig_num = 1
        
    #### Set bounds of slice df according to window
    for ts_start in cumsheardf[cumsheardf.index <= last_ts - window].index:

        #### Resume run
#        if fig_num != 3:
#            print "Skipping frame {:04d}".format(fig_num)
#            fig_num+=1
#            continue
        
        #### Print current progress
        print "Plotting 'Velocity Interpretation' figure number {:04d}".format(fig_num)        
        
        #### Set end ts according to window        
        ts_end = ts_start + window
        
        #### Slice df
        slicedf = cumsheardf[ts_start:ts_end]
        
        #### Get time and displacement values
        time_delta = slicedf.time.values
        disp = slicedf.cumshear.values
        
        #### Commence interpolation
        try:
            #### Take the gaussian average of data points and its variance
            _,var = moving_average(disp)
            sp = UnivariateSpline(time_delta,disp,w=1/np.sqrt(var))
            
            #### Use 10000 points for interpolation
            time_int = np.linspace(time_delta[0],time_delta[-1],10000)
            time_int_extended = np.linspace(time_delta[0],time_delta[-1]+1/24.,10000)            
            
            #### Spline interpolation values    
            disp_int = sp(time_int)
            disp_int_extended = sp(time_int_extended)
            vel_int = sp.derivative(n=1)(time_int)
                        
        except:
            print "Interpolation Error {}".format(pd.to_datetime(str(time_delta[-1])).strftime("%m/%d/%Y %H:%M"))
            disp_int = np.ones(len(time_int))*np.nan
            vel_int = np.ones(len(time_int))*np.nan
        
        #### Perform velocity computations using different methods
        
        #### Method 1: From spline interpolation
        vel_spline = vel_int[-1]
        
        #### Method 2: From ordinary least squares
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_delta[-7:],disp[-7:])
        vel_ols = slope
        
        #### Method 3: From backward difference formula (sixth order)
        vel_back = np.sum(disp[-4:] *np.array([-1/3.,3/2.,-3.,11/6.]))/ (time_delta[-1] - time_delta[-2])
        
        #### Method 4: From central difference formula (sixth order)
        vel_center = np.sum(disp[-7:] *np.array([-1/60.,3/20.,-3/4.,0.,3/4.,-3/20.,1/60.])) / (time_delta[-1] - time_delta[-2])
        
        #### Method 5: From average velocity
        vel_average = (disp[-1] - disp[-7]) / (time_delta[-1] - time_delta[-7])
        
        #### Method 6: Spline Delay
        vel_spline_delay = sp.derivative(n=1)(time_delta[-4])
        
        #### Write results to the data frame
        cumsheardf.loc[ts_end,['spline','ols','back','center','average','spline_delay']] = vel_spline, vel_ols, vel_back, vel_center, vel_average, vel_spline_delay
        slicedf.loc[ts_end,['spline','ols','back','center','average','spline_delay']] = vel_spline, vel_ols, vel_back, vel_center, vel_average, vel_spline_delay
        
        #### Commence plotting
        #### Initialize figure parameters
        fig = plt.figure()
        disp_ax = fig.add_subplot(111)
        
        #### Plot Grid
        disp_ax.grid()
        
        #### Compute corresponding datetime array
        datetime_array = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_delta)        
        datetime_int_extended = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_int_extended)       
        
        #### Plot computed values
        
        ### Plot data and interpolated values
        data, = disp_ax.plot(datetime_array[-7:],disp[-7:],'.',color = tableau20[0],label = 'Data',markersize = 12,zorder = 3)
        disp_ax.plot(datetime_int_extended,disp_int_extended,'-',color = tableau20[12],label = 'Interpolation',lw = 2.5)
        
        ### Construct the interpretations of the different methods
        time_line = np.linspace(time_delta[-9],time_delta[-1]+1/24.,10000)
        datetime_line = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_line)         
        
        ### Method 1: Velocity from spline
        spline_line = vel_spline*(time_line - time_delta[-1]) + disp_int[-1]
        disp_ax.plot(datetime_line,spline_line,'--',color = tableau20[2],label = 'Spline', lw = 1.75)
        
        ### Method 2: From ols
        ols_line = slope*time_line + intercept
        disp_ax.plot(datetime_line,ols_line,'--',color = tableau20[4],label = 'OLS', lw = 1.75)
        
        ### Plot mean disp and mean time for the window
        disp_mean = np.mean(disp[-7:])
        time_mean = np.mean(time_delta[-7:])
        datetime_mean = cumsheardf.index[0] + time_mean*pd.Timedelta(1,'D')
        disp_ax.plot(datetime_mean, disp_mean,'*',color = tableau20[4],markersize = 6)        
        
        ### Method 3: From backward difference formula
        back_line = vel_back*(time_line - time_delta[-1]) + disp[-1]
        disp_ax.plot(datetime_line,back_line,'--',color = tableau20[6],label = 'Back',lw = 1.75)
        
        ### Method 4: From central difference formula
        center_line = vel_center*(time_line - time_delta[-4]) + disp[-4]
        disp_ax.plot(datetime_line,center_line,'--',color = tableau20[8],label = 'Center',lw = 1.75)
        
        ### Method 5: From average
        average_line = vel_average*(time_line - time_delta[-1]) + disp[-1]
        disp_ax.plot(datetime_line,average_line,'--',color = tableau20[18],label = 'Average',lw = 1.75)
        
        ### Method 6: From spline delay
        spline_delay_line = vel_spline_delay*(time_line - time_delta[-4]) + sp(time_delta[-4])
        disp_ax.plot(datetime_line,spline_delay_line,'--',color = tableau20[10],label = 'Spline (Delay)',lw = 1.75)
        
        #### Set datetime format for x axis
        disp_ax.xaxis.set_major_formatter(md.DateFormatter("%H:%M"))
        
        #### Set xlim for plots
        disp_ax.set_xlim(datetime_array[-9],ts_end+pd.Timedelta(1,'h'))        
        
        #### Set ylim for plots
        disp_max = max(np.concatenate((disp[-7:],disp_int[np.where(time_int <= time_delta[-7])[-1][-1]:])))
        disp_min = min(np.concatenate((disp[-7:],disp_int[np.where(time_int <= time_delta[-7])[-1][-1]:])))
        disp_range = abs(disp_max - disp_min)
        
        disp_ax.set_ylim(disp_min - disp_range*0.075,disp_max + disp_range *0.075)
        
        #### Incorporate frame number in the figure
        plt.figtext(1-0.005,0.005,str(fig_num),ha = 'right',va='bottom',fontsize = 8)
        
        #### Plot legend
        disp_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5,fontsize = 12,handler_map = {data: HandlerLine2D(numpoints = 1)},numpoints = 8)
        
        #### Set fig title
        fig.suptitle('Velocity Interpretation Plot for Site {} Sensor {}'.format(name[:3].upper(),name.upper()),fontsize = 15)        
        
        #### Set axis labels
        disp_ax.set_ylabel('Displacement (cm)',fontsize = 14)
        disp_ax.set_xlabel('Timestamp',fontsize = 14)
                
        #### Set fig size borders and spacing
        fig.set_figheight(7.5)
        fig.set_figwidth(10)
#        fig.subplots_adjust(right = 0.96,top = 0.93,left = 0.085,bottom = 0.07,hspace = 0.11, wspace = 0.20)
        
        #### Set save path
        save_path = "{}/{}/Event {} to {}/Velocity Interpretation".format(data_path,name,cumsheardf.index[0].strftime("%d %b %y"),last_ts.strftime("%d %b %y"))
        if not os.path.exists(save_path+'/'):
            os.makedirs(save_path+'/')    
        
        #### Save figure
        plt.savefig('{}/node{}.{:04d}.png'.format(save_path,str(nodes[0])+'to'+str(nodes[-1]),fig_num),
                dpi=240, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
        
        #### Close figure
        plt.close()
        
        #### Increment figure number
        fig_num += 1

def GenerateThresholds(threshold_file,threshold_type,v_range,confidence = 0.95,interval = 'confidence'):
    #### Obtain data frame from file
    threshold_df = pd.read_csv(threshold_file,index_col = 0)         
    
    #### Obtain critical values
    if threshold_type != 'ALL':        
        v = threshold_df[threshold_df.type == threshold_type].velocity.values
        a = threshold_df[threshold_df.type == threshold_type].acceleration.values         
    else:
        v = threshold_df.velocity.values
        a = threshold_df.acceleration.values         
        
    #### Obtain the logarithm of the critical values
    log_v = np.log(v)
    log_a = np.log(a)
    
    #### Compute the parameters of linear regression and confidence interval
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_v,log_a)
    delta = uncertainty(log_v,log_a,slope,intercept,confidence,np.log(v_range),interval)
    
    #### Compute the threshold line and confidence interval values
    a_threshold = np.exp(slope*np.log(v_range) + intercept)
    a_threshold_upper = np.exp(np.log(a_threshold) + delta)
    a_threshold_lower = np.exp(np.log(a_threshold) - delta)
    
    return a_threshold,a_threshold_upper,a_threshold_lower
    

def PlotOOAResults(disp,name,nodes,window,threshold_type = 'ALL',confidence = 0.95,interval = 'prediction'):
    #### Get cumshear df
    disp_df = disp
    cumsheardf = GetCumShearDF(disp,nodes)
    
    #### Get vel df
    vel_df = GetVelDF(disp)
    
    #### Compute for time delta values
    cumsheardf['time'] = map(lambda x: x / np.timedelta64(1,'D'),cumsheardf.index - cumsheardf.index[0])
    
    #### Convert displacement to centimeters
    cumsheardf['cumshear'] = cumsheardf.cumshear.apply(lambda x:x*100)
    
    #### Initialize spline, velocity, and acceleration computation data
    cumsheardf['spline'] = None
    cumsheardf['velocity'] = None
    cumsheardf['acceleration'] = None
    cumsheardf['spline_gof'] = None

    
    #### Get last timestamp in df
    last_ts = cumsheardf.index[-1]        
    
    #### Set computation number
    comp_num = 1    
    
    #### Perform spline computations
    for ts_start in cumsheardf[cumsheardf.index <= last_ts - window].index:
        ### Control run
#        if comp_num != 106:
#           print "Skipping computation #{:04d}".format(comp_num)
#           comp_num += 1
#           continue
        
        #### Print current progress
        print "Performing spline computation #{:04d}".format(comp_num)        
        
        #### Set end ts according to window        
        ts_end = ts_start + window
        
        #### Slice df
        slicedf = cumsheardf[ts_start:ts_end]
        
        #### Get time and displacement values
        time_delta = slicedf.time.values
        disp = slicedf.cumshear.values
        
        #### Commence interpolation
        try:
            #### Take the gaussian average of data points and its variance
            _,var = moving_average(disp)
            sp = UnivariateSpline(time_delta,disp,w=1/np.sqrt(var))
            
            #### Perform velocity computation
            vel_spline = sp.derivative(n=1)(time_delta[-1])
            acc_spline = sp.derivative(n=2)(time_delta[-1])
            
            #### Evaluate goodness of fit of spline
            SS_res, r2, RMSE = GoodnessOfSplineFit(time_delta,disp,sp)            
                                                
        except:
            print "Interpolation Error {}".format(pd.to_datetime(str(time_delta[-1])).strftime("%m/%d/%Y %H:%M"))
        
        #### Store the velocity, acceleration and spline results to df
        cumsheardf.loc[ts_end,['spline','velocity','acceleration','spline_gof']] = sp, vel_spline, acc_spline,(SS_res, r2, RMSE)
        
        #### Increment computation number
        comp_num += 1
    
    #### Plotting all the results to one figure
    
    #### Set figure number
    fig_num = 1
    
    #### Determine maximum and minimum values for v
    vel_max = max(np.abs(cumsheardf.velocity.dropna().values))
    vel_min = min(np.abs(cumsheardf.velocity.dropna().values))
    vel_range = np.linspace(vel_min,vel_max,100000)
    #### Generate threshold lines
    a_threshold, a_threshold_upper, a_threshold_lower = GenerateThresholds(threshold_file,threshold_type,vel_range,confidence,interval = interval)
    
    ### Get node level alert
    node_level_df = GetNodeLevelAlert((disp_df.ix[0].ts,disp_df.ix[len(disp_df)-1].ts),name)    
    
    #### Plot result per time window
    
    for ts_start in cumsheardf[cumsheardf.index <= last_ts - window].index:
        
        #### Control run
        if fig_num != 10:
#            print "Skipping figure #{:04d}".format(fig_num)
            fig_num += 1
            continue
        
        #### Print current progress
        print "Plotting 'OOA result' figure #{:04d}".format(fig_num)        
        
        #### Set end ts according to window        
        ts_end = ts_start + window
        
        #### Slice df
        slicedf = cumsheardf[ts_start:ts_end]
        
        #### Get time and displacement values
        time_delta = slicedf.time.values
        disp = slicedf.cumshear.values
    
        #### Initialize figure
        fig = plt.figure()
        va_ax = fig.add_subplot(121)
        disp_ax = fig.add_subplot(222)
        vel_ax = fig.add_subplot(224, sharex = disp_ax)
        
        #### Plot grids
        va_ax.grid()
        disp_ax.grid()
        vel_ax.grid()        
        
        #### Compute for average velocity
        vel_ave = (disp[-1] - disp[-7])/(time_delta[-1] - time_delta[-7])
        
        #### Generate proposed alert
        
        ### Filter values with average velocity less than 6.00 cm/day
        if np.abs(vel_ave) <= 6.00:
            ooa_alert = 'L0'
        elif np.abs(vel_ave) <= 43.2:
            ooa_alert = 'L2'
        else:
            ooa_alert = 'L3'
        #### Generate thresholds and spline parameters for the point
        vel_t = slicedf.velocity.values[-1]
        acc_t = slicedf.acceleration.values[-1]
        sign_t = np.sign(vel_t*acc_t)
        a_t, a_u, a_l = GenerateThresholds(threshold_file,threshold_type,np.abs(vel_t),confidence,interval)
        r_2 = slicedf.spline_gof.values[-1][1]
        
        if r_2 >= 0.950:
            if sign_t < 0:
                ooa_alert = 'L0'
            else:
                if np.abs(acc_t) <= a_l or np.abs(acc_t) >= a_u:
                    ooa_alert = 'L0'
                
        #### Plot velocity acceleration results
        
        #### Plot threshold line and confidence interval
        thresh_line = va_ax.plot(vel_range,a_threshold,'-',color = tableau20[0],label = 'Threshold Line',lw = 1.5)
        va_ax.plot(vel_range,a_threshold_upper,'--',color = tableau20[0])
        va_ax.plot(vel_range,a_threshold_lower,'--',color = tableau20[0])
        va_ax.fill_between(vel_range,a_threshold_upper,a_threshold_lower,color = tableau20[0],alpha = 0.2)
        
        #### Plot v-a line for the last 7 values
        
        ### Filter None valuess
        vel_7 = filter(partial(is_not,None),slicedf.velocity.values[-14:])
        acc_7 = filter(partial(is_not,None),slicedf.acceleration.values[-14:])
    
        va_ax.plot(np.abs(vel_7),np.abs(acc_7),'-',color = tableau20[6],lw = 1.5)
        
        #### Initialize line containers        
        prev_pt = []
        cur_pt = []
        prev_pt_slow = []
        cur_pt_slow = []
        
        for index in range(len(vel_7)):
            #### Get the sign of velocity * acceleration and take the absolute value
            sign = np.sign(vel_7[index]*acc_7[index])
            velocity = np.abs(vel_7[index])
            acceleration = np.abs(acc_7[index])


            #### Plot each velocity and acceleration
            if index != len(vel_7) -1 :
                if sign >= 0:
                    prev_pt = va_ax.plot(velocity,acceleration,'o',color = tableau20[6],label = 'Previous')
                else:
                    prev_pt_slow = va_ax.plot(velocity,acceleration,'o',color = tableau20[16],label = 'Previous (Slowing)')
            
            #### Choose a different marker for last point
            else:
                if sign >= 0:
                    cur_pt = va_ax.plot(velocity,acceleration,'s',color = tableau20[6],label = 'Current')
                else:
                    cur_pt_slow = va_ax.plot(velocity,acceleration,'s',color = tableau20[16],label = 'Current (Slowing)')
        
        #### Plotting data and interpolation for last 7 values
        ### Set time allowance for plotting        
        time_int = np.linspace(time_delta[-7] - 0.5/24. -0.5/24.*7,time_delta[-1] + 0.5/24)
        disp_int = slicedf.spline.values[-1](time_int)
        
        #### Plot data and interpolation
        disp_ax.plot(time_delta[-7:],disp[-7:],'.',color = tableau20[0],label = 'Data',markersize = 8)
        disp_ax.plot(time_int,disp_int,'-',color = tableau20[12],label = 'Interpolation',lw = 1.75)
        
        #### Plot velocities from spline computation and nodes
        
        #### Plotting spline computation velocity
        vel_ax.plot(slicedf.time.values[-7:],slicedf.velocity.values[-7:],color = tableau20[4],label = 'Spline',lw = 1.75)
        
        #### Check nodes with alert
        number = 1
        for node_vel in np.unique(node_level_df.id.values):
            #### Select only relevant node
            node_df = vel_df[vel_df.id == node_vel]
            
            #### Select only relevant time
            vel_ax.plot(time_delta[-7:],node_df[node_df.index <= ts_end].vel.values[-7:],color = tableau20[2*(number + 2)%20],label = 'Node {}'.format(node_vel),lw = 1.75)            
            number += 1
        
        #### Incorporate anchored texts
        # Goodness of fit parameters
        int_at = AnchoredText("SSR = {}\n$r^2$ = {}\n RMSE = {}".format(np.round(slicedf.spline_gof.values[-1][0],4),np.round(slicedf.spline_gof.values[-1][1],4),np.round(slicedf.spline_gof.values[-1][2],4)),prop=dict(size=10), frameon=True,loc = 4)        
        int_at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        int_at.patch.set_alpha(0.5)
        
        ### average velocity
        ave_v_at = AnchoredText(r"$\bar{v} =$"+r"$ {}$".format(np.round((disp[-1] - disp[-7])/(time_delta[-1]-time_delta[-7]),2)),prop=dict(size=13), frameon=False,loc = 4)        
        
        ### OOA Alert
        ooa_alert_text = "OOA Alert:"
        ooa_alert_text_at = AnchoredText(ooa_alert_text,prop=dict(size=12), frameon=False,loc = 3)
        
        if ooa_alert == 'L3':
            ooa_alert_at = AnchoredText("{:>{width}}".format(ooa_alert,width = len(ooa_alert_text) + 10),prop=dict(size=12,color = tableau20[6]), frameon=False,loc = 3)
        elif ooa_alert == 'L2':
            ooa_alert_at = AnchoredText("{:>{width}}".format(ooa_alert,width = len(ooa_alert_text) + 10),prop=dict(size=12,color = tableau20[16]), frameon=False,loc = 3)
        else:
            ooa_alert_at = AnchoredText("{:>{width}}".format(ooa_alert,width = len(ooa_alert_text) + 10),prop=dict(size=12,color = tableau20[4]), frameon=False,loc = 3)
        
        # Get Current Node alert
        cur_node_level = node_level_df[node_level_df.timestamp == ts_end]
        if len(cur_node_level) == 0:
            nla_at = AnchoredText("Current node alert:     ",prop=dict(size=12), frameon=False,loc = 4)
            va_ax.add_artist(nla_at)
            nla_at = AnchoredText("L0",prop=dict(size=12,color = tableau20[4]), frameon=False,loc = 4)
            va_ax.add_artist(nla_at)
        else:
            spaces = '\n'*len(cur_node_level)
            nla_at = AnchoredText("Current node alert:"+spaces,prop=dict(size=12), frameon=False,loc = 4)
            va_ax.add_artist(nla_at)
            line_num = 0
            for node_id,node_alert,disp_alert,vel_alert in cur_node_level[['id','col_alert','disp_alert','vel_alert']].values:
                if node_alert == 'L2':
                    c = tableau20[16]
                else:
                    c = tableau20[6]
                node_at = AnchoredText("{} {}{}{} {}{}".format(node_alert,'disp'*disp_alert,'-'*disp_alert*vel_alert,'vel'*vel_alert,node_id,spaces[len(spaces)-line_num:]),prop=dict(size=12,color = c), frameon=False,loc = 4)
                va_ax.add_artist(node_at)
                line_num += 1
        #### Plot anchored texts
        disp_ax.add_artist(int_at)
        vel_ax.add_artist(ave_v_at)
        va_ax.add_artist(ooa_alert_text_at)        
        va_ax.add_artist(ooa_alert_at)
        
        #### Set scale for velocity acceleration graph
        try:
            va_ax.set_xscale('log')
            va_ax.set_yscale('log')
        except:
            print "Cannot set log axis for figure #{:04d}".format(fig_num)
        
        #### Set xlim and ylim for all subplots
        ## Velocity vs. Accelertion Axes        
        va_ax_y_max = max(filter(partial(is_not,None),cumsheardf.acceleration.values))
        va_ax_y_min = min(filter(partial(is_not,None),cumsheardf.acceleration.values))
        va_ax_y_range = va_ax_y_max - va_ax_y_min
        
        va_ax.set_xlim(vel_min,vel_max)
        va_ax.set_ylim(va_ax_y_min - va_ax_y_range*0.05,va_ax_y_max + va_ax_y_range*0.05)
        
        ## Displacement vs. Time Axes
        disp_ax_y_max = max(np.concatenate((disp_int,disp[-7:])))
        disp_ax_y_min = min(np.concatenate((disp_int,disp[-7:])))
        disp_ax_y_range = disp_ax_y_max - disp_ax_y_min        
        
        disp_ax.set_xlim(time_int[0],time_int[-1])
        disp_ax.set_ylim(disp_ax_y_min - disp_ax_y_range*0.05,disp_ax_y_max + disp_ax_y_range*0.050)
        
        ## Velocity vs. Time Axes
        try:
            all_y = np.array([])
            for lines in vel_ax.get_lines():
                x,y = lines.get_data()
                all_y = np.concatenate((y,all_y))
            vel_ax_y_max = max(all_y)
            vel_ax_y_min = min(all_y)
            vel_ax_y_range = vel_ax_y_max - vel_ax_y_min
            
            vel_ax.set_ylim(vel_ax_y_min - 0.05*vel_ax_y_range,vel_ax_y_max + 0.05*vel_ax_y_range)
        except:
            print "Cannot set Velocity axes x-limit"
        
        #### Set axis labels
        va_ax.set_xlabel('Velocity (cm/day)',fontsize = 14)
        va_ax.set_ylabel('Acceleration (cm$^2$/day)',fontsize = 14)        
        
        disp_ax.set_ylabel('Displacement (cm)',fontsize = 14)
        vel_ax.set_ylabel('Velocity (cm/day)',fontsize = 14)
        vel_ax.set_xlabel('Time (days)',fontsize = 14)
        
        ### Hide x-axis tick labels of displacement
        disp_ax.tick_params(axis = 'x',labelbottom = 'off')
        
        #### Incorporate frame number in the figure
        plt.figtext(1-0.005,0.005,str(fig_num),ha = 'right',va='bottom',fontsize = 8)        
        
        #### Set legends for va plot
        va_lines = thresh_line + cur_pt + cur_pt_slow + prev_pt + prev_pt_slow
        va_labels = [l.get_label() for l in va_lines]
        
        #### Set handler map
        h_map = {}
        for lines in va_lines[1:]:
            h_map[lines] = HandlerLine2D(numpoints = 1)
        
        #### Plot legends
        va_ax.legend(va_lines,va_labels,loc = 'upper left', fancybox = True,framealpha = 0.5,handler_map = h_map,fontsize = 12)
        disp_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5,fontsize = 12)
        vel_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5,fontsize = 12)        
        
        #### Set fig title
        fig.suptitle('Velocity Acceleration Plot for Site {} Sensor {}'.format(name[:3].upper(),name.upper()),fontsize = 15)        
        
        #### Set fig size borders and spacing
        fig.set_figheight(7.5*1.25)
        fig.set_figwidth(15)
        fig.subplots_adjust(right = 0.98,top = 0.93,left = 0.060,bottom = 0.06,hspace = 0.05, wspace = 0.13)
        
        #### Set save path
        save_path = "{}/{}/Event {} to {}/Velocity Acceleration".format(data_path,name,cumsheardf.index[0].strftime("%d %b %y"),last_ts.strftime("%d %b %y"))
        if not os.path.exists(save_path+'/'):
            os.makedirs(save_path+'/')    
        
        #### Save figure
        plt.savefig('{}/node{}.{:04d}.png'.format(save_path,str(nodes[0])+'to'+str(nodes[-1]),fig_num),
                dpi=320, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
        
        #### Close figure
        plt.close()        
        
        #### Increment figure number
        fig_num += 1
        
        return pd.DataFrame({'v':np.abs(vel_7),'a':np.abs(acc_7)}),pd.DataFrame({'v_threshold':vel_range,'a_threshold_line':a_threshold,'a_threshold_up':a_threshold_upper,'a_threshold_down':a_threshold_lower}),pd.DataFrame({'data_ts':time_delta[-14:]-time_delta[-14]+0.5/24.,'data_disp':disp[-14:]}),pd.DataFrame({'interp_ts':time_int - time_int[0],'interp_disp':disp_int})

def AerialViewPlot(disp,name,nodes):    
    
    #### Select only relevant nodes
    mask = np.zeros(len(disp.id.values))
    for values in nodes:
        mask = np.logical_or(mask,disp.id.values == values)
    disp = disp[mask]
    
    #### Set initial displacements to zero
    disp_id = disp.groupby('id',as_index = False)
    disp = disp_id.apply(set_zero_disp)      
    
    #### Compute Shear Displacements
    disp_ts = disp.groupby('ts',as_index = True)
    cumsheardf = disp_ts.apply(ComputeCumShear).reset_index(drop = True)
    
    #### Compute displacements    
    cumsheardf['cumshear_disp'] = cumsheardf['cumshear'] - cumsheardf['cumshear'].shift()
    cumsheardf['cum_xz_disp'] = cumsheardf['cum_xz'] - cumsheardf['cum_xz'].shift()
    cumsheardf['cum_xy_disp'] = cumsheardf['cum_xy'] - cumsheardf['cum_xy'].shift()
    cumsheardf['r_disp'] = np.round(np.sqrt(cumsheardf['cum_xz_disp'].values**2 + cumsheardf['cum_xy_disp'].values**2),10)
    
    #### Get non-zero values of r_disp
    non_zero_r_disp = cumsheardf[cumsheardf.r_disp != 0].dropna()
    
    #### Compute relative error    
    disp_comp_rel_error = np.abs(non_zero_r_disp.r_disp.values - np.abs(non_zero_r_disp.cumshear_disp.values))/np.abs(non_zero_r_disp.r_disp.values)
    ave_rel_error = np.average(disp_comp_rel_error)    

    #### Obtain computed values
    xz_values = cumsheardf.cum_xz.values
    xy_values = cumsheardf.cum_xy.values
    
    #### Compute slope and intercept of average direction
    slope = (xy_values[-1]-xy_values[0])/(xz_values[-1]-xz_values[0])
    intercept = xy_values[0] - slope*xz_values[0]

    #### Compute average displacement angle
    angle = np.rad2deg(np.arctan2((xy_values[-1]-xy_values[0]),(xz_values[-1]-xz_values[0])))
    
    #### Generate average line values
    ave_line_xz = np.linspace(xz_values[0],xz_values[-1],10000)    
    ave_line_xy = slope*ave_line_xz + intercept
    
    #### Set figure parameters for default plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Draw grid
    ax.grid()
    
    #### Plot computed values
    average, = ax.plot(ave_line_xz,ave_line_xy,'--',color = tableau20[6],label = 'Average',lw = 1.25)
    data_line, = ax.plot(xz_values,xy_values,'-',color = tableau20[1],label = 'Data',lw = 1.25)
    data_dot, = ax.plot(xz_values,xy_values,'.',color = tableau20[0],markersize = 6)
    
    #### Determine range for xlim and ylim
    xz_range = max(xz_values) - min(xz_values)
    xy_range = max(xy_values) - min(xy_values)
    
    #### Set xlim and ylim according to ranges
    ax.set_xlim([min(xz_values)-0.05*xz_range,max(xz_values)+0.05*xz_range])
    ax.set_ylim([min(xy_values) - 0.05*xy_range,max(xy_values)+0.05*xy_range])
    
    #### Plot legends
    ax.legend([(data_line,data_dot),average],(l.get_label() for l in [data_line,average]),fancybox = True,framealpha = 0.5,fontsize = 12)
    
    #### Plot axis labels
    ax.set_xlabel('xz (meters)',fontsize = 14)
    ax.set_ylabel('xy (meters)',fontsize = 14)
    
    #### Create anchored text for display
    text = 'Average direction = {}$^\circ$\nAverage $\delta$ = {}%'.format(np.round(angle,2),np.round(ave_rel_error*100,2))
    text_at = AnchoredText(text,prop=dict(size=12), frameon=False,loc = 3)
    
    #### Plot anchored texts
    ax.add_artist(text_at)
    
    #### Plot figure title
    fig.suptitle('{} Top View Plot'.format(name.upper()),fontsize = 15)
    
    #### Set fig size and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.08)
    
    #### Set save path
    save_path = "{}/{}/Event {} to {}/AerialView".format(data_path,name,pd.to_datetime(min(disp.ts.values)).strftime("%d %b %y"),pd.to_datetime(max(disp.ts.values)).strftime("%d %b %y"))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/Top View {} {} to {} {}.png'.format(save_path,name,pd.to_datetime(min(disp.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(disp.ts.values)).strftime("%Y-%m-%d_%H-%M"),'.'.join(map(lambda x:str(x),nodes))),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
            
    #### Compute relative error per point for color mapping
    cumsheardf['rel_error']  = np.abs(cumsheardf.r_disp.values - np.abs(cumsheardf.cumshear_disp.values))/np.abs(cumsheardf.r_disp.values)
    rel_error = cumsheardf.fillna(0).rel_error.values

    #### Set figure plotting for color mapped depending on relative error
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Draw grid
    ax.grid()
    
    #### Plot computed values
    average, = ax.plot(ave_line_xz,ave_line_xy,'--',color = tableau20[6],label = 'Average',lw = 1.25)
    data_line, = ax.plot(xz_values,xy_values,'-',color = tableau20[1],label = 'Data',lw = 1.25)
    scatter_plot = ax.scatter(xz_values,xy_values,c = rel_error,cmap = 'plasma',zorder = 3,marker='o',lw = 0,s = 12)
    
    #### Determine range for xlim and ylim
    xz_range = max(xz_values) - min(xz_values)
    xy_range = max(xy_values) - min(xy_values)
    
    #### Set xlim and ylim according to ranges
    ax.set_xlim([min(xz_values)-0.05*xz_range,max(xz_values)+0.05*xz_range])
    ax.set_ylim([min(xy_values) - 0.05*xy_range,max(xy_values)+0.05*xy_range])
    
    #### Plot legends
    ax.legend([data_line,average],(l.get_label() for l in [data_line,average]),fancybox = True,framealpha = 0.5,fontsize = 12)
    
    #### Plot colorbar
    color_bar = fig.colorbar(scatter_plot)
    
    #### Plot legend of color bar
    color_bar.set_label('$\delta$',fontsize = 18,rotation = 270,labelpad = 20)

    #### Plot axis labels
    ax.set_xlabel('xz (meters)',fontsize = 14)
    ax.set_ylabel('xy (meters)',fontsize = 14)
    
    #### Create anchored text for display
    text = 'Average direction = {}$^\circ$\nAverage $\delta$ = {}%'.format(np.round(angle,2),np.round(ave_rel_error*100,2))
    text_at = AnchoredText(text,prop=dict(size=12), frameon=False,loc = 3)
    
    #### Plot anchored texts
    ax.add_artist(text_at)
    
    #### Plot figure title
    fig.suptitle('{} Top View Plot'.format(name.upper()),fontsize = 15)
    
    #### Set fig size and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.975,top = 0.92,left = 0.100,bottom = 0.08)
    
    #### Set save path
    save_path = "{}/{}/Event {} to {}/AerialView".format(data_path,name,pd.to_datetime(min(disp.ts.values)).strftime("%d %b %y"),pd.to_datetime(max(disp.ts.values)).strftime("%d %b %y"))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/Top View {} {} to {} {} Color Mapped.png'.format(save_path,name,pd.to_datetime(min(disp.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(disp.ts.values)).strftime("%Y-%m-%d_%H-%M"),'.'.join(map(lambda x:str(x),nodes))),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    #### Set figure for corrected axes plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #### Draw grid
    ax.grid()
    
    #### Plot computed values
    average, = ax.plot(ave_line_xz,ave_line_xy,'--',color = tableau20[6],label = 'Average',lw = 1.25)
    data_line, = ax.plot(xz_values,xy_values,'-',color = tableau20[1],label = 'Data',lw = 1.25)
    data_dot, = ax.plot(xz_values,xy_values,'.',color = tableau20[0],markersize = 6)
    
    #### Determine range for xlim and ylim
    xz_range = max(xz_values) - min(xz_values)
    xy_range = max(xy_values) - min(xy_values)
    max_range = 1.10*max([xz_range,xy_range])
    xz_mid = 0.5*(max(xz_values) + min(xz_values))
    xy_mid = 0.5*(max(xy_values) + min(xy_values))

    #### Set xlim and ylim according to ranges
    ax.set_xlim([xz_mid-0.5*max_range,xz_mid+0.5*max_range])
    ax.set_ylim([xy_mid-0.5*max_range,xy_mid+0.5*max_range])

    #### Plot legends
    ax.legend([(data_line,data_dot),average],(l.get_label() for l in [data_line,average]),fancybox = True,framealpha = 0.5,fontsize = 12)
    
    #### Plot axis labels
    ax.set_xlabel('xz (meters)',fontsize = 14)
    ax.set_ylabel('xy (meters)',fontsize = 14)
    
    #### Create anchored text for display
    text = 'Average direction = {}$^\circ$'.format(np.round(angle,2))#$\nAverage $\delta$ = {}%'.format(np.round(angle,2),np.round(ave_rel_error*100,2))
    text_at = AnchoredText(text,prop=dict(size=12), frameon=False,loc = 3)
    
    #### Plot anchored texts
    ax.add_artist(text_at)
    
    #### Plot figure title
    fig.suptitle('{} Top View Plot (Corrected)'.format(name.upper()),fontsize = 15)
    
    #### Set fig size and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(9)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.100,bottom = 0.08)
    
    #### Set aspect ratio as equal
    ax.set_aspect('equal','box')    
    
    #### Set save path
    save_path = "{}/{}/Event {} to {}/AerialView".format(data_path,name,pd.to_datetime(min(disp.ts.values)).strftime("%d %b %y"),pd.to_datetime(max(disp.ts.values)).strftime("%d %b %y"))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/Top View Corrected {} {} to {} {}.png'.format(save_path,name,pd.to_datetime(min(disp.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(disp.ts.values)).strftime("%Y-%m-%d_%H-%M"),'.'.join(map(lambda x:str(x),nodes))),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')

def DoubleInterpolationPlot(disp,name,nodes,window,sigma=3,start_from = 1):    
    #### Select only relevant nodes
    mask = np.zeros(len(disp.id.values))
    for values in nodes:
        mask = np.logical_or(mask,disp.id.values == values)
    disp = disp[mask]
    
    #### Set initial displacements to zero
    disp_id = disp.groupby('id',as_index = False)
    disp = disp_id.apply(set_zero_disp)      
    
    #### Compute Shear Displacements 
    disp_ts = disp.groupby('ts',as_index = True)
    cumsheardf = disp_ts.apply(ComputeCumShear).reset_index(drop = True).set_index('ts')

    #### Compute for time delta values
    cumsheardf['time'] = map(lambda x: x / np.timedelta64(1,'D'),cumsheardf.index - cumsheardf.index[0])
    
    #### Convert displacement to centimeters
    cumsheardf['cum_xy'] = cumsheardf.cum_xy.apply(lambda x:x*100)
    cumsheardf['cum_xz'] = cumsheardf.cum_xz.apply(lambda x:x*100)
    
    #### Get last timestamp in df
    last_ts = cumsheardf.index[-1]        
    
    #### Set figure number    
    fig_num = 1

    #### Set bounds of slice df according to window
    for ts_start in cumsheardf[cumsheardf.index <= last_ts - window].index:

        #### Start from specified frame
        if fig_num < start_from:
            print "Skipping frame {:04d}".format(fig_num)
            fig_num+=1
            continue
        
        ### Set end ts according to window        
        ts_end = ts_start + window
        
        #### Slice df
        slicedf = cumsheardf[ts_start:ts_end]
        
        #### Get time and displacement values
        time_delta = slicedf.time.values
        xy_disp = slicedf.cum_xy.values
        xz_disp = slicedf.cum_xz.values
        
        #### Compute slope and intercept of average direction
        slope = (xy_disp[-1]-xy_disp[0])/(xz_disp[-1]-xz_disp[0])
        intercept = xy_disp[0] - slope*xz_disp[0]
    
        #### Compute average displacement angle
        angle = np.rad2deg(np.arctan2((xy_disp[-1]-xy_disp[0]),(xz_disp[-1]-xz_disp[0])))
        
        #### Generate average line values
        ave_line_xz = np.linspace(xz_disp[0],xz_disp[-1],10000)    
        ave_line_xy = slope*ave_line_xz + intercept
        
        #### Get average velocity
        ave_disp = np.sqrt((xy_disp[-1]-xy_disp[0])**2 + (xz_disp[-1] - xz_disp[0])**2)
        ave_velocity = ave_disp/(time_delta[-1] - time_delta[0])
        
        #### Commence interpolation
        try:
            #### Take the gaussian average of data points and its variance
            _,var_xy = moving_average(xy_disp,sigma)
            _,var_xz = moving_average(xz_disp,sigma)
            sp_xy = UnivariateSpline(time_delta,xy_disp,w=1/np.sqrt(var_xy))
            sp_xz = UnivariateSpline(time_delta,xz_disp,w=1/np.sqrt(var_xz))
            
            #### Use 10000 points for interpolation
            time_int = np.linspace(time_delta[0],time_delta[-1],10000)
            
            #### Spline interpolation values    
            int_disp_xy = sp_xy(time_int)
            int_disp_xz = sp_xz(time_int)
            
            int_vel_xy = sp_xy.derivative(n=1)(time_int)
            int_vel_xz = sp_xz.derivative(n=1)(time_int)
            
            int_acc_xy = sp_xy.derivative(n=2)(time_int)
            int_acc_xz = sp_xz.derivative(n=2)(time_int)
            
            #### Compute for goodness of fit values
            SS_res_xy, r2_xy, RMSE_xy = GoodnessOfSplineFit(time_delta,xy_disp,sp_xy)            
            SS_res_xz, r2_xz, RMSE_xz = GoodnessOfSplineFit(time_delta,xz_disp,sp_xz)            
        
        except:
            print "Interpolation Error {}".format(pd.to_datetime(slicedf.index[-1]).strftime("%m/%d/%Y %H:%M"))
            #### Use 10000 points for interpolation
            time_int = np.linspace(time_delta[0],time_delta[-1],10000)
            int_disp_xy = np.ones(len(time_int))*np.nan
            int_disp_xz = np.ones(len(time_int))*np.nan
            int_vel_xy = np.ones(len(time_int))*np.nan
            int_vel_xz = np.ones(len(time_int))*np.nan
            int_acc_xy = np.ones(len(time_int))*np.nan
            int_acc_xz = np.ones(len(time_int))*np.nan
            SS_res_xy,r2_xy,RMSE_xy = np.nan,np.nan,np.nan
            SS_res_xz,r2_xz,RMSE_xz = np.nan,np.nan,np.nan
        
        #### Compute for resultant acceleration and velocity of final values        
        res_vel = np.sqrt(int_vel_xy[-1]**2 + int_vel_xz[-1]**2)        
        res_acc = np.sqrt(int_acc_xy[-1]**2 + int_acc_xz[-1]**2)
        
        #### Commence plotting

        #### Initialize figure parameters
        fig = plt.figure()
        int_xz_ax = fig.add_subplot(221)
        int_xy_ax = fig.add_subplot(223,sharex = int_xz_ax)
        top_ax = fig.add_subplot(122)
        
        #### Plot Grid
        int_xz_ax.grid()
        int_xy_ax.grid()
        top_ax.grid()
        
        #### Compute corresponding datetime array
        datetime_array = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_delta)        
        datetime_int = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_int)
        
        #### Plot computed values
        
        ### Plot data and interpolated values
        int_xy_ax.plot(datetime_array,xy_disp,'.',color = tableau20[0],label = 'Data')
        int_xy_ax.plot(datetime_int,int_disp_xy,'-',color = tableau20[12],label = 'Interpolation')
        
        int_xz_ax.plot(datetime_array,xz_disp,'.',color = tableau20[0],label = 'Data')
        int_xz_ax.plot(datetime_int,int_disp_xz,'-',color = tableau20[12],label = 'Interpolation')        
        
        ### Create inset axes
        inset_xy_ax = inset_axes(int_xy_ax,width = "20%",height = "20%",loc = 3)
        inset_xz_ax = inset_axes(int_xz_ax,width = "20%",height = "20%",loc = 3)
        
        ### Plot current range to the inset axes
        inset_xy_ax.plot(cumsheardf.index,cumsheardf.cum_xy.values)
        inset_xy_ax.axvspan(ts_start,ts_end,facecolor = tableau20[2],alpha = 0.5)
        
        inset_xz_ax.plot(cumsheardf.index,cumsheardf.cum_xz.values)
        inset_xz_ax.axvspan(ts_start,ts_end,facecolor = tableau20[2],alpha = 0.5)
        
        ### Hide x tick labels for xz plot
        int_xz_ax.tick_params(labelbottom = 'off')
        
        ### Hide ticks and labels for inset plot
        inset_xy_ax.tick_params(top = 'off',left = 'off',bottom = 'off',right = 'off',labelleft = 'off',labelbottom = 'off')        
        inset_xz_ax.tick_params(top = 'off',left = 'off',bottom = 'off',right = 'off',labelleft = 'off',labelbottom = 'off')        
        
        ### Set transparency for inset plot
        inset_xy_ax.patch.set_alpha(0.5)        
        inset_xz_ax.patch.set_alpha(0.5)                
        
        ### Plot aerial view
        ## Different color for first point
        interpolation, = top_ax.plot(int_disp_xz,int_disp_xy,'-',color = tableau20[6],lw = 1.25, label = 'Interpolation')
        succeding_data, = top_ax.plot(xz_disp,xy_disp,'.',color = tableau20[18],markersize = 6,label = 'Data')

        ## Plot average direction
        average, = top_ax.plot(ave_line_xz,ave_line_xy,'--',color = tableau20[8],lw = 1.25,label = 'Average')        
        
        #### Set datetime format for x axis
        int_xy_ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
        int_xz_ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
        
        #### Set ylim for displacement plots
        disp_xy_max = max(np.concatenate((xy_disp,int_disp_xy)))
        disp_xy_min = min(np.concatenate((xy_disp,int_disp_xy)))
        disp_xy_range = abs(disp_xy_max - disp_xy_min)

        disp_xz_max = max(np.concatenate((xz_disp,int_disp_xz)))
        disp_xz_min = min(np.concatenate((xz_disp,int_disp_xz)))
        disp_xz_range = abs(disp_xz_max - disp_xz_min)
        
        int_xy_ax.set_ylim(disp_xy_min - disp_xy_range*0.05,disp_xy_max + disp_xy_range *0.05)
        int_xz_ax.set_ylim(disp_xz_min - disp_xz_range*0.05,disp_xz_max + disp_xz_range *0.05)
        
        #### Set xlim and ylim for aerial view plot        
        #### Determine range for xlim and ylim
        xz_range = max(xz_disp) - min(xz_disp)
        xy_range = max(xy_disp) - min(xy_disp)
        max_range = 1.10*max([xz_range,xy_range])
        xz_mid = 0.5*(max(xz_disp) + min(xz_disp))
        xy_mid = 0.5*(max(xy_disp) + min(xy_disp))
    
        #### Set xlim and ylim according to ranges
        top_ax.set_xlim([xz_mid-0.5*max_range,xz_mid+0.5*max_range])
        top_ax.set_ylim([xy_mid-0.5*max_range,xy_mid+0.5*max_range])  
        
        #### Incorporate Anchored Texts
        int_xy_at = AnchoredText("SSR = {}\n$r^2$ = {}\n RMSE = {}".format(np.round(SS_res_xy,4),np.round(r2_xy,4),np.round(RMSE_xy,4)),prop=dict(size=10), frameon=True,loc = 4)
        int_xz_at = AnchoredText("SSR = {}\n$r^2$ = {}\n RMSE = {}".format(np.round(SS_res_xz,4),np.round(r2_xz,4),np.round(RMSE_xz,4)),prop=dict(size=10), frameon=True,loc = 4)
        
        ave_vel_at = AnchoredText("Average Velocity\n{} cm/day, {}$^\circ$".format(np.round(ave_velocity,2),np.round(angle,2)),prop=dict(size=12), frameon=False,loc = 3)
        top_ax.text(0.025,-0.15,"Velocity = {} \nAcceleration = {}".format(np.round(res_vel,2),np.round(res_acc,2)),fontsize = 12,ha = 'left',va='center',transform = top_ax.transAxes)
        
        int_xy_at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        int_xz_at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        
        int_xy_at.patch.set_alpha(0.5)
        int_xz_at.patch.set_alpha(0.5)
        
        int_xy_ax.add_artist(int_xy_at)
        int_xz_ax.add_artist(int_xz_at)
        top_ax.add_artist(ave_vel_at)

        
        #### Incorporate frame number in the figure
        plt.figtext(1-0.005,0.005,str(fig_num),ha = 'right',va='bottom',fontsize = 8)
        
        #### Plot legend for interpolation graph
        int_xy_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5,fontsize = 12)
        int_xz_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5,fontsize = 12)
    
        top_legend = top_ax.legend([succeding_data,interpolation,average],(l.get_label() for l in [succeding_data,interpolation,average]),loc = 'upper center', bbox_to_anchor = (0.5,1.07),ncol = 3,fontsize = 12)        
        top_legend.get_frame().set_visible(False)
        #### Set fig title
        fig.suptitle('Interpolation Plot for Site {} Sensor {}'.format(name[:3].upper(),name.upper()),fontsize = 15)        
        
        #### Set axis labels
        int_xy_ax.set_ylabel('xy displacement (cm)',fontsize = 14)
        int_xz_ax.set_ylabel('xz displacement (cm)',fontsize = 14)
        int_xy_ax.set_xlabel('Date',fontsize = 14)
        
        top_ax.set_ylabel('xy displacement (cm)',fontsize = 14)
        top_ax.set_xlabel('xz displacement (cm)',fontsize = 14)
                
        #### Set fig size borders and spacing
        fig.set_figheight(7.5*1.25)
        fig.set_figwidth(15)
        fig.subplots_adjust(right = 0.96,top = 0.94,left = 0.075,bottom = 0.05,hspace = 0.10, wspace = 0.18)
        
        #### Set aspect ratio of aerial view as equal
        top_ax.set_aspect('equal','box')
        
        #### Set save path
        save_path = "{}/{}/Event {} to {}/Double Interpolation Sigma {}".format(data_path,name,cumsheardf.index[0].strftime("%d %b %y"),last_ts.strftime("%d %b %y"),sigma)
        if not os.path.exists(save_path+'/'):
            os.makedirs(save_path+'/')    
        
        #### Save figure
        plt.savefig('{}/node{}.{:04d}.png'.format(save_path,str(nodes[0])+'to'+str(nodes[-1]),fig_num),
                dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
        
        #### Close figure
        plt.close()
        
        #### Increment figure number
        fig_num += 1
    
def DoubleInterpolationPlotVarSigma(disp,name,nodes,window):    
    #### Select only relevant nodes
    mask = np.zeros(len(disp.id.values))
    for values in nodes:
        mask = np.logical_or(mask,disp.id.values == values)
    disp = disp[mask]
    
    #### Set initial displacements to zero
    disp_id = disp.groupby('id',as_index = False)
    disp = disp_id.apply(set_zero_disp)      
    
    #### Compute Shear Displacements 
    disp_ts = disp.groupby('ts',as_index = True)
    cumsheardf = disp_ts.apply(ComputeCumShear).reset_index(drop = True).set_index('ts')

    #### Compute for time delta values
    cumsheardf['time'] = map(lambda x: x / np.timedelta64(1,'D'),cumsheardf.index - cumsheardf.index[0])
    
    #### Convert displacement to centimeters
    cumsheardf['cum_xy'] = cumsheardf.cum_xy.apply(lambda x:x*100)
    cumsheardf['cum_xz'] = cumsheardf.cum_xz.apply(lambda x:x*100)
    
    #### Get last timestamp in df
    last_ts = cumsheardf.index[-1]        
    
    #### Set figure number    
    fig_num = 1

    #### Set bounds of slice df according to window
    for ts_start in cumsheardf[cumsheardf.index <= last_ts - window].index:

        #### Resume run
        if fig_num <= 3296:
            print "Skipping frame {:04d}".format(fig_num)
            fig_num+=1
            continue
        
        ### Set end ts according to window        
        ts_end = ts_start + window
        
        #### Slice df
        slicedf = cumsheardf[ts_start:ts_end]
        
        #### Get time and displacement values
        time_delta = slicedf.time.values
        xy_disp = slicedf.cum_xy.values
        xz_disp = slicedf.cum_xz.values
        
        #### Compute slope and intercept of average direction
        slope = (xy_disp[-1]-xy_disp[0])/(xz_disp[-1]-xz_disp[0])
        intercept = xy_disp[0] - slope*xz_disp[0]
    
        #### Compute average displacement angle
        angle = np.rad2deg(np.arctan2((xy_disp[-1]-xy_disp[0]),(xz_disp[-1]-xz_disp[0])))
        
        #### Generate average line values
        ave_line_xz = np.linspace(xz_disp[0],xz_disp[-1],10000)    
        ave_line_xy = slope*ave_line_xz + intercept
        
        #### Get average velocity
        ave_disp = np.sqrt((xy_disp[-1]-xy_disp[0])**2 + (xz_disp[-1] - xz_disp[0])**2)
        ave_velocity = ave_disp/(time_delta[-1] - time_delta[0])
        
        #### Compute for the sample standard deviation
        sigma_xy = np.sqrt(np.sum(np.power(xy_disp - np.average(xy_disp),2))/float(len(xy_disp)-1))        
        sigma_xz = np.sqrt(np.sum(np.power(xz_disp - np.average(xz_disp),2))/float(len(xz_disp)-1))        
        
        print "Frame number {}".format(fig_num)
        print "Sigma XY = {}".format(sigma_xy)
        print "Sigma XZ = {}".format(sigma_xz)        
        
        #### Commence interpolation
        try:
            #### Take the gaussian average of data points and its variance
            _,var_xy = moving_average(xy_disp,sigma_xy)
            _,var_xz = moving_average(xz_disp,sigma_xz)
            sp_xy = UnivariateSpline(time_delta,xy_disp,w=1/np.sqrt(var_xy))
            sp_xz = UnivariateSpline(time_delta,xz_disp,w=1/np.sqrt(var_xz))
            
            #### Use 10000 points for interpolation
            time_int = np.linspace(time_delta[0],time_delta[-1],10000)
            
            #### Spline interpolation values    
            int_disp_xy = sp_xy(time_int)
            int_disp_xz = sp_xz(time_int)
            
            int_vel_xy = sp_xy.derivative(n=1)(time_int)
            int_vel_xz = sp_xz.derivative(n=1)(time_int)
            
            int_acc_xy = sp_xy.derivative(n=2)(time_int)
            int_acc_xz = sp_xz.derivative(n=2)(time_int)
            
            #### Compute for goodness of fit values
            SS_res_xy, r2_xy, RMSE_xy = GoodnessOfSplineFit(time_delta,xy_disp,sp_xy)            
            SS_res_xz, r2_xz, RMSE_xz = GoodnessOfSplineFit(time_delta,xz_disp,sp_xz)            
        
        except:
            print "Interpolation Error {}".format(pd.to_datetime(slicedf.index[-1]).strftime("%m/%d/%Y %H:%M"))
            #### Use 10000 points for interpolation
            time_int = np.linspace(time_delta[0],time_delta[-1],10000)
            int_disp_xy = np.ones(len(time_int))*np.nan
            int_disp_xz = np.ones(len(time_int))*np.nan
            int_vel_xy = np.ones(len(time_int))*np.nan
            int_vel_xz = np.ones(len(time_int))*np.nan
            int_acc_xy = np.ones(len(time_int))*np.nan
            int_acc_xz = np.ones(len(time_int))*np.nan
            SS_res_xy,r2_xy,RMSE_xy = np.nan,np.nan,np.nan
            SS_res_xz,r2_xz,RMSE_xz = np.nan,np.nan,np.nan
        
        #### Compute for resultant acceleration and velocity of final values        
        res_vel = np.sqrt(int_vel_xy[-1]**2 + int_vel_xz[-1]**2)        
        res_acc = np.sqrt(int_acc_xy[-1]**2 + int_acc_xz[-1]**2)
        
        #### Commence plotting

        #### Initialize figure parameters
        fig = plt.figure()
        int_xz_ax = fig.add_subplot(221)
        int_xy_ax = fig.add_subplot(223,sharex = int_xz_ax)
        top_ax = fig.add_subplot(122)
        
        #### Plot Grid
        int_xz_ax.grid()
        int_xy_ax.grid()
        top_ax.grid()
        
        #### Compute corresponding datetime array
        datetime_array = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_delta)        
        datetime_int = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_int)
        
        #### Plot computed values
        
        ### Plot data and interpolated values
        int_xy_ax.plot(datetime_array,xy_disp,'.',color = tableau20[0],label = 'Data')
        int_xy_ax.plot(datetime_int,int_disp_xy,'-',color = tableau20[12],label = 'Interpolation')
        
        int_xz_ax.plot(datetime_array,xz_disp,'.',color = tableau20[0],label = 'Data')
        int_xz_ax.plot(datetime_int,int_disp_xz,'-',color = tableau20[12],label = 'Interpolation')        
        
        ### Create inset axes
        inset_xy_ax = inset_axes(int_xy_ax,width = "20%",height = "20%",loc = 3)
        inset_xz_ax = inset_axes(int_xz_ax,width = "20%",height = "20%",loc = 3)
        
        ### Plot current range to the inset axes
        inset_xy_ax.plot(cumsheardf.index,cumsheardf.cum_xy.values)
        inset_xy_ax.axvspan(ts_start,ts_end,facecolor = tableau20[2],alpha = 0.5)
        
        inset_xz_ax.plot(cumsheardf.index,cumsheardf.cum_xz.values)
        inset_xz_ax.axvspan(ts_start,ts_end,facecolor = tableau20[2],alpha = 0.5)
        
        ### Hide x tick labels for xz plot
        int_xz_ax.tick_params(labelbottom = 'off')
        
        ### Hide ticks and labels for inset plot
        inset_xy_ax.tick_params(top = 'off',left = 'off',bottom = 'off',right = 'off',labelleft = 'off',labelbottom = 'off')        
        inset_xz_ax.tick_params(top = 'off',left = 'off',bottom = 'off',right = 'off',labelleft = 'off',labelbottom = 'off')        
        
        ### Set transparency for inset plot
        inset_xy_ax.patch.set_alpha(0.5)        
        inset_xz_ax.patch.set_alpha(0.5)                
        
        ### Plot aerial view
        ## Different color for first point
        interpolation, = top_ax.plot(int_disp_xz,int_disp_xy,'-',color = tableau20[6],lw = 1.25, label = 'Interpolation')
        succeding_data, = top_ax.plot(xz_disp,xy_disp,'.',color = tableau20[18],markersize = 6,label = 'Data')

        ## Plot average direction
        average, = top_ax.plot(ave_line_xz,ave_line_xy,'--',color = tableau20[8],lw = 1.25,label = 'Average')        
        
        #### Set datetime format for x axis
        int_xy_ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
        int_xz_ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
        
        #### Set ylim for displacement plots
        disp_xy_max = max(np.concatenate((xy_disp,int_disp_xy)))
        disp_xy_min = min(np.concatenate((xy_disp,int_disp_xy)))
        disp_xy_range = abs(disp_xy_max - disp_xy_min)

        disp_xz_max = max(np.concatenate((xz_disp,int_disp_xz)))
        disp_xz_min = min(np.concatenate((xz_disp,int_disp_xz)))
        disp_xz_range = abs(disp_xz_max - disp_xz_min)
        
        int_xy_ax.set_ylim(disp_xy_min - disp_xy_range*0.05,disp_xy_max + disp_xy_range *0.05)
        int_xz_ax.set_ylim(disp_xz_min - disp_xz_range*0.05,disp_xz_max + disp_xz_range *0.05)
        
        #### Set xlim and ylim for aerial view plot        
        #### Determine range for xlim and ylim
        xz_range = max(xz_disp) - min(xz_disp)
        xy_range = max(xy_disp) - min(xy_disp)
        max_range = 1.10*max([xz_range,xy_range])
        xz_mid = 0.5*(max(xz_disp) + min(xz_disp))
        xy_mid = 0.5*(max(xy_disp) + min(xy_disp))
    
        #### Set xlim and ylim according to ranges
        top_ax.set_xlim([xz_mid-0.5*max_range,xz_mid+0.5*max_range])
        top_ax.set_ylim([xy_mid-0.5*max_range,xy_mid+0.5*max_range])  
        
        #### Incorporate Anchored Texts
        int_xy_at = AnchoredText("SSR = {}\n$r^2$ = {}\n RMSE = {}".format(np.round(SS_res_xy,4),np.round(r2_xy,4),np.round(RMSE_xy,4)),prop=dict(size=10), frameon=True,loc = 4)
        int_xz_at = AnchoredText("SSR = {}\n$r^2$ = {}\n RMSE = {}".format(np.round(SS_res_xz,4),np.round(r2_xz,4),np.round(RMSE_xz,4)),prop=dict(size=10), frameon=True,loc = 4)
        
        ave_vel_at = AnchoredText("Average Velocity\n{} cm/day, {}$^\circ$".format(np.round(ave_velocity,2),np.round(angle,2)),prop=dict(size=12), frameon=False,loc = 3)
        top_ax.text(0.025,-0.15,"Velocity = {} \nAcceleration = {}".format(np.round(res_vel,2),np.round(res_acc,2)),fontsize = 12,ha = 'left',va='center',transform = top_ax.transAxes)
        
        int_xy_at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        int_xz_at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        
        int_xy_at.patch.set_alpha(0.5)
        int_xz_at.patch.set_alpha(0.5)
        
        int_xy_ax.add_artist(int_xy_at)
        int_xz_ax.add_artist(int_xz_at)
        top_ax.add_artist(ave_vel_at)

        
        #### Incorporate frame number in the figure
        plt.figtext(1-0.005,0.005,str(fig_num),ha = 'right',va='bottom',fontsize = 8)
        
        #### Plot legend for interpolation graph
        int_xy_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5,fontsize = 12)
        int_xz_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5,fontsize = 12)
    
        top_legend = top_ax.legend([succeding_data,interpolation,average],(l.get_label() for l in [succeding_data,interpolation,average]),loc = 'upper center', bbox_to_anchor = (0.5,1.07),ncol = 3,fontsize = 12)        
        top_legend.get_frame().set_visible(False)
        #### Set fig title
        fig.suptitle('Interpolation Plot for Site {} Sensor {}'.format(name[:3].upper(),name.upper()),fontsize = 15)        
        
        #### Set axis labels
        int_xy_ax.set_ylabel('xy displacement (cm)',fontsize = 14)
        int_xz_ax.set_ylabel('xz displacement (cm)',fontsize = 14)
        int_xy_ax.set_xlabel('Date',fontsize = 14)
        
        top_ax.set_ylabel('xy displacement (cm)',fontsize = 14)
        top_ax.set_xlabel('xz displacement (cm)',fontsize = 14)
                
        #### Set fig size borders and spacing
        fig.set_figheight(7.5*1.25)
        fig.set_figwidth(15)
        fig.subplots_adjust(right = 0.96,top = 0.94,left = 0.075,bottom = 0.05,hspace = 0.10, wspace = 0.18)
        
        #### Set aspect ratio of aerial view as equal
        top_ax.set_aspect('equal','box')
        
        #### Set save path
        save_path = "{}/{}/Event {} to {}/Double Interpolation Var Sigma".format(data_path,name,cumsheardf.index[0].strftime("%d %b %y"),last_ts.strftime("%d %b %y"))
        if not os.path.exists(save_path+'/'):
            os.makedirs(save_path+'/')    
        
        #### Save figure
        plt.savefig('{}/node{}.{:04d}.png'.format(save_path,str(nodes[0])+'to'+str(nodes[-1]),fig_num),
                dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
        
        #### Close figure
        plt.close()
        
        #### Increment figure number
        fig_num += 1    

def DoubleInterpolationPlotWithDirection(disp,name,nodes,window,sigma = 'var',start_from = 1,end_at = None):    
    #### Select only relevant nodes
    mask = np.zeros(len(disp.id.values))
    for values in nodes:
        mask = np.logical_or(mask,disp.id.values == values)
    disp = disp[mask]
    
    #### Set initial displacements to zero
    disp_id = disp.groupby('id',as_index = False)
    disp = disp_id.apply(set_zero_disp)      
    
    #### Compute Shear Displacements 
    disp_ts = disp.groupby('ts',as_index = True)
    cumsheardf = disp_ts.apply(ComputeCumShear).reset_index(drop = True).set_index('ts')

    #### Compute for time delta values
    cumsheardf['time'] = map(lambda x: x / np.timedelta64(1,'D'),cumsheardf.index - cumsheardf.index[0])
    
    #### Convert displacement to centimeters
    cumsheardf['cum_xy'] = cumsheardf.cum_xy.apply(lambda x:x*100)
    cumsheardf['cum_xz'] = cumsheardf.cum_xz.apply(lambda x:x*100)
    
    #### Get last timestamp in df
    last_ts = cumsheardf.index[-1]        
    
    #### Set figure number    
    fig_num = 1
    
    #### Set bounds of slice df according to window
    for ts_start in cumsheardf[cumsheardf.index <= last_ts - window].index:

        #### Start from specified frame
        if fig_num < start_from:
            print "Skipping frame {:04d}".format(fig_num)
            fig_num+=1
            continue

        
        ### Set end ts according to window        
        ts_end = ts_start + window
        
        #### Slice df
        slicedf = cumsheardf[ts_start:ts_end]
        
        #### Get time and displacement values
        time_delta = slicedf.time.values
        xy_disp = slicedf.cum_xy.values
        xz_disp = slicedf.cum_xz.values
        
        #### Compute slope and intercept of average direction
        slope = (xy_disp[-1]-xy_disp[0])/(xz_disp[-1]-xz_disp[0])
        intercept = xy_disp[0] - slope*xz_disp[0]
    
        #### Compute average displacement angle
        angle = np.rad2deg(np.arctan2((xy_disp[-1]-xy_disp[0]),(xz_disp[-1]-xz_disp[0])))
        
        #### Generate average line values
        ave_line_xz = np.linspace(xz_disp[0],xz_disp[-1],10000)    
        ave_line_xy = slope*ave_line_xz + intercept
        
        #### Get average velocity
        ave_disp = np.sqrt((xy_disp[-1]-xy_disp[0])**2 + (xz_disp[-1] - xz_disp[0])**2)
        ave_velocity = ave_disp/(time_delta[-1] - time_delta[0])
        
        #### Choose standard deviation based on settings
        if sigma == 'var':
            sigma_xy = np.sqrt(np.sum(np.power(xy_disp - np.average(xy_disp),2))/float(len(xy_disp)-1))        
            sigma_xz = np.sqrt(np.sum(np.power(xz_disp - np.average(xz_disp),2))/float(len(xz_disp)-1))        
        elif sigma == '1.5var':
            sigma_xy = 1.5*np.sqrt(np.sum(np.power(xy_disp - np.average(xy_disp),2))/float(len(xy_disp)-1))        
            sigma_xz = 1.5*np.sqrt(np.sum(np.power(xz_disp - np.average(xz_disp),2))/float(len(xz_disp)-1))        
        elif sigma == '3var':
            sigma_xy = 3*np.sqrt(np.sum(np.power(xy_disp - np.average(xy_disp),2))/float(len(xy_disp)-1))        
            sigma_xz = 3*np.sqrt(np.sum(np.power(xz_disp - np.average(xz_disp),2))/float(len(xz_disp)-1))        
        else:
            sigma_xy = sigma
            sigma_xz = sigma
        
        print "Frame number {}".format(fig_num)
        print "Sigma XY = {}".format(sigma_xy)
        print "Sigma XZ = {}".format(sigma_xz)        
        
        #### Commence interpolation
        try:
            print "Commencing interpolation"
            #### Take the gaussian average of data points and its variance
            _,var_xy = moving_average(xy_disp,sigma_xy)
            _,var_xz = moving_average(xz_disp,sigma_xz)
            sp_xy = UnivariateSpline(time_delta,xy_disp,w=1/np.sqrt(var_xy))
            sp_xz = UnivariateSpline(time_delta,xz_disp,w=1/np.sqrt(var_xz))
            
            #### Use 10000 points for interpolation
            time_int = np.linspace(time_delta[0],time_delta[-1],10000)
            
            #### Spline interpolation values    
            int_disp_xy = sp_xy(time_int)
            int_disp_xz = sp_xz(time_int)
            
            int_vel_xy = sp_xy.derivative(n=1)(time_int)
            int_vel_xz = sp_xz.derivative(n=1)(time_int)
            
            int_acc_xy = sp_xy.derivative(n=2)(time_int)
            int_acc_xz = sp_xz.derivative(n=2)(time_int)
            
            #### Compute for goodness of fit values
            SS_res_xy, r2_xy, RMSE_xy = GoodnessOfSplineFit(time_delta,xy_disp,sp_xy)            
            SS_res_xz, r2_xz, RMSE_xz = GoodnessOfSplineFit(time_delta,xz_disp,sp_xz)            
        
            print "Interpolation Complete"
        except:
            print "Interpolation Error {}".format(pd.to_datetime(slicedf.index[-1]).strftime("%m/%d/%Y %H:%M"))
            #### Use 10000 points for interpolation
            time_int = np.linspace(time_delta[0],time_delta[-1],10000)
            int_disp_xy = np.ones(len(time_int))*np.nan
            int_disp_xz = np.ones(len(time_int))*np.nan
            int_vel_xy = np.ones(len(time_int))*np.nan
            int_vel_xz = np.ones(len(time_int))*np.nan
            int_acc_xy = np.ones(len(time_int))*np.nan
            int_acc_xz = np.ones(len(time_int))*np.nan
            SS_res_xy,r2_xy,RMSE_xy = np.nan,np.nan,np.nan
            SS_res_xz,r2_xz,RMSE_xz = np.nan,np.nan,np.nan
        
        #### Compute for resultant acceleration and velocity and direction of final values        
        res_vel = np.sqrt(int_vel_xy[-1]**2 + int_vel_xz[-1]**2)        
        res_acc = np.sqrt(int_acc_xy[-1]**2 + int_acc_xz[-1]**2)
        
        res_vel_angle = np.arctan2(int_vel_xy[-1],int_vel_xz[-1])
        res_acc_angle = np.arctan2(int_acc_xy[-1],int_acc_xz[-1])
        
        #### Commence plotting

        #### Initialize figure parameters
        fig = plt.figure()
        int_xz_ax = fig.add_subplot(221)
        int_xy_ax = fig.add_subplot(223,sharex = int_xz_ax)
        top_ax = fig.add_subplot(122)
        
        #### Plot Grid
        int_xz_ax.grid()
        int_xy_ax.grid()
        top_ax.grid()
        
        #### Compute corresponding datetime array
        datetime_array = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_delta)        
        datetime_int = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_int)
        
        #### Plot computed values
        
        ### Plot data and interpolated values
        int_xy_ax.plot(datetime_array,xy_disp,'.',color = tableau20[0],label = 'Data')
        int_xy_ax.plot(datetime_int,int_disp_xy,'-',color = tableau20[12],label = 'Interpolation')
        
        int_xz_ax.plot(datetime_array,xz_disp,'.',color = tableau20[0],label = 'Data')
        int_xz_ax.plot(datetime_int,int_disp_xz,'-',color = tableau20[12],label = 'Interpolation')        
        
        ### Create inset axes for displacement vs time plots
        inset_xy_ax = inset_axes(int_xy_ax,width = "20%",height = "20%",loc = 3)
        inset_xz_ax = inset_axes(int_xz_ax,width = "20%",height = "20%",loc = 3)
        
        ### Create inset axes for top view plot
        ## Get position of top ax plot
        top_ax_pos = top_ax.get_position()
        
        ## Generate inset axes
        inset_top_ax = fig.add_axes([top_ax_pos.x0 + 0.04,top_ax_pos.y0 - 0.005,top_ax_pos.width*0.3,top_ax_pos.height*0.3/1.3],polar = True)
        
        ## Customize tick parameters
        inset_top_ax.tick_params(labelleft = 'off')
        inset_top_ax.yaxis.grid(False)
        
        ## Customize theta labels
        theta_ticks = np.arange(0,360,45)
        inset_top_ax.set_thetagrids(theta_ticks,frac = 1.25,fontsize = 10)
        
        ### Plot directions to top axes inset
        if np.round(res_vel,2) != 0:
            inset_top_ax.arrow(res_vel_angle,0,0,1,length_includes_head = True,color = tableau20[4],width = 0.015,head_length = 0.25*0.75,head_width = 0.25*0.70*0.75)
        if np.round(res_acc,2) != 0:
            inset_top_ax.arrow(res_acc_angle,0,0,0.75,length_includes_head = True,color = tableau20[16],width = 0.015, head_length = 0.25 * 0.75,head_width = 0.25*0.70*0.90)
        inset_top_ax.plot(np.deg2rad(angle)*np.ones(10000),np.linspace(0,1,10000),'--',lw = 1.25, color = tableau20[8])
        inset_top_ax.plot(np.deg2rad(angle)*np.ones(10000)+np.pi,np.linspace(0,1,10000),'--',lw = 1.25, color = tableau20[8])
        
        ### Plot current range to the inset axes
        inset_xy_ax.plot(cumsheardf.index,cumsheardf.cum_xy.values)
        inset_xy_ax.axvspan(ts_start,ts_end,facecolor = tableau20[2],alpha = 0.5)
        
        inset_xz_ax.plot(cumsheardf.index,cumsheardf.cum_xz.values)
        inset_xz_ax.axvspan(ts_start,ts_end,facecolor = tableau20[2],alpha = 0.5)
        
        ### Hide x tick labels for xz plot
        int_xz_ax.tick_params(labelbottom = 'off')
        
        ### Hide ticks and labels for inset plot
        inset_xy_ax.tick_params(top = 'off',left = 'off',bottom = 'off',right = 'off',labelleft = 'off',labelbottom = 'off')        
        inset_xz_ax.tick_params(top = 'off',left = 'off',bottom = 'off',right = 'off',labelleft = 'off',labelbottom = 'off')        
        
        ### Set transparency for inset plot
        inset_xy_ax.patch.set_alpha(0.5)        
        inset_xz_ax.patch.set_alpha(0.5)                
        
        ### Plot aerial view
        ## Different color for first point
        interpolation, = top_ax.plot(int_disp_xz,int_disp_xy,'-',color = tableau20[6],lw = 1.25, label = 'Interpolation')
        succeding_data, = top_ax.plot(xz_disp,xy_disp,'.',color = tableau20[18],markersize = 6,label = 'Data')

        ## Plot average direction
        average, = top_ax.plot(ave_line_xz,ave_line_xy,'--',color = tableau20[8],lw = 1.25,label = 'Average')        
        
        #### Set datetime format for x axis
        int_xy_ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
        int_xz_ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
        
        #### Set ylim for displacement plots
        disp_xy_max = max(np.concatenate((xy_disp,int_disp_xy)))
        disp_xy_min = min(np.concatenate((xy_disp,int_disp_xy)))
        disp_xy_range = abs(disp_xy_max - disp_xy_min)

        disp_xz_max = max(np.concatenate((xz_disp,int_disp_xz)))
        disp_xz_min = min(np.concatenate((xz_disp,int_disp_xz)))
        disp_xz_range = abs(disp_xz_max - disp_xz_min)
        
        int_xy_ax.set_ylim(disp_xy_min - disp_xy_range*0.05,disp_xy_max + disp_xy_range *0.05)
        int_xz_ax.set_ylim(disp_xz_min - disp_xz_range*0.05,disp_xz_max + disp_xz_range *0.05)
        
        #### Set xlim and ylim for aerial view plot        
        #### Determine range for xlim and ylim
        xz_range = max(xz_disp) - min(xz_disp)
        xy_range = max(xy_disp) - min(xy_disp)
        max_range = 1.10*max([xz_range,xy_range])
        xz_mid = 0.5*(max(xz_disp) + min(xz_disp))
        xy_mid = 0.5*(max(xy_disp) + min(xy_disp))
    
        #### Set xlim and ylim according to ranges (h = 1.3 w)
        top_ax.set_xlim([xz_mid-0.5*max_range,xz_mid+0.5*max_range])
        top_ax.set_ylim([xy_mid-0.8*max_range,xy_mid+0.5*max_range])  
        
        #### Incorporate Anchored Texts
        int_xy_at = AnchoredText("SSR = {}\n$r^2$ = {}\n RMSE = {}".format(np.round(SS_res_xy,4),np.round(r2_xy,4),np.round(RMSE_xy,4)),prop=dict(size=10), frameon=True,loc = 4)
        int_xz_at = AnchoredText("SSR = {}\n$r^2$ = {}\n RMSE = {}".format(np.round(SS_res_xz,4),np.round(r2_xz,4),np.round(RMSE_xz,4)),prop=dict(size=10), frameon=True,loc = 4)
        ave_vel_at = AnchoredText("v = {}\na = {}\nAverage Velocity \n{} cm/day, {}$^\circ$".format(np.round(res_vel,2),np.round(res_acc,2),np.round(ave_velocity,2),np.round(angle,2)),prop=dict(size=12), frameon=False,loc = 4)
        
        int_xy_at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        int_xz_at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        
        int_xy_at.patch.set_alpha(0.5)
        int_xz_at.patch.set_alpha(0.5)
        
        int_xy_ax.add_artist(int_xy_at)
        int_xz_ax.add_artist(int_xz_at)
        top_ax.add_artist(ave_vel_at)

        
        #### Incorporate frame number in the figure
        plt.figtext(1-0.005,0.005,str(fig_num),ha = 'right',va='bottom',fontsize = 8)
        
        #### Plot legend for interpolation graph
        int_xy_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5,fontsize = 12)
        int_xz_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5,fontsize = 12)
    
        top_legend = top_ax.legend([succeding_data,interpolation,average],(l.get_label() for l in [succeding_data,interpolation,average]),loc = 'upper center', bbox_to_anchor = (0.5,1.05),ncol = 3,fontsize = 12)        
        top_legend.get_frame().set_visible(False)
        #### Set fig title
        fig.suptitle('Interpolation Plot for Site {} Sensor {}'.format(name[:3].upper(),name.upper()),fontsize = 15)        
        
        #### Set axis labels
        int_xy_ax.set_ylabel('xy displacement (cm)',fontsize = 14)
        int_xz_ax.set_ylabel('xz displacement (cm)',fontsize = 14)
        int_xy_ax.set_xlabel('Date',fontsize = 14)
        
        top_ax.set_ylabel('xy displacement (cm)',fontsize = 14)
        top_ax.set_xlabel('xz displacement (cm)',fontsize = 14)
                
        #### Set fig size borders and spacing
        fig.set_figheight(7.5*1.25)
        fig.set_figwidth(15)
        fig.subplots_adjust(right = 0.96,top = 0.94,left = 0.075,bottom = 0.05,hspace = 0.10, wspace = 0.18)
        
        #### Set aspect ratio of aerial view as equal
        top_ax.set_aspect(1,'box')
        
        #### Set save path
        save_path = "{}/{}/Event {} to {}/Double Interpolation With Direction Sigma {}".format(data_path,name,cumsheardf.index[0].strftime("%d %b %y"),last_ts.strftime("%d %b %y"),sigma)
        if not os.path.exists(save_path+'/'):
            os.makedirs(save_path+'/')    
        
        #### Save figure
        plt.savefig('{}/node{}.{:04d}.png'.format(save_path,str(nodes[0])+'to'+str(nodes[-1]),fig_num),
                dpi=320, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
        
        #### Close figure
        plt.close()
        
        #### Stop loop if at the final specified frame
        if end_at:
            if fig_num >= end_at:
                print "Stopping at frame {:04d}".format(fig_num)
                break
        
        #### Increment figure number
        fig_num += 1    
        
def PlotDVA(disp,name,nodes,window):
    #### Get cumshear df
    cumsheardf = GetCumShearDF(disp,nodes)
    
    #### Compute for time delta values
    cumsheardf['time'] = map(lambda x: x / np.timedelta64(1,'D'),cumsheardf.index - cumsheardf.index[0])
    
    #### Convert displacement to centimeters
    cumsheardf['cumshear'] = cumsheardf.cumshear.apply(lambda x:x*100)
    
    #### Get last timestamp in df
    last_ts = cumsheardf.index[-1]        
    
    #### Set figure number    
    fig_num = 1
    
    #### Set bounds of slice df according to window
    for ts_start in cumsheardf[cumsheardf.index <= last_ts - window].index:

        #### Resume run
#        if fig_num != 3:
#            print "Skipping frame {:04d}".format(fig_num)
#            fig_num+=1
#            continue
        
        #### Print current progress
        print "Plotting 'Velocity Acceleration Computation' figure number {:04d}".format(fig_num)        
        
        #### Set end ts according to window        
        ts_end = ts_start + window
        
        #### Slice df
        slicedf = cumsheardf[ts_start:ts_end]
        
        #### Get time and displacement values
        time_delta = slicedf.time.values
        disp = slicedf.cumshear.values
        
        #### Commence interpolation
        try:
            #### Take the gaussian average of data points and its variance
            _,var = moving_average(disp)
            sp = UnivariateSpline(time_delta,disp,w=1/np.sqrt(var))
            
            #### Use 10000 points for interpolation
            time_int = np.linspace(time_delta[0],time_delta[-1],10000)
            
            #### Spline interpolation values    
            disp_int = sp(time_int)
            vel_int = sp.derivative(n=1)(time_int)
            acc_int = sp.derivative(n=2)(time_int)
            
            #### Compute for goodness of fit values
            SS_res, r2, RMSE = GoodnessOfSplineFit(time_delta,disp,sp)            
            
        except:
            print "Interpolation Error {}".format(pd.to_datetime(str(time_delta[-1])).strftime("%m/%d/%Y %H:%M"))
            disp_int = np.ones(len(time_int))*np.nan
            vel_int = np.ones(len(time_int))*np.nan
            acc_int = np.ones(len(time_int))*np.nan
         
        #### Commence plotting
        #### Initialize figure parameters
        fig = plt.figure()
        disp_ax = fig.add_subplot(211)
        vel_ax = fig.add_subplot(212,sharex = disp_ax)
        acc_ax = vel_ax.twinx()
        
        #### Plot Grid
        disp_ax.grid()
        vel_ax.grid()
        
        #### Compute corresponding datetime array
        datetime_array = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_delta)        
        datetime_int = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_int)
        
        #### Plot computed values
        
        ### Plot data and interpolated values
        disp_ax.plot(datetime_array[-72:],disp[-72:],'.',color = tableau20[0],label = 'Data')
        disp_ax.plot(datetime_int,disp_int,'-',color = tableau20[12],label = 'Interpolation')
        
        #### Plot velocity values
        vel_ax.plot(datetime_int,vel_int,'-',color = tableau20[4],label = 'Velocity', lw = 1.75)
        acc_ax.plot(datetime_int,acc_int,'-',color = tableau20[6],label = 'Acceleration', lw = 1.75)

        #### Set datetime format for x axis
        disp_ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
        vel_ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
        
        #### Set xlim for plots
        disp_ax.set_xlim(datetime_array[-72],ts_end)        
        
        #### Set ylim for plots
        disp_max = max(np.concatenate((disp[-72:],disp_int[-72:])))
        disp_min = min(np.concatenate((disp[-72:],disp_int[-72:])))
        disp_range = abs(disp_max - disp_min)
        
        disp_ax.set_ylim(disp_min - disp_range*0.05,disp_max + disp_range *0.05)
        
        ### Create inset axes
        inset_ax = inset_axes(disp_ax,width = "14%",height = "20%",loc = 3)
        
        ### Plot current range to the inset axes
        inset_ax.plot(cumsheardf.index,cumsheardf.cumshear.values)
        inset_ax.axvspan(datetime_array[-72],ts_end,facecolor = tableau20[2],alpha = 0.5)
        
        ### Hide ticks and labels for inset plot
        inset_ax.tick_params(top = 'off',left = 'off',bottom = 'off',right = 'off',labelleft = 'off',labelbottom = 'off')        
        
        ### Set transparency for inset plot
        inset_ax.patch.set_alpha(0.5)     
        
        #### Incorporate Anchored Texts
        disp_at = AnchoredText("SSR = {}\n$r^2$ = {}\n RMSE = {}".format(np.round(SS_res,4),np.round(r2,4),np.round(RMSE,4)),prop=dict(size=10), frameon=True,loc = 4)        
        disp_at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        disp_at.patch.set_alpha(0.5)
        
        disp_ax.add_artist(disp_at)
        
        #### Incorporate frame number in the figure
        plt.figtext(1-0.005,0.005,str(fig_num),ha = 'right',va='bottom',fontsize = 8)
        
        #### Plot legend
        disp_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5,fontsize = 12)
        vel_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5, fontsize = 12)
        acc_ax.legend(loc = 'upper right',fancybox = True,framealpha = 0.5, fontsize = 12)
        
        #### Set fig title
        fig.suptitle('Velocity and Acceleration Computation Plot for Site {} Sensor {}'.format(name[:3].upper(),name.upper()),fontsize = 15)        
        
        #### Set axis labels
        disp_ax.set_ylabel('Displacement (cm)',fontsize = 14)
        vel_ax.set_ylabel('Velocity (cm/day)',fontsize = 14)
        vel_ax.set_xlabel('Date',fontsize = 14)
        acc_ax.set_ylabel('Acceleration (cm/day$^2$)',fontsize = 14)
        
        ### Hide x-axis tick labels of displacement
        disp_ax.tick_params(axis = 'x',labelbottom = 'off')
        
        #### Set fig size borders and spacing
        fig.set_figheight(7.5*1.25)
        fig.set_figwidth(15*0.70)
        fig.subplots_adjust(right = 0.89,top = 0.93,left = 0.085,bottom = 0.07,hspace = 0.11, wspace = 0.20)
        
        #### Set save path
        save_path = "{}/{}/Event {} to {}/VA Computation".format(data_path,name,cumsheardf.index[0].strftime("%d %b %y"),last_ts.strftime("%d %b %y"))
        if not os.path.exists(save_path+'/'):
            os.makedirs(save_path+'/')    
        
        #### Save figure
        plt.savefig('{}/node{}.{:04d}.png'.format(save_path,str(nodes[0])+'to'+str(nodes[-1]),fig_num),
                dpi=240, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
        
        #### Close figure
        plt.close()
        
        #### Increment figure number
        fig_num += 1
        
def PlotCumulativeDisplacementPlotAOGS(colposdf,sensor_column,zeroed = False,cmap = 'magma'):
    #### Set figure and subplots
    fig = plt.figure()
    ax_xz = fig.add_subplot(111)
#    ax_xy = fig.add_subplot(122,sharey = ax_xz)
    
    #### Ensure non repeating colors and color scheme = plasma
    ax_xz=cp.nonrepeat_colors(ax_xz,len(set(colposdf.ts.values)),color=cmap)
#    ax_xy=cp.nonrepeat_colors(ax_xy,len(set(colposdf.ts.values)),color='plasma')
    
    #### Set grid
#    ax_xz.grid()
#    ax_xy.grid()
    
    if zeroed == True:
        colposdf_id = colposdf.groupby('id',as_index = False)
        colposdf = colposdf_id.apply(set_zero_disp)
        zeroed = 'zeroed '
    else:
        zeroed = ''
        
    #### Compute for cumulative displacement
    colposdf_ts = colposdf.groupby('ts',as_index = False)
    colposdf = colposdf_ts.apply(compute_cumdisp)
    
    for ts in np.unique(colposdf.ts.values):
        #### Get df for a specific timestamp
        ts = pd.to_datetime(ts)
        cur_df = colposdf[colposdf.ts == ts]
        
        #### Obtain values to plot
        cur_depth = cur_df['depth'].values
        cur_xz = cur_df['cum_xz'].values * 1000
#        cur_xy = cur_df['cum_xy'].values * 1000
        
        #### Plot values write label to xz plot only
        cur_plot = ax_xz.plot(cur_xz,cur_depth,'.-',lw = 10,markersize = 32,label = ts.strftime("%d %b '%y %H:%M"),alpha = transparency)
#        ax_xy.plot(cur_xy,cur_depth,'.-',lw = 1.25,markersize = 10,alpha = transparency)
        
        
#        #### Contain all plots in 'plots' variable
#        try:
#            plots = plots + cur_plot
#        except:
#            plots = cur_plot
    
    #### Set fontsize and rotate ticks for x axis
    for tick in ax_xz.xaxis.get_minor_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
        
#    for tick in ax_xy.xaxis.get_minor_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
   
    for tick in ax_xz.xaxis.get_major_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
        
#    for tick in ax_xy.xaxis.get_major_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(10)
    
    #### Remove spines
    ax_xz.spines["top"].set_visible(False)    
    ax_xz.spines["bottom"].set_visible(False)    
    ax_xz.spines["right"].set_visible(False)    
    ax_xz.spines["left"].set_visible(False)     
        
    #### Plot the legends and labels
#    labels = [l.get_label() for l in plots]
#    fig.legend(plots,labels,loc = 'center right',fontsize = 12)
#    if zeroed == 'zeroed ':
#        fig.suptitle("Cumulative Deflection Plot for {} (zeroed)".format(sensor_column.upper()),fontsize = 15)
#    else:
#        fig.suptitle("Cumulative Deflection Plot for {}".format(sensor_column.upper()),fontsize = 15)
        
#    ax_xz.set_ylabel('Depth (meters)',fontsize = 14)
#    ax_xz.set_xlabel('Cumulative Deflection (mm)\ndownslope direction',fontsize = 14)
#    ax_xy.set_xlabel('Cumulative Deflection (mm)\nacross slope direction',fontsize = 14)
    
    #### Set xlim and ylim
    depth_range = abs(max(colposdf.depth.values) - min(colposdf.depth.values))
    cum_xz_range = abs(max(colposdf.cum_xz.values)- min(colposdf.cum_xz.values))
    cum_xy_range = abs(max(colposdf.cum_xy.values)- min(colposdf.cum_xy.values))
    
    ax_xz.set_ylim([min(colposdf.depth.values)-0.05*depth_range,max(colposdf.depth.values)+0.05*depth_range])
    
    #### Set automatic adjustement of x-axis limits
    if cum_xz_range > cum_xy_range:
        total_range = cum_xz_range*1.1
        ax_xz.set_xlim(np.array([min(colposdf.cum_xz.values)-0.05*cum_xz_range,max(colposdf.cum_xz.values)+0.05*cum_xz_range])*1000)
        
#        cum_xy_center = 0.5*(max(colposdf.cum_xy.values) + min(colposdf.cum_xy.values))
#        ax_xy.set_xlim(np.array([cum_xy_center-total_range*0.5,cum_xy_center + total_range * 0.5])*1000)
    else:
        total_range = cum_xy_range*1.1
#        ax_xy.set_xlim(np.array([min(colposdf.cum_xy.values)-0.05*cum_xy_range,max(colposdf.cum_xy.values)+0.05*cum_xy_range])*1000)
        
        cum_xz_center = 0.5*(max(colposdf.cum_xz.values) + min(colposdf.cum_xz.values))
        ax_xz.set_xlim(np.array([cum_xz_center-total_range*0.5,cum_xz_center + total_range * 0.5])*1000)
    

    #### Remove ticks
    ax_xz.tick_params(axis="both", which="both", bottom="on", top="off",labelbottom="on", left="off", right="off", labelleft="off",labelright = 'off')      
    
    #### Set y ticks
    ticks = ax_xz.get_yticks()
    ax_xz.set_yticklabels([int(abs(tick)) for tick in ticks])
    
    #### Set fig size, borders and spacing
    fig.set_figheight(18)
    fig.set_figwidth(8)
    fig.subplots_adjust(right = 0.795,top = 0.925,left = 0.100)
    
    #### Set save path
    save_path = "{}/{}/Event {} to {}/ColPos".format(data_path,sensor_column,pd.to_datetime(min(colposdf.ts.values)).strftime("%d %b %y"),pd.to_datetime(max(colposdf.ts.values)).strftime("%d %b %y"))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/CumDef {}{} {} to {} {} AOGS.png'.format(save_path,zeroed,sensor_column,pd.to_datetime(min(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M"),cmap),
            dpi=520,orientation='landscape',mode='w')
    
    print '{}/CumDef {}{} {} to {} AOGS.png'.format(save_path,zeroed,sensor_column,pd.to_datetime(min(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M"))
    
    #### Set white to transparent
    img = Image.open('{}/CumDef {}{} {} to {} {} AOGS.png'.format(save_path,zeroed,sensor_column,pd.to_datetime(min(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M"),cmap))
    img = img.convert("RGBA")
    datas = img.getdata()
    
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    
    img.putdata(newData)
    img.save('{}/CumDef {}{} {} to {} {} AOGS Transparent.png'.format(save_path,zeroed,sensor_column,pd.to_datetime(min(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M"),cmap), "PNG")

def PlotIncrementalDisplacementAOGS(colposdf,sensor_column,zeroed = False,zoomin=False):
    #### Set figure and subplots
    fig = plt.figure()
    ax_xz = fig.add_subplot(111)
#    ax_xy = fig.add_subplot(122, sharey = ax_xz)
    
    #### Ensure non repeating colors and color scheme = plasma
    ax_xz=cp.nonrepeat_colors(ax_xz,len(set(colposdf.ts.values)),color='inferno')
#    ax_xy=cp.nonrepeat_colors(ax_xy,len(set(colposdf.ts.values)),color='plasma')
    
    #### Set grid
#    ax_xz.grid()
#    ax_xy.grid()
    
    if zeroed == True:
        colposdf_id = colposdf.groupby('id',as_index = False)
        colposdf = colposdf_id.apply(set_zero_disp)
        zeroed = 'zeroed '
    else:
        zeroed = ''
    
    for ts in np.unique(colposdf.ts.values):
        #### Get df for a specific timestamp
        ts = pd.to_datetime(ts)
        cur_df = colposdf[colposdf.ts == ts]
        
        #### Obtain values to plot
        cur_depth = cur_df['depth'].values
        cur_xz = cur_df['xz'].values * 1000
#        cur_xy = cur_df['xy'].values * 1000
        
        #### Plot values write label to xz plot only
        cur_plot = ax_xz.plot(cur_xz,cur_depth,'.-',lw = 10,markersize = 32,label = ts.strftime("%d %b '%y %H:%M"),alpha = transparency)
#        ax_xy.plot(cur_xy,cur_depth,'.-',lw = 1.25,markersize = 10)
        
        
        #### Contain all plots in 'plots' variable
        try:
            plots = plots + cur_plot
        except:
            plots = cur_plot
    
#    #### Set fontsize and rotate ticks for x axis
#    for tick in ax_xz.xaxis.get_minor_ticks():
#        tick.label.set_rotation('vertical')
#        tick.label.set_fontsize(10)
        
#    for tick in ax_xy.xaxis.get_minor_ticks():
#        tick.label.set_rotation('vertical')
#        tick.label.set_fontsize(10)
#   
#    for tick in ax_xz.xaxis.get_major_ticks():
#        tick.label.set_rotation('vertical')
#        tick.label.set_fontsize(10)
#        
#    for tick in ax_xy.xaxis.get_major_ticks():
#        tick.label.set_rotation('vertical')
#        tick.label.set_fontsize(10)
    
    #### Plot the legends and labels
#    labels = [l.get_label() for l in plots]
#    fig.legend(plots,labels,loc = 'center right',fontsize = 12)
#    if zeroed == 'zeroed ':
#        fig.suptitle("Incremental Displacement Plot for {} (zeroed)".format(sensor_column.upper()),fontsize = 15)
#    else:
#        fig.suptitle("Incremental Displacement Plot for {}".format(sensor_column.upper()),fontsize = 15)
        
#    ax_xz.set_ylabel('Depth (meters)',fontsize = 14)
#    ax_xz.set_xlabel('Displacement (mm)\ndownslope direction',fontsize = 14)
##    ax_xy.set_xlabel('Displacement (mm)\nacross slope direction',fontsize = 14)
    
    #### Set xlim and ylim
    depth_range = abs(max(colposdf.depth.values) - min(colposdf.depth.values))
    xz_range = abs(max(colposdf.xz.values)- min(colposdf.xz.values))
    xy_range = abs(max(colposdf.xy.values)- min(colposdf.xy.values))
    
    ax_xz.set_ylim([min(colposdf.depth.values)-0.05*depth_range,max(colposdf.depth.values)+0.05*depth_range])
    
    if zoomin:
        ax_xz.set_xlim(np.array([min(colposdf.xz.values)-0.05*xz_range,max(colposdf.xz.values)+0.05*xz_range])*1000)
#        ax_xy.set_xlim(np.array([min(colposdf.xy.values)-0.05*xy_range,max(colposdf.xy.values)+0.05*xy_range])*1000)
        zoomin = 'zoomin '
    else:
    #### Set automatic adjustement of x-axis limits
        if xz_range > xy_range:
            total_range = xz_range*1.1
            ax_xz.set_xlim(np.array([min(colposdf.xz.values)-0.05*xz_range,max(colposdf.xz.values)+0.05*xz_range])*1000)
            
#            xy_center = 0.5*(max(colposdf.xy.values) + min(colposdf.xy.values))
#            ax_xy.set_xlim(np.array([xy_center-total_range*0.5,xy_center + total_range * 0.5])*1000)
        else:
            total_range = xy_range*1.1
#            ax_xy.set_xlim(np.array([min(colposdf.xy.values)-0.05*xy_range,max(colposdf.xy.values)+0.05*xy_range])*1000)
            
            xz_center = 0.5*(max(colposdf.xz.values) + min(colposdf.xz.values))
            ax_xz.set_xlim(np.array([xz_center-total_range*0.5,xz_center + total_range * 0.5])*1000)
        zoomin = ''
    
    #### Remove spines
    ax_xz.spines["top"].set_visible(False)    
    ax_xz.spines["bottom"].set_visible(False)    
    ax_xz.spines["right"].set_visible(False)    
    ax_xz.spines["left"].set_visible(False)   
    
    #### Remove ticks
    ax_xz.tick_params(axis="both", which="both", bottom="on", top="off",labelbottom="on", left="off", right="off", labelleft="off",labelright = 'off')      

    
    #### Set fig size, borders and spacing
    fig.set_figheight(18)
    fig.set_figwidth(8)
    fig.subplots_adjust(right = 0.795,top = 0.925,left = 0.100)
    
    #### Set save path
    save_path = "{}/{}/Event {} to {}/IncDisp".format(data_path,sensor_column,pd.to_datetime(min(colposdf.ts.values)).strftime("%d %b %y"),pd.to_datetime(max(colposdf.ts.values)).strftime("%d %b %y"))
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/IncDisp {}{}{} {} to {} AOGS.png'.format(save_path,zeroed,zoomin,sensor_column,pd.to_datetime(min(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M")),
            dpi=520, facecolor='w', edgecolor='w',orientation='landscape',mode='w')

    #### Set white to transparent
    img = Image.open('{}/IncDisp {}{}{} {} to {} AOGS.png'.format(save_path,zeroed,zoomin,sensor_column,pd.to_datetime(min(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M")))
    img = img.convert("RGBA")
    datas = img.getdata()
    
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    
    img.putdata(newData)
    img.save('{}/IncDisp {}{}{} {} to {} AOGS Transparent.png'.format(save_path,zeroed,zoomin,sensor_column,pd.to_datetime(min(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(colposdf.ts.values)).strftime("%Y-%m-%d_%H-%M")), "PNG")

def DoubleInterpolationPlotWithDirectionAOGS(disp,name,nodes,window,sigma = 'var',start_from = 1,end_at = None):    
    #### Select only relevant nodes
    mask = np.zeros(len(disp.id.values))
    for values in nodes:
        mask = np.logical_or(mask,disp.id.values == values)
    disp = disp[mask]
    
    #### Set initial displacements to zero
    disp_id = disp.groupby('id',as_index = False)
    disp = disp_id.apply(set_zero_disp)      
    
    #### Compute Shear Displacements 
    disp_ts = disp.groupby('ts',as_index = True)
    cumsheardf = disp_ts.apply(ComputeCumShear).reset_index(drop = True).set_index('ts')

    #### Compute for time delta values
    cumsheardf['time'] = map(lambda x: x / np.timedelta64(1,'D'),cumsheardf.index - cumsheardf.index[0])
    
    #### Convert displacement to centimeters
    cumsheardf['cum_xy'] = cumsheardf.cum_xy.apply(lambda x:x*100)
    cumsheardf['cum_xz'] = cumsheardf.cum_xz.apply(lambda x:x*100)
    
    #### Get last timestamp in df
    last_ts = cumsheardf.index[-1]        
    
    #### Set figure number    
    fig_num = 1
    
    #### Set bounds of slice df according to window
    for ts_start in cumsheardf[cumsheardf.index <= last_ts - window].index:

        #### Start from specified frame
        if fig_num < start_from:
            print "Skipping frame {:04d}".format(fig_num)
            fig_num+=1
            continue

        
        ### Set end ts according to window        
        ts_end = ts_start + window
        
        #### Slice df
        slicedf = cumsheardf[ts_start:ts_end]
        
        #### Get time and displacement values
        time_delta = slicedf.time.values
        xy_disp = slicedf.cum_xy.values
        xz_disp = slicedf.cum_xz.values
        
        #### Compute slope and intercept of average direction
        slope = (xy_disp[-1]-xy_disp[0])/(xz_disp[-1]-xz_disp[0])
        intercept = xy_disp[0] - slope*xz_disp[0]
    
        #### Compute average displacement angle
        angle = np.rad2deg(np.arctan2((xy_disp[-1]-xy_disp[0]),(xz_disp[-1]-xz_disp[0])))
        
        #### Generate average line values
        ave_line_xz = np.linspace(xz_disp[0],xz_disp[-1],10000)    
        ave_line_xy = slope*ave_line_xz + intercept
        
        #### Get average velocity
        ave_disp = np.sqrt((xy_disp[-1]-xy_disp[0])**2 + (xz_disp[-1] - xz_disp[0])**2)
        ave_velocity = ave_disp/(time_delta[-1] - time_delta[0])
        
        #### Choose standard deviation based on settings
        if sigma == 'var':
            sigma_xy = np.sqrt(np.sum(np.power(xy_disp - np.average(xy_disp),2))/float(len(xy_disp)-1))        
            sigma_xz = np.sqrt(np.sum(np.power(xz_disp - np.average(xz_disp),2))/float(len(xz_disp)-1))        
        elif sigma == '1.5var':
            sigma_xy = 1.5*np.sqrt(np.sum(np.power(xy_disp - np.average(xy_disp),2))/float(len(xy_disp)-1))        
            sigma_xz = 1.5*np.sqrt(np.sum(np.power(xz_disp - np.average(xz_disp),2))/float(len(xz_disp)-1))        
        elif sigma == '3var':
            sigma_xy = 3*np.sqrt(np.sum(np.power(xy_disp - np.average(xy_disp),2))/float(len(xy_disp)-1))        
            sigma_xz = 3*np.sqrt(np.sum(np.power(xz_disp - np.average(xz_disp),2))/float(len(xz_disp)-1))        
        else:
            sigma_xy = sigma
            sigma_xz = sigma
        
        print "Frame number {}".format(fig_num)
        print "Sigma XY = {}".format(sigma_xy)
        print "Sigma XZ = {}".format(sigma_xz)        
        
        #### Commence interpolation
        try:
            print "Commencing interpolation"
            #### Take the gaussian average of data points and its variance
            _,var_xy = moving_average(xy_disp,sigma_xy)
            _,var_xz = moving_average(xz_disp,sigma_xz)
            sp_xy = UnivariateSpline(time_delta,xy_disp,w=1/np.sqrt(var_xy))
            sp_xz = UnivariateSpline(time_delta,xz_disp,w=1/np.sqrt(var_xz))
            
            #### Use 10000 points for interpolation
            time_int = np.linspace(time_delta[0],time_delta[-1],10000)
            
            #### Spline interpolation values    
            int_disp_xy = sp_xy(time_int)
            int_disp_xz = sp_xz(time_int)
            
            int_vel_xy = sp_xy.derivative(n=1)(time_int)
            int_vel_xz = sp_xz.derivative(n=1)(time_int)
            
            int_acc_xy = sp_xy.derivative(n=2)(time_int)
            int_acc_xz = sp_xz.derivative(n=2)(time_int)
            
            #### Compute for goodness of fit values
            SS_res_xy, r2_xy, RMSE_xy = GoodnessOfSplineFit(time_delta,xy_disp,sp_xy)            
            SS_res_xz, r2_xz, RMSE_xz = GoodnessOfSplineFit(time_delta,xz_disp,sp_xz)            
        
            print "Interpolation Complete"
        except:
            print "Interpolation Error {}".format(pd.to_datetime(slicedf.index[-1]).strftime("%m/%d/%Y %H:%M"))
            #### Use 10000 points for interpolation
            time_int = np.linspace(time_delta[0],time_delta[-1],10000)
            int_disp_xy = np.ones(len(time_int))*np.nan
            int_disp_xz = np.ones(len(time_int))*np.nan
            int_vel_xy = np.ones(len(time_int))*np.nan
            int_vel_xz = np.ones(len(time_int))*np.nan
            int_acc_xy = np.ones(len(time_int))*np.nan
            int_acc_xz = np.ones(len(time_int))*np.nan
            SS_res_xy,r2_xy,RMSE_xy = np.nan,np.nan,np.nan
            SS_res_xz,r2_xz,RMSE_xz = np.nan,np.nan,np.nan
        
        #### Compute for resultant acceleration and velocity and direction of final values        
        res_vel = np.sqrt(int_vel_xy[-1]**2 + int_vel_xz[-1]**2)        
        res_acc = np.sqrt(int_acc_xy[-1]**2 + int_acc_xz[-1]**2)
        
        res_vel_angle = np.arctan2(int_vel_xy[-1],int_vel_xz[-1])
        res_acc_angle = np.arctan2(int_acc_xy[-1],int_acc_xz[-1])
        
        #### Commence plotting

        #### Initialize figure parameters
        fig = plt.figure()
#        int_xz_ax = fig.add_subplot(221)
#        int_xy_ax = fig.add_subplot(223,sharex = int_xz_ax)
        top_ax = fig.add_subplot(111)
        
        #### Plot Grid
#        int_xz_ax.grid()
#        int_xy_ax.grid()
#        top_ax.grid()
        
        #### Compute corresponding datetime array
        datetime_array = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_delta)        
        datetime_int = map(lambda x:cumsheardf.index[0] + x*pd.Timedelta(1,'D'),time_int)
        
        #### Plot computed values
        
        ### Plot data and interpolated values
#        int_xy_ax.plot(time_int,int_disp_xy,'-',color = tableau20[12],label = 'Interpolation')
#        int_xy_ax.plot(time_delta,xy_disp,'.',color = tableau20[0],label = 'Data')

#        int_xz_ax.plot(time_int,int_disp_xz,'-',color = tableau20[12],label = 'Interpolation')        
#        int_xz_ax.plot(time_delta,xz_disp,'.',color = tableau20[0],label = 'Data')
        
        #### Remove spines
#        for ax_xz in [int_xy_ax,int_xz_ax]:
#            ax_xz.spines["top"].set_visible(False)    
#            ax_xz.spines["left"].set_visible(False)   
#            ax_xz.tick_params(axis="both", which="both", bottom="on", top="off",labelbottom="on", left="off", right="on", labelleft="off",labelright = 'on')      
        
        ### Create inset axes for displacement vs time plots
#        inset_xy_ax = inset_axes(int_xy_ax,width = "20%",height = "20%",loc = 3)
#        inset_xz_ax = inset_axes(int_xz_ax,width = "20%",height = "20%",loc = 3)
        
        ### Create inset axes for top view plot
        ## Get position of top ax plot
#        top_ax_pos = top_ax.get_position()
#        
#        ## Generate inset axes
#        inset_top_ax = fig.add_axes([top_ax_pos.x0 + 0.04,top_ax_pos.y0 - 0.005,top_ax_pos.width*0.3,top_ax_pos.height*0.3/1.3],polar = True)
#        
#        ## Customize tick parameters
#        inset_top_ax.tick_params(labelleft = 'off')
#        inset_top_ax.yaxis.grid(False)
#        
#        ## Customize theta labels
#        theta_ticks = np.arange(0,360,45)
#        inset_top_ax.set_thetagrids(theta_ticks,frac = 1.25,fontsize = 10)
#        
#        ### Plot directions to top axes inset
#        if np.round(res_vel,2) != 0:
#            inset_top_ax.arrow(res_vel_angle,0,0,1,length_includes_head = True,color = tableau20[4],width = 0.015,head_length = 0.25*0.75,head_width = 0.25*0.70*0.75)
#        if np.round(res_acc,2) != 0:
#            inset_top_ax.arrow(res_acc_angle,0,0,0.75,length_includes_head = True,color = tableau20[16],width = 0.015, head_length = 0.25 * 0.75,head_width = 0.25*0.70*0.90)
#        inset_top_ax.plot(np.deg2rad(angle)*np.ones(10000),np.linspace(0,1,10000),'--',lw = 1.25, color = tableau20[8])
#        inset_top_ax.plot(np.deg2rad(angle)*np.ones(10000)+np.pi,np.linspace(0,1,10000),'--',lw = 1.25, color = tableau20[8])
        
        ### Plot current range to the inset axes
#        inset_xy_ax.plot(cumsheardf.index,cumsheardf.cum_xy.values)
#        inset_xy_ax.axvspan(ts_start,ts_end,facecolor = tableau20[2],alpha = 0.5)
#        
#        inset_xz_ax.plot(cumsheardf.index,cumsheardf.cum_xz.values)
#        inset_xz_ax.axvspan(ts_start,ts_end,facecolor = tableau20[2],alpha = 0.5)
#        
        ### Hide x tick labels for xz plot
#        int_xz_ax.tick_params(labelbottom = 'off')
#        
#        ### Hide ticks and labels for inset plot
#        inset_xy_ax.tick_params(top = 'off',left = 'off',bottom = 'off',right = 'off',labelleft = 'off',labelbottom = 'off')        
#        inset_xz_ax.tick_params(top = 'off',left = 'off',bottom = 'off',right = 'off',labelleft = 'off',labelbottom = 'off')        
#        
#        ### Set transparency for inset plot
#        inset_xy_ax.patch.set_alpha(0.5)        
#        inset_xz_ax.patch.set_alpha(0.5)                
        
        ### Plot aerial view
        ## Different color for first point
        interpolation, = top_ax.plot(int_disp_xz,int_disp_xy,'-',color =np.array((234,182,79))/255.,lw = 3, label = 'Interpolation')
        succeding_data, = top_ax.plot(xz_disp,xy_disp,'.',color = np.array((248,153,29))/255.,markersize = 15,label = 'Data')

        ## Plot average direction
#        average, = top_ax.plot(ave_line_xz,ave_line_xy,'--',color = tableau20[8],lw = 1.25,label = 'Average')        
        
        #### Set datetime format for x axis
#        int_xy_ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
#        int_xz_ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
        
        #### Set ylim for displacement plots
        disp_xy_max = max(np.concatenate((xy_disp,int_disp_xy)))
        disp_xy_min = min(np.concatenate((xy_disp,int_disp_xy)))
#        disp_xy_range = abs(disp_xy_max - disp_xy_min)

        disp_xz_max = max(np.concatenate((xz_disp,int_disp_xz)))
        disp_xz_min = min(np.concatenate((xz_disp,int_disp_xz)))
#        disp_xz_range = abs(disp_xz_max - disp_xz_min)
        
#        int_xy_ax.set_ylim(disp_xy_min - disp_xy_range*0.05,disp_xy_max + disp_xy_range *0.05)
#        int_xz_ax.set_ylim(disp_xz_min - disp_xz_range*0.05,disp_xz_max + disp_xz_range *0.05)
#        
        #### Set xlim and ylim for aerial view plot        
        #### Determine range for xlim and ylim
        xz_range = max(xz_disp) - min(xz_disp)
        xy_range = max(xy_disp) - min(xy_disp)
        max_range = 1.10*max([xz_range,xy_range])
        xz_mid = 0.5*(max(xz_disp) + min(xz_disp))
        xy_mid = 0.5*(max(xy_disp) + min(xy_disp))
        
        #### Set xlim and ylim according to ranges (h = 1.3 w)
        top_ax.set_xlim([xz_mid-0.45*max_range,xz_mid+0.45*max_range])
        top_ax.set_ylim([-3.25,-1.60])  
        
        #### Incorporate Anchored Texts
#        int_xy_at = AnchoredText("SSR = {}\n$r^2$ = {}\n RMSE = {}".format(np.round(SS_res_xy,4),np.round(r2_xy,4),np.round(RMSE_xy,4)),prop=dict(size=10), frameon=True,loc = 4)
#        int_xz_at = AnchoredText("SSR = {}\n$r^2$ = {}\n RMSE = {}".format(np.round(SS_res_xz,4),np.round(r2_xz,4),np.round(RMSE_xz,4)),prop=dict(size=10), frameon=True,loc = 4)
#        ave_vel_at = AnchoredText("v = {}\na = {}\nAverage Velocity \n{} cm/day, {}$^\circ$".format(np.round(res_vel,2),np.round(res_acc,2),np.round(ave_velocity,2),np.round(angle,2)),prop=dict(size=12), frameon=False,loc = 4)
#        
#        int_xy_at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
#        int_xz_at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
#        
#        int_xy_at.patch.set_alpha(0.5)
#        int_xz_at.patch.set_alpha(0.5)
#        
#        int_xy_ax.add_artist(int_xy_at)
#        int_xz_ax.add_artist(int_xz_at)
#        top_ax.add_artist(ave_vel_at)

        #### Remove spines
        top_ax.spines["top"].set_visible(False)     
        top_ax.spines["right"].set_visible(False)    
        
        plt.setp(top_ax.spines.values(), color=np.array((248,153,29))/255.)
        plt.setp([top_ax.get_xticklines(), top_ax.get_yticklines()], color=np.array((248,153,29))/255.)
        top_ax.xaxis.label.set_color(np.array((248,153,29))/255.)
        top_ax.yaxis.label.set_color(np.array((248,153,29))/255.)
        
        top_ax.spines['left'].set_linewidth(3)
        top_ax.spines['bottom'].set_linewidth(3)
        
        #### Set tick width
        top_ax.tick_params(width = 2,length = 5)
        
        
        #### Incorporate frame number in the figure
        plt.figtext(1-0.005,0.005,str(fig_num),ha = 'right',va='bottom',fontsize = 8)
        
        #### Plot legend for interpolation graph
#        int_xy_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5,fontsize = 12)
#        int_xz_ax.legend(loc = 'upper left',fancybox = True, framealpha = 0.5,fontsize = 12)
#    
#        top_legend = top_ax.legend([succeding_data,interpolation,average],(l.get_label() for l in [succeding_data,interpolation,average]),loc = 'upper center', bbox_to_anchor = (0.5,1.05),ncol = 3,fontsize = 12)        
#        top_legend.get_frame().set_visible(False)
#        #### Set fig title
#        fig.suptitle('Interpolation Plot for Site {} Sensor {}'.format(name[:3].upper(),name.upper()),fontsize = 15)        
        
        #### Set axis labels
#        int_xy_ax.set_ylabel('xy displacement (cm)',fontsize = 14)
#        int_xz_ax.set_ylabel('xz displacement (cm)',fontsize = 14)
#        int_xy_ax.set_xlabel('Days',fontsize = 14)
        
#        top_ax.set_ylabel('xy displacement (cm)',fontsize = 14)
#        top_ax.set_xlabel('xz displacement (cm)',fontsize = 14)
#                
        #### Set fig size borders and spacing
        fig.set_figheight(7)
        fig.set_figwidth(10)
        fig.subplots_adjust(right = 0.96,top = 0.94,left = 0.075,bottom = 0.05,hspace = 0.10, wspace = 0.18)
        
        #### Set aspect ratio of aerial view as equal
#        top_ax.set_aspect('equal','box')
        
        #### Set save path
        save_path = "{}/{}/Event {} to {} AOGS/Double Interpolation With Direction Sigma {}".format(data_path,name,cumsheardf.index[0].strftime("%d %b %y"),last_ts.strftime("%d %b %y"),sigma)
        if not os.path.exists(save_path+'/'):
            os.makedirs(save_path+'/')    
        
        #### Save figure
        plt.savefig('{}/node{}.{:04d}.png'.format(save_path,str(nodes[0])+'to'+str(nodes[-1]),fig_num),
                dpi=520, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
        
        #### Set white to transparent
        img = Image.open('{}/node{}.{:04d}.png'.format(save_path,str(nodes[0])+'to'+str(nodes[-1]),fig_num))
        img = img.convert("RGBA")
        datas = img.getdata()
        
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        
        img.putdata(newData)
        img.save('{}/node{}.{:04d}.png'.format(save_path,str(nodes[0])+'to'+str(nodes[-1]),fig_num), "PNG")

        
        #### Close figure
        plt.close()
        
        #### Stop loop if at the final specified frame
        if end_at:
            if fig_num >= end_at:
                print "Stopping at frame {:04d}".format(fig_num)
                break
        
        #### Increment figure number
        fig_num += 1    
        
        

def PlotThresholdsForPaperAOGS(threshold_file,confidence = 0.95,interval = 'confidence'):
    #### Obtain data frame from file
    threshold_df = pd.read_csv(threshold_file,index_col = 0)         

    #### Initialize figure parameters
    fig = plt.figure()
    ax = fig.add_subplot(111)
#    ax.grid()    
    v_range = max(threshold_df.velocity.values) - min(threshold_df.velocity.values)
    all_v = np.linspace(np.exp(np.log(min(threshold_df.velocity.values)) - np.log(v_range)*0.05),np.exp(np.log(max(threshold_df.velocity.values))+np.log(v_range)*0.05),10000)
    plot_num = 1
    marker_type = ['o','s','^','+','x','d']    
    h_map = {}
    #### Loop for all threshold type
    for threshold_type in reversed(np.unique(threshold_df.type.values)):
        
        #### Obtain critical values        
        v = threshold_df[threshold_df.type == threshold_type].velocity.values
        a = threshold_df[threshold_df.type == threshold_type].acceleration.values         
        
        #### Obtain the logarithm of the critical values
#        log_v = np.log(v)
#        log_a = np.log(a)
#        
#        #### Compute the parameters of linear regression and confidence interval
#        slope, intercept, r_value, p_value, std_err = stats.linregress(log_v,log_a)
#        delta = uncertainty(log_v,log_a,slope,intercept,confidence,np.log(all_v),interval)
#        
#        #### Compute the threshold line and confidence interval values
#        a_threshold = np.exp(slope*np.log(all_v) + intercept)
#        a_threshold_upper = np.exp(np.log(a_threshold) + delta)
#        a_threshold_lower = np.exp(np.log(a_threshold) - delta)
        
        #### Plot critical values
        data, = ax.plot(v,a,'.',marker = marker_type[plot_num - 1],color = tableau20[(plot_num -1)*2],label = ' ')        
        
        #### Set handler map
        h_map[data] = HandlerLine2D(numpoints = 1)
        #### Plot all computed values
#        ax.plot(all_v,a_threshold,'-',color = tableau20[(plot_num -1)*2],label = threshold_type.title(),lw=1.5)
#        ax.plot(all_v,a_threshold_upper,'--',color = tableau20[(plot_num -1)*2])
#        ax.plot(all_v,a_threshold_lower,'--',color = tableau20[(plot_num - 1)*2])
        
        ### Plot threshold envelope
#        ax.fill_between(all_v,a_threshold_lower,a_threshold_upper,color = tableau20[(plot_num - 1)*2],alpha = 0.2)
        
        #### Increment plot number        
        plot_num += 1
        
    #### Compute parameters of linear regression for all values
    v = threshold_df.velocity.values
    a = threshold_df.acceleration.values
    log_v = np.log(v)
    log_a = np.log(a)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_v,log_a)
    delta = uncertainty(log_v,log_a,slope,intercept,confidence,np.log(all_v),interval)
    a_threshold = np.exp(slope*np.log(all_v) + intercept)
    a_threshold_upper = np.exp(np.log(a_threshold) + delta)
    a_threshold_lower = np.exp(np.log(a_threshold) - delta)
    ax.plot(all_v,a_threshold,'-',color = tableau20[(plot_num -1)*2],label = ' ',lw=1.5)
    ax.plot(all_v,a_threshold_upper,'--',color = tableau20[(plot_num -1)*2])
    ax.plot(all_v,a_threshold_lower,'--',color = tableau20[(plot_num - 1)*2])
    ax.fill_between(all_v,a_threshold_lower,a_threshold_upper,color = tableau20[(plot_num - 1)*2],alpha = 0.2)
        
    #### Set x and y labels and scales
    ax.set_xscale('log')        
    ax.set_yscale('log')
    ax.set_xlabel('Velocity (cm/day)',fontsize = 14)
    ax.set_ylabel('Acceleration(cm/day$^2$)',fontsize = 14)
 
    #### Set xlim and ylim
    v_max = max(all_v)
    v_min = min(all_v)
    a_max = max(a_threshold_upper)
    a_min = min(a_threshold_lower)
    ax.set_xlim(v_min,v_max)
    ax.set_ylim(a_min,a_max)
   
    #### Plot labels and figure title
    legend = ax.legend(loc = 'upper left', fancybox = True, framealpha = 0.5,handler_map = h_map)
    legend.get_frame().set_visible(False )
#    fig.suptitle('Velocity vs. Acceleration Threshold Line for Subsurface Movement',fontsize = 15)
    

    #### Remove spines
    ax.spines["top"].set_visible(False)     
    ax.spines["right"].set_visible(False)    
    
    plt.setp(ax.spines.values(), color=np.array((248,153,29))/255.)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=np.array((248,153,29))/255.)
    ax.xaxis.label.set_color(np.array((248,153,29))/255.)
    ax.yaxis.label.set_color(np.array((248,153,29))/255.)
    
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    
    #### Set tick width
    ax.tick_params(width = 2,length = 5)


    #### Write anchored text of threshold type
#    threshold_type_at = AnchoredText("{}% {} Interval".format(round(confidence*100,2),interval.title()),prop=dict(size=10), frameon=False,loc = 4)        
#    ax.add_artist(threshold_type_at)
    
    #### Set fig size borders and spacing
    fig.set_figheight(7.5)
    fig.set_figwidth(10)
    fig.subplots_adjust(right = 0.95,top = 0.92,left = 0.09,bottom = 0.09)
    
    #### Set save path
    save_path = "{}".format(data_path)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    now = pd.to_datetime(datetime.now())
    
    #### Save figure
    plt.savefig('{}/Velocity vs Acceleration {} {} Threshold Line {} For AOGS.png'.format(save_path,round(confidence*100,2),interval.title(),now.strftime("%Y-%m-%d_%H-%M")),
            dpi=160, facecolor='w', edgecolor='w',orientation='landscape',mode='w')
    
    #### Set white to transparent
    img = Image.open('{}/Velocity vs Acceleration {} {} Threshold Line {} For AOGS.png'.format(save_path,round(confidence*100,2),interval.title(),now.strftime("%Y-%m-%d_%H-%M")))
    img = img.convert("RGBA")
    datas = img.getdata()
    
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    
    img.putdata(newData)
    img.save('{}/Velocity vs Acceleration {} {} Threshold Line {} For AOGS.png'.format(save_path,round(confidence*100,2),interval.title(),now.strftime("%Y-%m-%d_%H-%M")), "PNG")
