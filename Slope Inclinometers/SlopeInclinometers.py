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
from scipy.stats import spearmanr
import sys

from scipy.interpolate import UnivariateSpline
from scipy.signal import gaussian
from scipy.ndimage import filters


#### Include Analysis folder of updews-pycodes (HARD CODED!!)
path = os.path.abspath("C:\Users\Win8\Documents\Dynaslope\Data Analysis\updews-pycodes\Analysis")
if not path in sys.path:
    sys.path.insert(1,path)
del path 

import RealtimePlotter as rp
import rtwindow as rtw
import querySenslopeDb as q
import genproc as gp

#### Global Parameters
data_path = os.path.dirname(os.path.realpath(__file__))

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
event_window = pd.Timedelta(30,'D')
colpos_interval = 1 ### in days
window = 3 ### in days


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
        query = "SELECT DISTINCT site FROM senslopedb.node_level_alert WHERE timestamp >= '{}' AND timestamp <= '{}' AND site LIKE '{}%'".format(timestamp[0].strftime("%Y-%m-%d %H:%M:%S"),timestamp[1].strftime("%Y-%m-%d %H:%M:%S"),site)
        sensor_columns = q.GetDBDataFrame(query)
        site_events.loc[(site_events.timestamp == timestamp)&(site_events.site == site),['sensor_columns']] = ', '.join(sensor_columns['site'].values)
    return site_events

def set_zero_disp(colposdf):
    initial_ts = min(colposdf['ts'].values)
    colposdf['xz'] = colposdf['xz'].values - colposdf[colposdf.ts == initial_ts]['xz'].values[0]
    colposdf['xy'] = colposdf['xy'].values - colposdf[colposdf.ts == initial_ts]['xy'].values[0]
    
    return colposdf

def SubsurfaceValidEvents():
    #### This function gets all valid L2/L3 subsurface events then outputs the appropriate start and end timestamp for each site and relevant sensor columns
    
    #### Obtain from site level alerts all L2/L3 candidate triggers
    query = "SELECT * FROM senslopedb.site_level_alert WHERE source = 'internal' AND alert LIKE '%s%' AND alert not LIKE '%s0%' AND alert not LIKE '%ND%' ORDER BY updateTS desc"
    candidate_triggers = q.GetDBDataFrame(query)
    
    #### Filter invalid alerts
    candidate_triggers = CheckIfEventIsValid(candidate_triggers)
    
    #### Set initial and final plotting timestamp
    candidate_triggers['timestamp_start'] = map(lambda x:x - event_window,candidate_triggers['timestamp'])
    candidate_triggers['timestamp_end'] = map(lambda x:x + event_window,candidate_triggers['updateTS'])
    
    #### Remove overlaps and merge timestamps per site
    candidate_triggers_group = candidate_triggers.groupby(['site'],as_index = False)
    site_events = candidate_triggers_group.apply(site_events_to_plot)
    
    #### Determine columns to plot
    site_events['sensor_columns'] = None
    return sensor_columns_to_plot(site_events).reset_index()[['timestamp','site','sensor_columns']]
    
def GetDispAndColPosDataFrame(event_timestamp,sensor_column):
    #### Get all required parameters from realtime plotter
    col = q.GetSensorList(sensor_column)
    window, config = rtw.getwindow(pd.to_datetime(event_timestamp[-1]))
    window.start = pd.to_datetime(event_timestamp[0])
    window.offsetstart = window.start - timedelta(days=(config.io.num_roll_window_ops*window.numpts-1)/48.)
    config.io.col_pos_interval = str(int(colpos_interval)) + 'D'
    config.io.num_col_pos = int((window.end - window.start).days/colpos_interval + 1)
    monitoring = gp.genproc(col[0], window, config, config.io.column_fix,comp_vel = False)
    
    #### Get colname, num nodes and segment length
    num_nodes = monitoring.colprops.nos
    seg_len = monitoring.colprops.seglen
    
    monitoring_vel = monitoring.vel.reset_index()[['ts', 'id', 'xz', 'xy']]
    monitoring_vel = monitoring_vel.loc[(monitoring_vel.ts >= window.start)&(monitoring_vel.ts <= window.end)]
    
    colposdf = rp.compute_colpos(window, config, monitoring_vel, num_nodes, seg_len)
    
    #### Recomputing depth
    colposdf['yz'] = np.sqrt(seg_len**2 - np.power(colposdf['xz'],2) - np.power(colposdf['xy'],2))
    colposdf_ts = colposdf.groupby('ts',as_index = False)
    colposdf = colposdf_ts.apply(compute_depth)
    colposdf['depth'] = map(lambda x:x - max(colposdf.depth.values),colposdf['depth'])
    return monitoring_vel.reset_index(drop = True),colposdf[['ts','id','depth','xy','xz']].reset_index(drop = True),sensor_column

def ComputeCumShear(disp_ts):
    cumsheardf = pd.DataFrame(columns = ['ts','cumshear'])
    cumsheardf.ix['ts'] = disp_ts.ts.values[0]
    
    cum_xz = np.sum(disp_ts.xz.values)
    cum_xy = np.sum(disp_ts.xy.values)
    sum_cum = np.sqrt(np.square(cum_xz) + np.square(cum_xy))
    cumsheardf['cumshear'] = sum_cum
    
    return cumsheardf

def PlotIncrementalDisplacement(colposdf,sensor_column,zeroed = False,zoomin=False):
    #### Set figure and subplots
    fig = plt.figure()
    ax_xz = fig.add_subplot(121)
    ax_xy = fig.add_subplot(122, sharey = ax_xz)
    
    #### Ensure non repeating colors and color scheme = plasma
    ax_xz=rp.nonrepeat_colors(ax_xz,len(set(colposdf.ts.values)),color='plasma')
    ax_xy=rp.nonrepeat_colors(ax_xy,len(set(colposdf.ts.values)),color='plasma')
    
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
        cur_plot = ax_xz.plot(cur_xz,cur_depth,'.-',lw = 1.25,markersize = 10,label = ts.strftime("%d %b '%y"))
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
    fig.set_figwidth(8)
    fig.subplots_adjust(right = 0.795,top = 0.925,left = 0.100)
    
    #### Set save path
    save_path = "{}/{}".format(data_path,sensor_column)
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
    ax_xz=rp.nonrepeat_colors(ax_xz,len(set(colposdf.ts.values)),color='plasma')
    ax_xy=rp.nonrepeat_colors(ax_xy,len(set(colposdf.ts.values)),color='plasma')
    
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
        cur_plot = ax_xz.plot(cur_xz,cur_depth,'.-',lw = 1.25,markersize = 10,label = ts.strftime("%d %b '%y"))
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
    fig.set_figwidth(8)
    fig.subplots_adjust(right = 0.795,top = 0.925,left = 0.100)
    
    #### Set save path
    save_path = "{}/{}".format(data_path,sensor_column)
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
    events_list = SubsurfaceValidEvents()[-2:]
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
    save_path = "{}/{}".format(data_path,name)
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
        ax.plot(cur_cum_shear.ts.values,cur_cum_shear.cumshear.values*100,lw = 2,color = tableau20[(plot_num-1)*2],label = "Nodes "+' '.join(map(lambda x:str(x),set_nodes)))
        
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
    save_path = "{}/{}".format(data_path,name)
    if not os.path.exists(save_path+'/'):
        os.makedirs(save_path+'/')    
    
    #### Save figure
    plt.savefig('{}/DispvTime Multiple {} {} to {} {}.png'.format(save_path,name,pd.to_datetime(min(disp.ts.values)).strftime("%Y-%m-%d_%H-%M"),pd.to_datetime(max(disp.ts.values)).strftime("%Y-%m-%d_%H-%M"),'.'.join(map(lambda x:str(x),nodes))),
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
    cumsheardf['time'] = (cumsheardf.index - cumsheardf.index[0]).apply(lambda x: x / np.timedelta64(1,'D'))
    
    #### Convert displacement to centimeters
    cumsheardf['cumshear'] = cumsheardf.cumshear.apply(lambda x:x*100)
    
    
    #### Get last timestamp in df
    last_ts = cumsheardf.timestamp.values[-1]        
    
    #### Set bounds of slice df according to window
    for ts_start in cumsheardf[cumsheardf.timestamp <= last_ts - window].timestamp.values:
        
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
            
        except:
            print "Interpolation Error"
            
        
    
        


