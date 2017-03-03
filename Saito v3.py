import os
import sys
from datetime import datetime, date, time, timedelta
import pandas as pd
from pandas.stats.api import ols
import numpy as np
import matplotlib.pyplot as plt
import ConfigParser
from collections import Counter
import csv
import fileinput
from scipy import interpolate, optimize
from matplotlib.patches import Rectangle

path = os.path.abspath('C:\Users\Win8\Documents\Dynaslope\Data Analysis\updews-pycodes\Analysis')
if not path in sys.path:
    sys.path.insert(1,path)

from querySenslopeDb import *

def GetGroundDF():
    try:

        query = 'SELECT timestamp, meas_type, site_id, crack_id, observer_name, meas, weather, reliability FROM gndmeas'
        
        df = GetDBDataFrame(query)
        return df
    except:
        raise ValueError('Could not get sensor list from database')


###### Directories
out_path = 'C:\Users\Win8\Documents\Dynaslope\Development-Codes\\Saito v3\\'
for paths in [out_path]:
    if not os.path.exists(paths):
        os.makedirs(paths)



#######################################
########   Fixed Parameters   #########
#######################################

site = 'msl'
crack = 'F'
size = 4
sample_size = str(size)+'H'
datetime_start = '2016-10-07 07:00'
datetime_end = '2016-11-04 09:00'
n_from_datum = 7

##### Get and initialize data

raw_data = GetGroundDF()
raw_data.set_index(['timestamp'],inplace = True)
raw_data['site_id'] = map(lambda x: x.lower(),raw_data['site_id'])
raw_data['crack_id'] = map(lambda x: x.title(),raw_data['crack_id'])
raw_data=raw_data[raw_data['meas']!=np.nan]
raw_data=raw_data[raw_data['site_id']!=' ']


raw_data=raw_data.dropna(subset=['meas'])
data = raw_data[raw_data['site_id']==site]
data = data.loc[data.crack_id == crack,['meas']]
data.sort_index(inplace = True)


##### Splice data into specified date range

if datetime_start == '':
    datetime_start = None
if datetime_end == '':
    datetime_end = None
data = data[datetime_start:datetime_end]

#### Choose starting interval from n = 1 to n_from_datum
data_stops = data[n_from_datum-1:]

#### Initialize rupture life curve time displacement values
rup_t = []
rup_x = []

##### Solve time to failure for the whole date ranges, use starting time as t1
for cur_end in data_stops.index:
    #### Splice and resample the data
    cur_data = data[datetime_start:cur_end]
    resampled_data = cur_data.resample(sample_size,how = 'mean',base = 0)
    resampled_data = resampled_data.interpolate()
    
    #### Solve for numerical value of time and displacement
    time_delta = resampled_data.index - resampled_data.index[0]
    t = map(lambda x: x/np.timedelta64(1,'D'),time_delta)
    x =np.array(resampled_data.meas.values)
    
    #### Solve for the interpolation function of resampled time series    
    f = interpolate.interp1d(t,x)
    
    #### Compute for t1, t2 and t3
    t1 = t[0]
    t3 = t[-1]
    t2 = optimize.ridder(lambda y:f(y)-0.5*(f(t1)+f(t3)),t1,t3)
    
    #### Compute for time to failure tr based from Saito (1979)
    tr = (t2**2.0 - t1*t3) / (2.0*t2 - (t1 + t3))
    
    rup_x.append(cur_data.meas.values[-1])
    rup_t.append(tr)

#Solve for t x for whole data
all_time_delta = data.index - data.index[0]
all_t = map(lambda x:  x/np.timedelta64(1,'D'),all_time_delta)
all_x = data.meas.values

#Plot the results

#Tableu 20 Colors
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

fig = plt.figure()
fig.set_size_inches(10,7.5)
ax = fig.add_subplot(111)
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
ax.grid()
plt.xlim(all_t[0],all_t[-1]+ 0.05*(all_t[-1]-all_t[0]))
plt.ylim(all_x[0],all_x[-1] + 0.05*(all_x[-1]-all_x[0]))
ax.plot(all_t,all_x,'.',c = tableau20[0],ms = 7)
l1 = ax.plot(all_t,all_x,'-',c = tableau20[0],lw = 1.5,label = 'Creep curve')
l2 = ax.plot(rup_t,rup_x,'--',c = tableau20[4],lw = 1.5, label = 'Estimated rupture-life curve')
ax.set_xlabel('Time (days)',fontsize = 14)
ax.set_ylabel('Displacement (cm)',fontsize = 14)

extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0,label = 'Sample size = {}'.format(sample_size))
lns = l1 + l2
lns.append(extra)
labs = [l.get_label() for l in lns]

ax.legend(lns,labs,loc = 'upper left',fancybox = True, framealpha=0.5)
fig.suptitle('{} Crack {} Creep-Rupture Life Curve'.format(site.upper(),crack),fontsize = 16)

fig_out_path = '{}{} Crack {} {} - {} {} ndatum {}'.format(out_path,site.upper(),crack,data.index[0].strftime("%Y-%m-%d_%H-%M-%S"),data.index[-1].strftime("%Y-%m-%d_%H-%M-%S"),sample_size,n_from_datum)
plt.savefig(fig_out_path,facecolor='w', edgecolor='w',orientation='landscape',mode='w',bbox_inches = 'tight')
