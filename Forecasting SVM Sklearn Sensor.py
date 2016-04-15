import os
from datetime import datetime, date, time, timedelta
import pandas as pd
from pandas.stats.api import ols
import numpy as np
import matplotlib.pyplot as plt
import ConfigParser
from collections import Counter
import csv
import fileinput
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from matplotlib.patches import Rectangle

def svm_regression(x,y,C,gamma):
    clf = svm.SVR(C = C, gamma = gamma,epsilon = 0.005)
    clf.fit(x,y)
    return clf

def svm_eval(clf,t):
    x = []
    for i in t:
        x.append(clf.predict([i])[0])
    return x

def svm_execute(C,gamma):
    clf = svm_regression(z[:int(len(time_data)/2)],x_data[:int(len(time_data)/2)],C,gamma)
    x = svm_eval(clf,t)
    plt.plot(t,x)
    print C, gamma

#from raw ground displacement data to SVM package
def input_output_data(df,n):
    x = []    
    for i in range(len(df)-n):
        x.append(df[i:i+n])
    return x,df[n:]

#from SVM package (results) to displacement
def out_in_data(x,y):
    z = []
    for i in range(len(x[0])):
        z.append(x[0][i])
    for i in range(len(y)):
        z.append(y[i])
    return z


method1 = 'mean'
site = 'blct'
crack = 12
steps_fwd = 5
size = 4
sample_size = str(size)+'H' 

raw_data = pd.read_csv('C:\Users\Win8\Documents\Dynaslope\Data Analysis\\'+site+'.csv',names = ['ts','node','xz','xy'], parse_dates=['ts'],index_col=0)
raw_data=raw_data.loc[raw_data.node == crack,['xz']]
data = raw_data
data.sort_index(inplace = True)


data = data[:]

data['tvalue'] = data.index
data['delta'] = (data['tvalue']-data.index[0])
data = data.drop_duplicates()
data['t'] = data['delta'].apply(lambda x: x  / np.timedelta64(1,'D'))
data['x'] = (data['xz']-np.array(data['xz'])[0])/100
data2 = data
data3 = data

#resampling data
#if method1 == 'pad':
#    data = data.resample(sample_size, base = 0,label = 'right',closed = 'right').fillna(method = method1)
#elif method1 == 'mean':
#    data = data.resample(sample_size, base = 0,label = 'right',closed = 'right')
#    data = data.interpolate()
    
data['im_disp'] = data['x'] - data['x'].shift()

#process the input data
input_data = data['im_disp'].dropna().values

#scale the input data
scale = max(input_data)
input_data = input_data/scale

#get the SVM ready data
x_data, y_data = input_output_data(input_data,steps_fwd)

#split the data into training and test data sets
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size = 0.25,random_state = 0)

#Set the parameters by cross validation
test_C = [2**-5,2**-4,2**-3,2**-2,2**-1,2**0,2**1,2**2,2**3,2**4,2**5]
test_gamma = test_C

tuned_parameters = [{'kernel':['rbf'],'gamma':test_gamma,'C':test_C,'epsilon' : [0.005]}]

#find the optimal cross validated predictor
clf = GridSearchCV(SVR(C=1),tuned_parameters,cv = 10 , scoring = 'mean_absolute_error')
clf.fit(x_train,y_train)

#evaluate the predictor function using the test data
y_test_fit = clf.predict(x_test)
mad_test = np.sum(np.abs(np.array(y_test)-np.array(y_test_fit)))/len(y_test)
mad_test_label = 'MAD (test) = '+str(np.round(mad_test*scale,5))
#get the predicted data
y_fit = clf.predict(x_data)
y_fit = np.concatenate((x_data[0],y_fit))
im_disp_fit = y_fit*scale
im_disp_fit = np.insert(im_disp_fit,0,0)

#input to the dataframe and compte for the actual displacement
data['im_pred'] = im_disp_fit
data['x_pred'] = data['im_pred'] + data['x'].shift().fillna(data['x'][0])

#evaluate the mean average deviation of the whole data set
mad_fit = np.sum(np.abs(np.array(y_data)-np.array(y_fit[steps_fwd:])))/len(y_test)
mad_fit_label = 'MAD (all) = ' + str(mad_fit*scale)
print mad_fit_label
#plotting the predicted, smoothened, and actual values of the displacement
fig = plt.figure(figsize = (20,8))
f1, = fig.add_subplot(111).plot(data.index,data['x'].values,'.-',label = 'Smoothened Data',markersize = 8)
f2, = fig.add_subplot(111).plot(data.index,data['x_pred'].values,'.--',label ='SVM Prediction', markersize = 8)
f3, = fig.add_subplot(111).plot(data3.index,data3['x'].values,'.',markersize = 8,label = 'Raw Data')

#add the mean average deviation as measuring parameter
MAD1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label = mad_test_label)
MAD2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label = mad_fit_label)
plt.xlabel('Timestamp')
plt.ylabel('Displacement (meters)')
plt.legend(handles=[f1, f2, f3, MAD1,MAD2], loc = 'best')
plt.savefig('C:\Users\Win8\Documents\Dynaslope\Data Analysis\SVM Algorithm\\figs sensor new\\'+site+' '+str(crack)+' '+' '+data.index[0].strftime('%Y-%m-%d %H-%M')+' to '+data.index[-1].strftime('%Y-%m-%d %H-%M') +' '+str(sample_size)+' '+method1+' '+'.png',
                            dpi=500, facecolor='w', edgecolor='w',orientation='landscape',mode='w')