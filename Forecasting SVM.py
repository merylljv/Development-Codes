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
from sklearn import svm
from matplotlib.patches import Rectangle

method1 = 'mean'
method2 = 'mean'
site = 'Nin'
crack = 'Crack B'
size = 4
sample_size = str(size)+'D' 
raw_data = pd.read_csv('C:\Users\Win8\Documents\Dynaslope\Data Analysis\\data.csv', parse_dates=['timestamp'],index_col=0)
raw_data=raw_data[raw_data['disp']!=np.nan]
raw_data=raw_data[raw_data['Site id']!=' ']
raw_data=raw_data[raw_data['id']!=np.nan]
raw_data=raw_data.dropna(subset=['disp'])
data = raw_data[raw_data['Site id']==site]
data = data.loc[data.id == crack,['disp']]
data.sort_index(inplace = True)


data = data[:]

data['tvalue'] = data.index
data['delta'] = (data['tvalue']-data.index[0])
data = data.drop_duplicates()
data['t'] = data['delta'].apply(lambda x: x  / np.timedelta64(1,'D'))
data['x'] = (data['disp']-np.array(data['disp'])[0])/100
data2 = data
data3 = data

#resampling data
if method1 == 'pad':
    data = data.resample(sample_size, base = 0,label = 'right',closed = 'right').fillna(method = method1)
elif method1 == 'mean':
    data = data.resample(sample_size, base = 0,label = 'right',closed = 'right')
    data = data.interpolate()
    
data['im_disp'] = data['x'] - data['x'].shift()

if method2 == 'pad':
    data2 = data2.resample(sample_size, base = 0,label = 'right',closed = 'right').fillna(method = method2)
elif method2 == 'mean':
    data2 = data2.resample(sample_size, base = 0,label = 'right',closed = 'right')
    data2 = data.interpolate()


data2['im_disp'] = data2['x'] - data2['x'].shift()
#traning data
training_data_df = data[:int(len(data)/2)].dropna()
validation_data_df = data[int(len(data))/2:int(len(data)*3/4)]
testing_data_df = data2[int(len(data)*0.75):]

training_data = training_data_df['im_disp'].values
validation_data = validation_data_df['im_disp'].values
testing_data = testing_data_df['im_disp'].values


print training_data
print validation_data
print testing_data

print data
time_data = data['t'].values
x_data = data['x'].values

z = []

t = np.arange(0,time_data[-1]/max(time_data)+0.5,0.01)

for i in time_data:
    z.append([i/65])

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
#plt.plot(time_data/max(time_data),x_data,'.-',markersize = 12)

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

#scaling data
scale = max(training_data)
training_data = training_data/scale
validation_data = validation_data/scale
testing_data = testing_data/scale

#Test parameters for C and gamma
test_C = [2**-5,2**-4,2**-3,2**-2,2**-1,2**0,2**1,2**2,2**3,2**4,2**5]
test_gamma = test_C

#using the validation data to get the best parameters of C and gamma
x_train, y_train = input_output_data(training_data)
x_valid, y_valid = input_output_data(validation_data)
x_test, y_test = input_output_data(testing_data)

min_mad = 1e10
max_parameters = [0,0]
for C in test_C:
    for gamma in test_gamma:
        #train function
        clf = svm_regression(x_train,y_train,C,gamma)
        #test on validation data
        y_valid_test = []
        for i in x_valid:
            y_valid_test.append(clf.predict(i)[0])
        #evaluate validation data
        mad = np.sum(np.abs(np.array(y_valid_test)-np.array(y_valid)))/len(y_valid_test)
        if mad < min_mad:
            min_mad = mad
            max_parameters = [C,gamma]

print max_parameters, mad

#testing most efficienct clf
clf = svm_regression(x_train,y_train,max_parameters[0],max_parameters[1])

y_predict_test = []
y_predict_valid = []
y_predict_train = []

for i in x_test:
    y_predict_test.append(clf.predict(i)[0])

for i in x_valid:
    y_predict_valid.append(clf.predict(i)[0])

for i in x_train:
    y_predict_train.append(clf.predict(i)[0])

mad_train = np.sum(np.abs(np.array(y_train)-np.array(y_predict_train)))/len(y_train)
mad_test = np.sum(np.abs(np.array(y_test)-np.array(y_predict_test)))/len(y_test)
mad_valid = np.sum(np.abs(np.array(y_valid)-np.array(y_predict_valid)))/len(y_valid)
mad_string = 'MAD = '+str(np.round(mad_test*scale,5))

eff_train = np.sum(np.abs(np.array(y_train)-np.array(y_predict_train))/np.abs(y_train))/len(y_train)
eff_test = np.sum(np.abs(np.array(y_test)-np.array(y_predict_test))/np.abs(y_predict_test))/len(y_test)
eff_valid = np.sum(np.abs(np.array(y_valid)-np.array(y_predict_valid))/np.abs(y_valid))/len(y_valid)

predict_train = np.array(out_in_data(x_train,y_predict_train))*scale
predict_valid = np.array(out_in_data(x_valid,y_predict_valid))*scale
predict_test = np.array(out_in_data(x_test,y_predict_test))*scale

predict_im = np.concatenate((predict_train,predict_valid,predict_test))
predict_im = np.insert(predict_im,0,0)
data['im_pred'] = predict_im
data['x_pred'] = data['im_pred'] + data['x'].shift().fillna(data['x'][0])
data2['x_pred'] = data['im_pred'] + data2['x'].shift().fillna(data['x'][0])
data['x_pred'][int(len(data)*0.75):] = data2['x_pred'][int(len(data)*0.75):]
data['x'][int(len(data)*0.75):] = data2['x'][int(len(data)*0.75):]

fig = plt.figure(figsize = (20,8))
f1, = fig.add_subplot(111).plot(data.index,data['x'].values,'.-',label = 'Smoothened Data',markersize = 8)
f2, = fig.add_subplot(111).plot(data.index,data['x_pred'].values,'.--',label ='SVM Prediction', markersize = 8)
f3, = fig.add_subplot(111).plot(data3.index,data3['x'].values,'.',markersize = 8,label = 'Raw Data')
plt.axvspan(data.index[0],validation_data_df.index[0],facecolor = 'b', alpha = 0.2)
plt.axvspan(validation_data_df.index[0],testing_data_df.index[0],facecolor = 'y', alpha = 0.2)
plt.axvspan(testing_data_df.index[0],testing_data_df.index[-1],facecolor = 'g', alpha = 0.2)
MAD = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label = mad_string)
plt.xlabel('Timestamp')
plt.ylabel('Displacement (meters)')
plt.legend(handles=[f1, f2, f3, MAD], loc = 'best')
plt.savefig('C:\Users\Win8\Documents\Dynaslope\Data Analysis\SVM Algorithm\\figs\\'+site+' '+crack+' '+' '+data.index[0].strftime('%Y-%m-%d %H-%M')+' to '+data.index[-1].strftime('%Y-%m-%d %H-%M') +' '+str(sample_size)+' '+method1+' '+method2+'.png',
                            dpi=500, facecolor='w', edgecolor='w',orientation='landscape',mode='w')