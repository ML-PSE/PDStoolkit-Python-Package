##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##             PLS model for Process Monitoring using PDStoolkit package
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import pandas as pd
import matplotlib.pyplot as plt
from PDStoolkit import PDS_PLS

#%% fetch data
data = pd.read_csv('LDPE.csv', usecols = range(1,20)).values
data_train = data[:-4,:] # exclude last 4 samples

#%% plot quality variables
quality_var = 5

plt.figure()
plt.plot(data[:,13+quality_var], '*')
plt.xlabel('sample #')
plt.ylabel('quality variable ' + str(quality_var))

#%% build PLS model
X_train = data_train[:,:-5]
Y_train = data_train[:,-5:]

pls = PDS_PLS(n_components = 3)
pls.fit(X_train, Y_train) #scaling is by default

metrics_train = pls.computeMetrics(X_train, Y_train, isTrainingData=True)
T2_CL, SPEx_CL, SPEy_CL = pls.computeThresholds(method='statistical', alpha=0.01)
pls.draw_monitoring_charts(metrics=metrics_train, title='training data')

#pls.fit_4_monitoring(X_train, Y_train, method='statistical', alpha=0.01)

##%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                         fault detection on complete data
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% get test data
X_test = data[:,:-5]
Y_test = data[:,-5:]

abnormalityFlags = pls.detect_abnormalities(X_test, Y_test, title='Test data')