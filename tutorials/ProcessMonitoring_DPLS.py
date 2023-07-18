##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##             DPLS-based Process Monitoring using PDStoolkit package
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np
from PDStoolkit import PDS_DPLS

#%% read training and faulty/test data
trainingData = np.loadtxt('d00.dat').T # Tennessee Eastman dataset
FaultyData =  np.loadtxt('d05_te.dat')

#%% separate inputs and outputs in datasets
X_train, Y_train = trainingData[:,41:52], trainingData[:,0:22]
X_test, Y_test = FaultyData[:,41:52], FaultyData[:,0:22]

#%% fit DPLS model
dpls = PDS_DPLS(n_lags=2)
dpls.fit(X_train, Y_train, autoFindNLatents=True, ratioThreshold=0.99) #scaling is by default

metrics_train = dpls.computeMetrics(X_train, Y_train, isTrainingData=True)
T2_CL, SPEx_CL, SPEy_CL = dpls.computeThresholds(method='percentile', percentile=99)
dpls.draw_monitoring_charts(title='training data')

# dpls.fit_4_monitoring(X_train, Y_train, autoFindNLatents=True, ratioThreshold=0.99, method='percentile', percentile=99)

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ##                 control charts for test data
# ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
metrics_test = dpls.computeMetrics(X_test, Y_test)
dpls.draw_monitoring_charts(metrics=metrics_test, logScaleY=True, title='Fault Class 5 data')

# abnormalityFlags = dpls.detect_abnormalities(X_test, Y_test, title='Fault Class 5 data')
