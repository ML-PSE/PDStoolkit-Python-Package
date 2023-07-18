##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##             CVA-based Process Monitoring using PDStoolkit package
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np
from PDStoolkit import PDS_CVA

#%% read training and faulty/test data
trainingData = np.loadtxt('d00.dat').T  # Tennessee Eastman dataset
FaultyData =  np.loadtxt('d05_te.dat')

#%% separate inputs and outputs in datasets
uData_training, yData_training = trainingData[:,41:52], trainingData[:,0:22]
uData_test, yData_test = FaultyData[:,41:52], FaultyData[:,0:22]

#%% fit CVA model
cva = PDS_CVA(n_lags=3, n_components=29)
cva.fit(uData_training, yData_training)

metrics_train = cva.computeMetrics(uData_training, yData_training, isTrainingData=True)
Ts2_CL, Te2_CL, Q_CL = cva.computeThresholds()
cva.draw_monitoring_charts(metrics=metrics_train, title='training data')

#cva.fit_4_monitoring(uData_training, yData_training)

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ##                 control charts for test data
# ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
metrics_test = cva.computeMetrics(uData_test, yData_test)
cva.draw_monitoring_charts(metrics=metrics_test, logScaleY=True, title='Fault Class 5 data')

