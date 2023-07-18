##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##             DPCA-based Process Monitoring using PDStoolkit package
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np
from sklearn.preprocessing import StandardScaler
from PDStoolkit import PDS_DPCA

#%% read training and faulty/test data
trainingData = np.loadtxt('d00.dat').T  # Tennessee Eastman dataset
FaultyData =  np.loadtxt('d05_te.dat')

#%% select variables
Data_training = np.hstack((trainingData[:,0:22], trainingData[:,41:52]))
Data_faulty = np.hstack((FaultyData[:,0:22], FaultyData[:,41:52]))

#%% scale data
scaler = StandardScaler()
Data_training_scaled = scaler.fit_transform(Data_training)
Data_faulty_scaled = scaler.transform(Data_faulty)

#%% fit DPCA model
dpca = PDS_DPCA(n_lags=2) # Number of lags taken from the work of Yin et al., https://doi.org/10.1016/j.jprocont.2012.06.009
dpca.fit(Data_training_scaled, autoFindNLatents=True)

T2_train, SPE_train = dpca.computeMetrics(Data_training_scaled, isTrainingData=True)
T2_CL, SPE_CL = dpca.computeThresholds(method='percentile', percentile=99)
dpca.draw_monitoring_charts(title='training data')

#dpca.fit_4_monitoring(Data_training_scaled, autoFindNLatents=True, method='percentile', percentile=99)

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ##                 control charts for test data
# ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
metrics_test = dpca.computeMetrics(Data_faulty_scaled)
dpca.draw_monitoring_charts(metrics=metrics_test, logScaleY=True, title='Fault Class 5 data')

# abnormalityFlags = dpca.detect_abnormalities(Data_faulty_scaled, title='Fault Class 5 data')