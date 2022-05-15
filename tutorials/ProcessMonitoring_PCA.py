##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##               PCA model for Process Monitoring using PDStoolkit package
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PDStoolkit import PDS_PCA

#%% fetch data
data = pd.read_excel('proc1a.xlsx', skiprows = 1,usecols = 'C:AI')

# separate training data
data_train = data.iloc[0:69,]
           
# scale data
scaler = StandardScaler()
data_train_normal = scaler.fit_transform(data_train)
           
#%% fit PDS_PCA model
pca = PDS_PCA()
pca.fit(data_train_normal, autoFindNLatents=True)

T2_train, SPE_train = pca.computeMetrics(data_train_normal, isTrainingData=True)
T2_CL, SPE_CL = pca.computeThresholds(method='statistical', alpha=0.01)
pca.draw_monitoring_charts(title='training data')

#pca.fit_4_monitoring(data_train_normal, autoFindNLatents=True, method='statistical', alpha=0.01)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    Test data: detect and diagnose
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% get test data, normalize it
data_test = data.iloc[69:,]
data_test_normal = scaler.transform(data_test) # using scaling parameters from training data
pca.detect_abnormalities(data_test_normal, title='test data')
T2_contri, SPE_contri = pca.get_contributions(data_test_normal[15,:])
