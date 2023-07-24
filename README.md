# PDStoolkit

### Table of Contents
1. [Project Description](#desc)
2. [Documentation & Tutorials](#docs)
3. [Package Contents](#content)
4. [Installation](#install)
5. [Usage](#usage)

## Description <a name="desc"></a>
The PDStoolkit (Process Data Science Toolkit) package has been created to provide easy-to-use modules to help quickly build data-based solutions for process systems such as those for process monitoring, modeling, fault diagnosis, system identification, etc. Current modules in the package are wrappers around pre-existing Sklearn's classes and provide several additional methods to facilitate a process data scientist's job. Details on these are provided in the following section. More modules relevant for process data science will be added over time.

## Documentation and Tutorials <a name="docs"></a>
- PDStoolkit_Manual.pdf provides some quick information on the algorithms implemented in the package
- Class documentations are provided in the 'docs' folder
- Tutorials are provided in the 'tutorials' folder
- The blog post (https://mlforpse.com/intro-to-pdstoolkit-python-package/) gives some perspective behind the motivation for development of PDStoolkit package 
- Theoretical and conceptual details on specific algorithms can be found in our books (https://mlforpse.com/books/) 

## Package Contents <a name="content"></a>
The main modules in the package currently are:

 - **PDS_PCA: Principal Component analysis for Process Data Science**
   - This class is a child of [sklearn.decomposition.PCA class](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) 
   - The following additional methods are provided
     - *computeMetrics*: computes the monitoring indices (Q or SPE, T2) for the supplied data
     - *computeThresholds*: computes the thresholds / control limits for the monitoring indices from training data
     - *draw_monitoring_charts*: draws the monitoring charts for the training or test data
     - *detect_abnormalities*: detects if the observations are abnormal or normal samples
     - *get_contributions*: returns abnormality contributions for T2 and SPE for an observation sample
       
 - **PDS_PLS: Partial Least Squares regression for Process Data Science**
   - This class is a child of [sklearn.cross_decomposition.PLSRegression class](http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html) 
   - The following additional methods are provided
     - *computeMetrics*: computes the monitoring indices (SPEx, SPEy, T2) for the supplied data
     - *computeThresholds*: computes the thresholds / control limits for the monitoring indices from training data
     - *draw_monitoring_charts*: draws the monitoring charts for the training or test data
     - *detect_abnormalities*: detects if the observations are abnormal or normal samples
       
 - **PDS_DPCA: Dynamic Principal Component analysis for Process Data Science**
   - This class is a child of [sklearn.decomposition.PCA class](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) 
   - The following additional methods are provided
     - *computeMetrics*: computes the monitoring indices (Q or SPE, T2) for the supplied data
     - *computeThresholds*: computes the thresholds / control limits for the monitoring indices from training data
     - *draw_monitoring_charts*: draws the monitoring charts for the training or test data
     - *detect_abnormalities*: detects if the observations are abnormal or normal samples
       
 - **PDS_DPLS: Dynamic Partial Least Squares regression for Process Data Science**
   - This class is a child of [sklearn.cross_decomposition.PLSRegression class](http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html) 
   - The following additional methods are provided
     - *computeMetrics*: computes the monitoring indices (SPEx, SPEy, T2) for the supplied data
     - *computeThresholds*: computes the thresholds / control limits for the monitoring indices from training data
     - *draw_monitoring_charts*: draws the monitoring charts for the training or test data
     - *detect_abnormalities*: detects if the observations are abnormal or normal samples
       
 - **PDS_CVA: Canonical Variate Analysis for Process Data Science**
   - This class is written from scratch 
   - The following additional methods are provided
     - *computeMetrics*: computes the monitoring indices (Ts2, Te2, Q) for the supplied data
     - *computeThresholds*: computes the thresholds / control limits for the monitoring indices from training data
     - *draw_monitoring_charts*: draws the monitoring charts for the training or test data
     - *detect_abnormalities*: detects if the observations are abnormal or normal samples
 
## Installation <a name="install"></a>
Installation from Pypi:

    pip install PDStoolkit

Import modules

    from PDStoolkit import PDS_PCA
    from PDStoolkit import PDS_PLS

## Usage <a name="usage"></a>
The following code builds a PCA-based process monitoring model using PDS-PCA class and uses it for subsequent fault detection and fault diagnosis on test data. For details on data and results, see the ProcessMonitoring_PCA notebook in the tutorials folder.

```
# import 
from PDStoolkit import PDS_PCA

# fit PDS_PCA model
pca = PDS_PCA()
pca.fit(data_train_normal, autoFindNLatents=True)

T2_train, SPE_train = pca.computeMetrics(data_train_normal, isTrainingData=True)
T2_CL, SPE_CL = pca.computeThresholds(method='statistical', alpha=0.01)
pca.draw_monitoring_charts(title='training data')

# fault detection and fault diagnosis on test data
pca.detect_abnormalities(data_test_normal, title='test data')
T2_contri, SPE_contri = pca.get_contributions(data_test_normal)
```
    
### License
All code is provided under MIT license. See LICENSE file for more information.
