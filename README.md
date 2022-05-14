# PDStoolkit

### Table of Contents
1. [Project Description](#desc)
2. [Documentation & Tutorials](#docs)
3. [Package Contents](#content)
4. [Installation](#install)

## Description <a name="desc"></a>
The PDStoolkit (Process Data Science Toolkit) package has been created to provide easy-to-use modules to help quickly build data-based solutions for process systems such as those for process monitoring, modeling, fault diagnosis, system identification, etc. Current modules in the package are wrappers around pre-existing Sklearn's classes and provide several additional methods to facilitate a process data scientist's job. Details on these are provided in the following section. More modules relevant for process data science will be added over time.

## Documentation and Tutorials <a name="docs"></a>
- Class documentations are provided in the 'docs' folder
- Tutorials are provided in the 'tutorials' folder
- The blog post (https://mlforpse.com/intro-to-pdstoolkit-python-package/) gives some perspective behind the motivation for development of PDStoolkit package 
- Theoretical and conceptual details on specific algorithms can be found in our book (https://leanpub.com/machineLearningPSE) 

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
 
## Instalation and Usage <a name="install"></a>
Installation from Pypi:

    pip install PDStoolkit

Import modules

    from PDStoolkit import PDS_PCA
    from PDStoolkit import PDS_PLS
    
### License
All code is provided under a BSD 3-clause license. See LICENSE file for more information.
