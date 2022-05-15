"""
A module for customized PCA class for process monitoring and fault diagnosis.
Monitoring methodology is described is our book  'Machine Learning for Process Systems Engineering' (https://leanpub.com/machineLearningPSE)
============================================================================
"""

# Version: 2022
# Author: Ankur Kumar @ MLforPSE.com
# License: BSD 3 clause

#%% Imports
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

#%% define class
class PDS_PCA(PCA):
    ''' This class wraps Sklearn's PCA class to provide the following additional methods
    
    1) computeMetrics: computes the monitoring indices for the supplied data
    2) computeThresholds: computes the thresholds / control limits for the monitoring indices from training data
    3) draw_monitoring_charts: Draw the monitoring charts for the training or test data
    4) detect_abnormalities: Detects if the observations are abnormal or normal samples
    5) get_contributions: Returns abnormality contributions for T2 and SPE for an observation vector
    
    Usage Example
    --------
    >>> from PDS_PCA import PDS_PCA
    >>> from sklearn.preprocessing import StandardScaler
    
    >>> X = np.random.rand(30,5)
    >>> scaler = StandardScaler()
    >>> X_normal = scaler.fit_transform(X)
    
    >>> pca = PDS_PCA()
    >>> pca.fit(X_normal, autoFindNLatents=True)
    >>> metrics_train = pca.computeMetrics(X_normal, isTrainingData=True)
    >>> T2_CL, SPE_CL = pca.computeThresholds(method='percentile')
    >>> pca.draw_monitoring_charts(metrics=metrics_train, title='training data')
    
    >>> X_test = scaler.transform(np.random.rand(30,5))
    >>> abnormalityFlags = pca.detect_abnormalities(X_test, title='Test data')
    '''
    
    def __init(self, **kws):
        self.eig_vals_all = None 
        PCA.__init__(**kws)
    
    def fit(self, X, autoFindNLatents=False, varianceThreshold=0.9):
        """
        Extends the fit method of Sklearn PCA to allow in-built computation of the 'optimal' number of latents.
        
        Parameters:
        ---------------------
        X: ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
            
        autoFindNLatents: bool, optional (default False)
            Bool to indicate whether the number of latents is to be determined by the fit method. 
            If True, number of latents that capture atleast varianceThreshold fraction of the total variance is chosen
            
        varianceThreshold: float, optional (default 0.9)
            Real value between 0 and 1.
            The number of PCs retained is such that fraction of variance captured is atleast varianceThreshold  
        
        Returns
        ---------------------
        self: object
            fitted model
        """
        
        if autoFindNLatents:
            PCA.fit(self, X)
            self.eig_vals_all = self.explained_variance_ # to be used later for threshold calculations 
            
            # decide # of PCs to retain
            explained_variance = 100*self.explained_variance_ratio_ # in percentage
            cum_explained_variance = np.cumsum(explained_variance) # cumulative % variance explained
            selected_Nlatent = np.argmax(cum_explained_variance >= varianceThreshold*100) + 1
            
            # print message for selected_Nlatent
            print('# of latents selected: ', selected_Nlatent)
            
            # plot cumulative % variance curve
            plt.figure()
            plt.plot(range(1,X.shape[1]+1), cum_explained_variance, 'r+', label = 'cumulative % variance explained')
            plt.plot(range(1,X.shape[1]+1), explained_variance, 'b+' , label = '% variance explained by each PC')
            plt.ylabel('Explained variance (in %)'), plt.xlabel('Principal component number')
            plt.title('Variance explained vs # of PCs'), plt.legend()
            plt.show()
            
            # store the computed optimal_Nlatents
            self.n_components = selected_Nlatent
        
        # call the original fit method of PCA class
        PCA.fit(self, X)
        if self.n_components == X.shape[1]:
            self.eig_vals_all = self.explained_variance_
        else:
            PCA_temp = PCA().fit(X)
            self.eig_vals_all = PCA_temp.explained_variance_
            
        return self
        
    def computeMetrics(self, X, isTrainingData=False):
        """
        computes the monitoring indices for the supplied data
        
        Parameters:
        ---------------------
        X: ndarray of shape (n_samples, n_features)
            n_samples is the number of samples and n_features is the number of features.
            
        isTrainingData: bool, optional (default False)
            If True, then the computed metrics are stored as object properties which are utilized during computation of metric thresholds
                          
        Returns:
        ---------------------
        SPE: ndarray of shape (n_samples, )
            Also called Q metrics
            
        T2: ndarray of shape (n_samples, ) 
            Also called Hotelling's T-square.
        """
        
        # compute SPE
        Xscores = self.transform(X)
        X_reconstruct = self.inverse_transform(Xscores) 
        X_error = X - X_reconstruct
        SPE = np.sum(X_error*X_error, axis = 1)
        
        # compute T2
        if isTrainingData:
            T_cov = np.cov(Xscores.T) # equivalent to np.diag(pca.explained_variance_)
            T_cov_inv = np.linalg.inv(T_cov)
            self.T_cov_inv = T_cov_inv
            
        T2 = np.zeros((X.shape[0],))
        for i in range(X.shape[0]):
            T2[i] = np.dot(np.dot(Xscores[i,:],self.T_cov_inv),Xscores[i,:].T)
         
        # save metrics for training data as attributes
        if isTrainingData:
            self.T2_train = T2
            self.SPE_train = SPE
        
        return T2, SPE
    
    def computeThresholds(self, method='percentile', percentile=99, alpha=0.01):
        """
        computes the thresholds / control limits for the monitoring indices from training data
        
        Parameters:
        ---------------------
        method: 'percentile' or 'statistical'; default 'percentile'
            
        percentile: int, optional (default 99)
            The percentile value to use if method='percentile'
            
        alpha: int, optional (default 99)
            The significance level to use if method='statistical'. 
            Default value of 0.01 imples 99% control limit.
                        
        Returns
        ---------------------
        T2_CL: float
            Control limit/threshold for T2 metric
        
        SPE_CL: float
            Control limit/threshold for SPE metric
        """
        
        if not hasattr(self, 'T2_train'):
            raise AttributeError('Training metrices not found. Run computeMetrics method before computing the thresholds.')
            
        if method == 'percentile':
            T2_CL = np.percentile(self.T2_train, percentile)
            SPE_CL = np.percentile(self.SPE_train, percentile)
                       
        elif method == 'statistical':
            # parameters
            N = self.T2_train.shape[0]
            k = self.n_components
            m = len(self.eig_vals_all)
            
            # T2_CL
            T2_CL = k*(N**2-1)*scipy.stats.f.ppf(1-alpha,k,N-k)/(N*(N-k))

            # SPE_CL
            theta1 = np.sum(self.eig_vals_all[k:])
            theta2 = np.sum([self.eig_vals_all[j]**2 for j in range(k,m)])
            theta3 = np.sum([self.eig_vals_all[j]**3 for j in range(k,m)])
            if theta3 == 0:
                raise ValueError('Division by zero encountered. Statistical threshold computation for SPE_CL is not possible')
            h0 = 1-2*theta1*theta3/(3*theta2**2)
            z_alpha = scipy.stats.norm.ppf(1-alpha)
            SPE_CL = theta1*(z_alpha*np.sqrt(2*theta2*h0**2)/theta1+ 1 + theta2*h0*(1-h0)/theta1**2)**2 
        
        else:
            raise ValueError('Incorrect choice of method parameter')
            
        # save control limits as attributes
        self.T2_CL = T2_CL
        self.SPE_CL = SPE_CL
                
        return T2_CL, SPE_CL
    
    def draw_monitoring_charts(self, metrics=None, title=''):
        """ 
        Draw the monitoring charts for the training or test data.
        The control limits are plotted as red dashed line. 
        
        Parameters:
        ---------------------
        metrics: list or tuple of monitoring metrics (1D numpy arrays). Should follow the order (T2, SPE)
            If not specified, then the object's stored metrics from training data are used.
                
        title: str, optional 
            Title for the charts                      
        """
        
        if metrics is None:
            metrics = (self.T2_train, self.SPE_train)
        
        # T2
        plt.figure()
        plt.plot(metrics[0], color='k', linestyle = ':', marker='o', markerfacecolor = 'C4')
        if hasattr(self, 'T2_CL'):
            plt.axhline(self.T2_CL, color = "red", linestyle = "-.")
        plt.xlabel('Sample #'), plt.ylabel('T2')
        plt.title(title), plt.show()
        
        # SPE
        plt.figure()
        plt.plot(metrics[1], color='k', linestyle = ':', marker='o', markerfacecolor = 'C4')
        if hasattr(self, 'SPE_CL'):
            plt.axhline(self.SPE_CL, color = "red", linestyle = "-.")
        plt.xlabel('Sample #'), plt.ylabel('SPE')
        plt.title(title), plt.show()
    
    def fit_4_monitoring(self, X, autoFindNLatents=False, varianceThreshold=0.9, method='percentile', percentile=99, alpha=0.01):
        """
        A utility method that calls the following model-training related methods in succession.
        
        fit(X, autoFindNLatents, varianceThreshold)
        computeMetrics(X, isTrainingData=True)
        computeThresholds(method, percentile, alpha)
        draw_monitoring_charts(title='training data')
        
        Note: Check doc string of individual methods for details on the method parameters.
        """
        
        self.fit(X, autoFindNLatents, varianceThreshold)
        self.computeMetrics(X, isTrainingData=True)
        self.computeThresholds(method, percentile, alpha)
        self.draw_monitoring_charts(title='training data')
        
        return self
    
    def detect_abnormalities(self, X, drawMonitoringChart=True, title=''):
        """
        Detects if the observations are abnormal or normal. 
        Detection is based on the logic that for a 'normal' sample, the monitoring metrics should lie below their thresholds.
        
        Parameters:
        ---------------------
        X: ndarray of shape (n_samples, n_features)
            n_samples is the number of samples and n_features is the number of features.
        
        drawMonitoringChart: bool, optional (default True)
            If True, then the monitoring charts are also drawn.
        
        title: str, optional 
            Title for the charts     
                                          
        Returns:
        ---------------------
        abnormalityFlags: ndarray of shape (n_samples,)
            Returns True for abnormal samples and False for normal samples
        """
        
        T2, SPE = self.computeMetrics(X)
        abnormalityFlags = np.logical_or(T2 > self.T2_CL, SPE > self.SPE_CL)
        print('Number of abnormal sample(s): ', np.sum(abnormalityFlags))
        
        if drawMonitoringChart:
            self.draw_monitoring_charts(metrics=(T2, SPE), title=title)
                
        return abnormalityFlags
    
    def get_contributions(self, x, plotContributions=True, title=''):
        """
        Returns abnormality contributions for T2 and SPE metrics for an observation vector.
        
        Parameters:
        ---------------------
        x: ndarray of shape (n_features,)
            Observation sample where n_features is the number of features
                                          
        Returns:
        ---------------------
        T2_contributions: ndarray of shape (n_features,)
            Contains contribution made to the T2 metric from each feature
                
        SPE_contributions: ndarray of shape (n_features,)
            Contains contribution made to the SPE metric from each feature
        """
        
        # check consistency of x
        if x.ndim != 1:
            raise ValueError('get_contributions method expects a 1D observation sample')
        
        # T2 contribution
        P_matrix = self.components_.T
        D = np.dot(np.dot(P_matrix,self.T_cov_inv),P_matrix.T)
        D_sqrt = scipy.linalg.sqrtm(D).real
        T2_contri = np.dot(D_sqrt, x)**2
        
        # SPE contribution
        x = x[None,:] # convert 1D array to 2D
        x_reconstruct = self.inverse_transform(self.transform(x)) 
        x_error = x - x_reconstruct
        SPE_contri = (x_error*x_error)[0,:]
        
        # plots
        if plotContributions:
            plt.figure()
            plt.plot(range(1,len(T2_contri)+1), T2_contri, color='k', linestyle = ':', marker='o', markerfacecolor = 'C4')
            plt.xlabel('Variable #'), plt.ylabel('T$^2$ contribution plot')
            plt.grid(), plt.show()
    
            
            plt.figure()
            plt.plot(range(1,len(SPE_contri)+1), SPE_contri, color='k', linestyle = ':', marker='o', markerfacecolor = 'C4')
            plt.xlabel('Variable #'), plt.ylabel('SPE contribution plot')
            plt.grid(), plt.show()
                
        return T2_contri, SPE_contri
        
        
        
        