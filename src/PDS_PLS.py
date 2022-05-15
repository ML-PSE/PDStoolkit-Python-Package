"""
A module for customized PLS class for process monitoring. 
Monitoring methodology is described is our book  'Machine Learning for Process Systems Engineering' (https://leanpub.com/machineLearningPSE)
=======================================================
"""

# Version: 2022
# Author: Ankur Kumar @ MLforPSE.com
# License: BSD 3 clause

#%% Imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sys

#%% define class
class PDS_PLS(PLSRegression):
    ''' This class wraps Sklearn's PLS class to provide the following addiitonal methods
    
    1) computeMetrics: computes the monitoring indices for the supplied data
    2) computeThresholds: computes the thresholds / control limits for the monitoring indices from training data
    3) draw_monitoring_charts: Draw the monitoring charts for the training or test data
    4) detect_abnormalities: Detects if the observations are abnormal or normal samples
    
    Usage Example
    --------
    >>> from PDS_PLS import PDS_PLS
    >>> X = np.random.rand(30,5)
    >>> Y = np.random.rand(30,2)
    >>> pls = PDS_PLS()
    >>> pls.fit(X, Y, autoFindNLatents=True)
    >>> metrics_train = pls.computeMetrics(X, Y, isTrainingData=True)
    >>> T2_CL, SPEx_CL, SPEy_CL = pls.computeThresholds(method='statistical', alpha=0.01)
    >>> pls.draw_monitoring_charts(metrics=metrics_train, title='training data')
    
    >>> X_test = np.random.rand(30,5)
    >>> Y_test = np.random.rand(30,2) 
    >>> abnormalityFlags = pls.detect_abnormalities(X_test, Y_test, title='Test data')
    '''
    
    def __init(self, **kws):
        PLSRegression.__init__(**kws)
    
    def fit(self, X, Y, autoFindNLatents=False, n_CVsplits=10, ratioThreshold=0.0):
        """
        Extends the fit method of Sklearn PLS to allow in-built computation of the 'optimal' number of latents.
        
        Parameters:
        ---------------------
        X: ndarray of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and n_features is the number of predictors
            
        Y: ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and n_targets is the number of response variables.
                
        autoFindNLatents: bool, optional (default False)
            Decides whether to compute the 'optimal' number of latents or not. 
            If True, cross-validation is used and by default, number of latents corresponding to the lowest validation MSE is chosen.
            
        n_CVsplits: int, optional (default 10) 
            The number of cross-validation splits to use if autoFindNLatents is True
            
        ratioThreshold: float, optional (default 0)
            Real value between 0 and 1. Value <= 0 => lowest validation MSE chosen.   
            If ratioThreshold > 0, then the number of latents (l) is chosen such that the ratio 
            of validation MSEs with l+1 and l latents is greater than ratioThreshold. 
            The underlying logic is that the number of retained latents is not increased unless significantly better validation prediction is obtained.
            
        Returns
        ---------------------
        self: object
            fitted model
        """
        
        if autoFindNLatents:
            # sanity check on X
            X = self._validate_data(X, dtype=np.float64, copy=self.copy)
            if X.shape[1] < 2:
                raise ValueError("Input data must have atleast 2 features")
            
            # compute fitting and validation errors for different number of latents
            scaler = StandardScaler()
            fit_MSE = []
            validate_MSE = []
            lowest_validation_mse = np.inf
            selected_Nlatent = X.shape[1] # placehoder
            
            for Nlatent in range(1,X.shape[1]+1):
                local_fit_MSEs = []
                local_validate_MSEs = []
                
                kfold = KFold(n_splits = n_CVsplits, shuffle = True, random_state = 100)
                for fit_index, validate_index in kfold.split(Y):
                    X_fit_normal = scaler.fit_transform(X[fit_index])
                    X_validate_normal = scaler.transform(X[validate_index])
                    
                    y_fit_normal = scaler.fit_transform(Y[fit_index])
                    y_validate_normal = scaler.transform(Y[validate_index])
                    
                    pls = PLSRegression(n_components = Nlatent)
                    pls.fit(X_fit_normal, y_fit_normal)
                    
                    local_fit_MSEs.append(mean_squared_error(y_fit_normal, pls.predict(X_fit_normal)))
                    local_validate_MSEs.append(mean_squared_error(y_validate_normal, 
                                                                    pls.predict(X_validate_normal)))
                
                fit_MSE.append(np.mean(local_fit_MSEs))
                validate_MSE.append(np.mean(local_validate_MSEs))
                
                # check if threshold criteria needs to be used
                if ratioThreshold > 0:
                    if Nlatent > 1:
                        mse_ratio =  validate_MSE[-1] / validate_MSE[-2]
                        if mse_ratio > ratioThreshold:
                            selected_Nlatent = Nlatent-1
                            break
                    continue
                
                # check if this is the lowest validation MSE till now
                if np.mean(local_validate_MSEs) < lowest_validation_mse:
                    lowest_validation_mse = np.mean(local_validate_MSEs)
                    selected_Nlatent = Nlatent
            
            # print message for selected_Nlatent
            print('# of latents selected: ', selected_Nlatent)
            
            # plot validation curve
            plt.figure()
            plt.plot(range(1,len(fit_MSE)+1), fit_MSE, 'b*', label = 'Training MSEs')
            plt.plot(range(1,len(fit_MSE)+1), validate_MSE, 'r*', label = 'Validation MSEs')
            plt.xticks(range(1,len(fit_MSE)+1))
            plt.ylabel('Mean Squared Error (MSE)'), plt.xlabel('# of latents')
            plt.title('Validation Curve') , plt.legend()
            plt.show()
            
            # store the computed optimal_Nlatents
            self.n_components = selected_Nlatent
        
        # call the original fit method of PLSRegression class
        PLSRegression.fit(self, X, Y)
        return self
        
    def computeMetrics(self, X, Y, isTrainingData=False):
        """
        computes the monitoring indices for the supplied data.
        
        Parameters:
        ---------------------
        X: ndarray of shape (n_samples, n_features)
            vectors, where n_samples is the number of samples and n_features is the number of predictors
            
        Y: ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and n_targets is the number of response variables.
            
        isTrainingData: bool, optional (default False)
            If True, then the computed metrics are stored as object properties which are utilized during computation of metric thresholds
                          
        Returns:
        ---------------------
        SPEx: ndarray of shape (n_samples, )   
        
        SPEy: ndarray of shape (n_samples, )
        
        T2: ndarray of shape (n_samples, )
        """
        
        # compute SPEx
        Xscores = self.transform(X)
        X_reconstruct = self.inverse_transform(Xscores) 
        X_error = X - X_reconstruct
        X_error_scaled = X_error/self._x_std
        SPEx = np.sum(X_error_scaled*X_error_scaled, axis = 1)
        
        # compute SPEy
        Y_error = Y - self.predict(X)
        Y_error_scaled = Y_error/self._y_std
        SPEy = np.sum(Y_error_scaled*Y_error_scaled, axis = 1)
        
        # compute T2
        if isTrainingData:
            T_cov = np.cov(Xscores.T)
            if T_cov.ndim == 0:
                T_cov_inv = 1/T_cov
            else:
                T_cov_inv = np.linalg.inv(T_cov)
            self.T_cov_inv = T_cov_inv
            
        T2 = np.zeros((X.shape[0],))
        for i in range(X.shape[0]):
            T2[i] = np.dot(np.dot(Xscores[i,:],self.T_cov_inv),Xscores[i,:].T)
        
        # save metrics for training data as attributes
        if isTrainingData:
            self.T2_train = T2
            self.SPEx_train = SPEx
            self.SPEy_train = SPEy
        
        return T2, SPEx, SPEy
    
    def computeThresholds(self, method='percentile', percentile=99, alpha=0.01):
        """
        computes the thresholds/control limits for the monitoring indices from training data
        
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
        
        SPEx_CL: float
            Control limit/threshold for SPEx metric
        
        SPEy_CL: float
            Control limit/threshold for SPEy metric
        """
        
        if not hasattr(self, 'T2_train'):
            raise AttributeError('Training metrices not found. Run computeMetrics method before computing the thresholds.')
            
        if method == 'percentile':
            T2_CL = np.percentile(self.T2_train, percentile)
            SPEx_CL = np.percentile(self.SPEx_train, percentile)
            SPEy_CL = np.percentile(self.SPEy_train, percentile)
            
        elif method == 'statistical':
            N = self.T2_train.shape[0]
            k = self.n_components
            
            # T2_CL
            T2_CL = k*(N**2-1)*scipy.stats.f.ppf(1-alpha,k,N-k)/(N*(N-k))

            # SPEx_CL
            mean_SPEx_train = np.mean(self.SPEx_train)
            var_SPEx_train = np.var(self.SPEx_train)
            
            if mean_SPEx_train == 0:
                mean_SPEx_train = sys.float_info.epsilon
            if var_SPEx_train == 0:
                var_SPEx_train = sys.float_info.epsilon

            g = var_SPEx_train/(2*mean_SPEx_train)
            h = 2*mean_SPEx_train**2/var_SPEx_train
            SPEx_CL = g*scipy.stats.chi2.ppf(1-alpha, h)

            # SPEy_CL
            mean_SPEy_train = np.mean(self.SPEy_train)
            var_SPEy_train = np.var(self.SPEy_train)
            
            if mean_SPEy_train == 0:
                mean_SPEy_train = sys.float_info.epsilon
            if var_SPEy_train == 0:
                var_SPEy_train = sys.float_info.epsilon

            g = var_SPEy_train/(2*mean_SPEy_train)
            h = 2*mean_SPEy_train**2/var_SPEy_train
            SPEy_CL = g*scipy.stats.chi2.ppf(1-alpha, h)
        
        else:
            raise ValueError('Incorrect choice of method parameter')
            
        # save control limits as attributes
        self.T2_CL = T2_CL
        self.SPEx_CL = SPEx_CL
        self.SPEy_CL = SPEy_CL
        
        return T2_CL, SPEx_CL, SPEy_CL
    
    def draw_monitoring_charts(self, metrics=None, title=''):
        """
        Draw the monitoring charts for the training or test data.
        The control limits are plotted as red dashed line. 
        
        Parameters:
        ---------------------
        metrics: (optional) list or tuple of monitoring metrics (1D numpy arrays). Should follow the order (T2, SPEx, SPEy).
            If not specified, then the object's stored metrics from training data are used.
            
        title: str, optional 
            Title for the charts                      
        """
        
        if metrics is None:
            metrics = (self.T2_train, self.SPEx_train, self.SPEy_train)
        
        # T2
        plt.figure()
        plt.plot(metrics[0], color='k', linestyle = ':', marker='o', markerfacecolor = 'C4')
        if hasattr(self, 'T2_CL'):
            plt.axhline(self.T2_CL, color = "red", linestyle = "-.")
        plt.xlabel('Sample #'), plt.ylabel('T2')
        plt.title(title), plt.show()
        
        # SPEx
        plt.figure()
        plt.plot(metrics[1], color='k', linestyle = ':', marker='o', markerfacecolor = 'C4')
        if hasattr(self, 'SPEx_CL'):
            plt.axhline(self.SPEx_CL, color = "red", linestyle = "-.")
        plt.xlabel('Sample #'), plt.ylabel('SPEx')
        plt.title(title), plt.show()
        
        # SPEy
        plt.figure()
        plt.plot(metrics[2], color='k', linestyle = ':', marker='o', markerfacecolor = 'C4')
        if hasattr(self, 'SPEy_CL'):
            plt.axhline(self.SPEy_CL, color = "red", linestyle = "-.")
        plt.xlabel('Sample #'), plt.ylabel('SPEy')
        plt.title(title), plt.show()
    
    def fit_4_monitoring(self, X, Y, autoFindNLatents=False, n_CVsplits=10, ratioThreshold=0.0, method='percentile', percentile=99, alpha=0.01):
        """
        A utility method that calls the following model-training related methods in succession.
        
        fit(X, Y, autoFindNLatents, n_CVsplits, ratioThreshold)
        computeMetrics(X, Y, isTrainingData=True)
        computeThresholds(method, percentile, alpha)
        draw_monitoring_charts(title='training data')  
        
        Note: Check doc string of individual methods for details on the method parameters.
        """
                
        self.fit(X, Y, autoFindNLatents, n_CVsplits, ratioThreshold)
        self.computeMetrics(X, Y, isTrainingData=True)
        self.computeThresholds(method, percentile, alpha)
        self.draw_monitoring_charts(title='training data')
        
        return self
    
    def detect_abnormalities(self, X, Y, drawMonitoringChart=True, title=''):
        """
        Detects if the observations are abnormal or normal. 
        Detection is based on the logic that for a 'normal' sample, the monitoring metrics should lie below their thresholds.
        
        Parameters:
        ---------------------
        X: ndarray of shape (n_samples, n_features)
            n_samples is the number of samples and n_features is the number of predictors
            
        Y: ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and n_targets is the number of response variables.
            
        drawMonitoringChart: bool, optional (default True)
            If True, then the monitoring charts are also drawn.
        
        title: str, optional 
            Title for the charts      
                              
        Returns:
        ---------------------
        abnormalityFlags: ndarray of shape (n_samples,)
            Returns True for abnormal samples and False for normal samples
        """
        
        T2, SPEx, SPEy = self.computeMetrics(X, Y)
        abnormalityFlags = np.logical_or.reduce((T2 > self.T2_CL, SPEx > self.SPEx_CL, SPEy > self.SPEy_CL))
        print('Number of abnormal sample(s): ', np.sum(abnormalityFlags))
        
        if drawMonitoringChart:
            self.draw_monitoring_charts(metrics=(T2, SPEx, SPEy), title=title)
                
        return abnormalityFlags
        
        
        
        