"""
A module for customized PLS class for dynamic process monitoring. 
Monitoring methodology is described in our book  'Machine Learning for Process Systems Engineering' (https://mlforpse.com/books/)
=======================================================
"""

# Version: 2023
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

#%% utility functions
def _augument(X, n_lags, Y=None):
    """ Generate augumented X matrix with n_lags lagged measurements for each input feature. The first n_lags rows of Y matrix are accordingly removed.
    
    Parameters:
    ---------------------
    X: ndarray of shape (n_samples, n_features)
        Predictor data, where n_samples is the number of samples and n_features is the number of features.
    
    Y: ndarray of shape (n_samples, n_targets), default=None
        Target data, where n_samples is the number of samples and n_targets is the number of response variables.
    
    n_lags: The number of lags to be used for data augumentation.   
    
    
    Returns:
    ---------------------
    X_augmented: ndarray of shape (n_samples-n_lags, (n_lags+1)*n_features).
        The n_lags+1 values of feature j go into columns (j-1)*(n_lags+1) to j*(n_lags+1)-1.
        The ith row of X_augmented contains data from row i to row i+n_lags from matrix X.
    
    Y_augmented: ndarray of shape (n_samples-n_lags, n_targets)
    
    """
    
    # sanity check (data matrix must have atleast n_lags+1 samples)
    if X.shape[0] < n_lags+1:
        raise ValueError("Data matrix must have atleast n_lags+1 samples")
    
    # augment training data
    N, m = X.shape
    X_augmented = np.zeros((N-n_lags, (n_lags+1)*m))
    
    for sample in range(n_lags, N):
        XBlock = X[sample-n_lags:sample+1,:]
        X_augmented[sample-n_lags,:] = np.reshape(XBlock, (1,-1), order = 'F')
        
    if Y is not None:
        return X_augmented, Y[n_lags:, :]
    else:
        return X_augmented, None

#%% define class
class PDS_DPLS(PLSRegression):
    ''' This class wraps Sklearn's PLSRegression class to provide the following addiitonal methods
    
    1) computeMetrics: computes the monitoring indices for the supplied data
    2) computeThresholds: computes the thresholds / control limits for the monitoring indices from training data
    3) draw_monitoring_charts: Draw the monitoring charts for the training or test data
    4) detect_abnormalities: Detects if the observations are abnormal or normal samples
    
    Additional Parameters
    ----------
    n_lags: integer (default 1)
        The number of lags to be used for data augumentation. Currently, lagged values of only input variables are used resulting in FIR type model.
    
 
    Additional Attributes
    ----------
    n_lags: integer
        The number of lags used for data augumentation.
        
        
    Original methods of PLS class
    ----------
    Apart from the fit method, the following methods of the original PLS class have been modified to augment the incoming X matrix before executing the original function.
    
    -- fit_transform(X, Y)
    -- score(X Y)
    -- inverse_transform(X)
    -- transform(X, Y)
    -- predict(X)
    
    
    Usage Example
    --------
    >>> import numpy as np
    >>> from PDStoolkit import PDS_DPLS
    
    >>> X = np.random.rand(30,5)
    >>> Y = np.random.rand(30,2)
    
    >>> dpls = PDS_DPLS(n_lags=2)
    >>> dpls.fit(X, Y, autoFindNLatents=True, ratioThreshold=0.99)
    
    >>> metrics_train = dpls.computeMetrics(X, Y, isTrainingData=True)
    >>> T2_CL, SPEx_CL, SPEy_CL = dpls.computeThresholds(method='percentile', percentile=99)
    >>> dpls.draw_monitoring_charts(metrics=metrics_train, title='training data')
    
    >>> X_test = np.random.rand(30,5)
    >>> Y_test = np.random.rand(30,2) 
    >>> abnormalityFlags = dpls.detect_abnormalities(X_test, Y_test, title='Test data')
    '''
    
    def __init__(self, n_lags=1, **kws):
        
        # sanity checks: positive lag
        if n_lags < 1 or (not isinstance(n_lags, int)):
            raise ValueError("Parameters must be positive integers")
               
        super().__init__(**kws)
        self.n_lags = n_lags
    
    def fit(self, X, Y, autoFindNLatents=False, n_CVsplits=10, ratioThreshold=0.0):
        """
        Extends the fit method of Sklearn PLS to allow in-built augmentation of lagged input observations and computation of the 'optimal' number of latents.
        
        Parameters:
        ---------------------
        X: ndarray of shape (n_samples, n_features)
            Training predictor data, where n_samples is the number of samples and n_features is the number of predictors
            
        Y: ndarray of shape (n_samples,) or (n_samples, n_targets)
            Training target data, where n_samples is the number of samples and n_targets is the number of response variables.
                
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
        
        # sanity check on X and Y
        X = self._validate_data(X, dtype=np.float64, copy=self.copy)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of observations must be same for both X and Y")
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
            
        # augument data
        X_aug, Y_aug = _augument(X, self.n_lags, Y)
        
        if autoFindNLatents:
            # compute fitting and validation errors for different number of latents
            scaler = StandardScaler()
            fit_MSE = []
            validate_MSE = []
            lowest_validation_mse = np.inf
            selected_Nlatent = X_aug.shape[1] # placehoder
            
            for Nlatent in range(1,X_aug.shape[1]+1):
                print('Performing cross-validation with {} latents'.format(Nlatent))
                
                local_fit_MSEs = []
                local_validate_MSEs = []
                
                kfold = KFold(n_splits = n_CVsplits, shuffle = True, random_state = 100)
                for fit_index, validate_index in kfold.split(Y_aug):
                    X_aug_fit_normal = scaler.fit_transform(X_aug[fit_index])
                    X_aug_validate_normal = scaler.transform(X_aug[validate_index])
                    
                    Y_aug_fit_normal = scaler.fit_transform(Y_aug[fit_index])
                    Y_aug_validate_normal = scaler.transform(Y_aug[validate_index])
                    
                    pls = PLSRegression(n_components = Nlatent)
                    pls.fit(X_aug_fit_normal, Y_aug_fit_normal)
                    
                    local_fit_MSEs.append(mean_squared_error(Y_aug_fit_normal, pls.predict(X_aug_fit_normal)))
                    local_validate_MSEs.append(mean_squared_error(Y_aug_validate_normal, 
                                                                    pls.predict(X_aug_validate_normal)))
                
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
        PLSRegression.fit(self, X_aug, Y_aug)
        return self
        
    def transform(self, X, Y=None):
        """
        Apply augmentation to X and Y and then perform dimensionality reduction.
        
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Input/Predictor data, where n_samples is the number of samples and n_features is the number of features.
        
        Y: ndarray of shape (n_samples, n_targets), default=None
            Output/Response data, where n_samples is the number of samples and n_features is the number of response variables.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples-n_lags, n_components)
            Transformed values or x_scores.
        
        Y_transformed : ndarray of shape (n_samples-n_lags, n_components). Returned if Y is not None.
            Transformed values or y_scores.
        """
        
        # augument data and call original function
        X_aug, Y_aug = _augument(X, self.n_lags, Y)
        return PLSRegression.transform(self, X_aug, Y_aug)
    
    def fit_transform(self, X, Y):
        """
        Augment X, fit the model with augmented X and Y, and apply the dimensionality reduction.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training predictor data, where n_samples is the number of samples and n_features is the number of predictors
            
        Y: ndarray of shape (n_samples,) or (n_samples, n_targets)
            Training target data, where n_samples is the number of samples and n_targets is the number of response variables.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples-n_lags, n_components)
            Transformed values  or x_scores.
        
        Y_transformed : ndarray of shape (n_samples-n_lags, n_components).
            Transformed values or y_scores.
        """
        
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # augument data and call original function
        X_aug, Y_aug = _augument(X, self.n_lags, Y)
        return PLSRegression.fit_transform(self, X_aug, Y_aug)
    
    def score(self, X, Y):
        """
        Apply augmentation to X (and Y) and then compute Score. Returns the coefficient of determination of the prediction.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data.
        
        Y:  ndarray of shape (n_samples,) or (n_samples, n_targets)
            True values for response variables.

        Returns
        -------
        score : float
            R-squared value of self.predict(X) w.r.t. Y.
        """
        
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
            
        # augument data and call original function
        X_aug, Y_aug = _augument(X, self.n_lags, Y)
        return PLSRegression.score(self, X_aug, Y_aug)
    
    def inverse_transform(self, X):
        """
        Transform data back to its original augmented space.
        
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_components)
            DPLS X score data, where n_samples is the number of samples and n_components is the number of components retained in the DPLS nodel.


        Returns
        -------
        X_augmented_reconstructed : ndarray of shape (n_samples, n_features)
            Reconstructed augmented X matrix.

        """
               
        return PLSRegression.inverse_transform(self, X)
    
    def predict(self, X):
        """
        Predict targets of given samples.
        
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Predictor data, where n_samples is the number of samples and n_features is the number of predictors.


        Returns
        -------
        Y_pred : ndarray of shape (n_samples, n_targets)
            Predicted values. Entries in the first n_lags rows are nan.
        """
        
        # augument data and call original function
        X_aug, _ = _augument(X, self.n_lags)
        Y_pred = PLSRegression.predict(self, X_aug)
        
        return np.append(np.full((self.n_lags, Y_pred.shape[1]), np.nan), Y_pred, axis=0) 
    
    def computeMetrics(self, X, Y, isTrainingData=False):
        """
        computes the monitoring indices for the supplied data.
        
        Parameters:
        ---------------------
        X: ndarray of shape (n_samples, n_features)
            Input data, n_samples is the number of samples and n_features is the number of predictors
            
        Y: ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target or response data, where n_samples is the number of samples and n_targets is the number of response variables.
            
        isTrainingData: bool, optional (default False)
            If True, then the computed metrics are stored as object properties which are utilized during computation of metric thresholds
                          
        Returns:
        ---------------------
        SPEx: ndarray of shape (n_samples, ).
            The first n_lags entries are nan.   
        
        SPEy: ndarray of shape (n_samples, ).
            The first n_lags entries are nan.
        
        T2: ndarray of shape (n_samples, ). Also called Hotelling's T-square.
            The first n_lags entries are nan.
        """
        
        # augument data
        X_aug, Y_aug = _augument(X, self.n_lags, Y)
        
        # compute SPEx
        X_aug_scores = self.transform(X)
        X_aug_reconstruct = self.inverse_transform(X_aug_scores) 
        X_aug_error = X_aug - X_aug_reconstruct
        X_aug_error_scaled = X_aug_error/self._x_std
        SPEx = np.append(np.repeat(np.nan, self.n_lags), np.sum(X_aug_error_scaled*X_aug_error_scaled, axis = 1))
        
        # compute SPEy
        Y_aug_pred = self.predict(X) # Entries in the first n_lags rows are nan.
        Y_aug_pred = Y_aug_pred[self.n_lags:,:] # removing the first n_lags rows
        Y_aug_error = Y_aug - Y_aug_pred
        Y_aug_error_scaled = Y_aug_error/self._y_std
        SPEy = np.append(np.repeat(np.nan, self.n_lags), np.sum(Y_aug_error_scaled*Y_aug_error_scaled, axis = 1))
        
        # compute T2
        if isTrainingData:
            T_cov = np.cov(X_aug_scores.T)
            if T_cov.ndim == 0:
                T_cov_inv = 1/T_cov
            else:
                T_cov_inv = np.linalg.inv(T_cov)
            self.T_cov_inv = T_cov_inv
            
        T2 = np.zeros((X_aug.shape[0],))
        for i in range(X_aug.shape[0]):
            T2[i] = np.dot(np.dot(X_aug_scores[i,:],self.T_cov_inv),X_aug_scores[i,:].T)
        T2 = np.append(np.repeat(np.nan, self.n_lags), T2)
        
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
            T2_CL = np.nanpercentile(self.T2_train, percentile)
            SPEx_CL = np.nanpercentile(self.SPEx_train, percentile)
            SPEy_CL = np.nanpercentile(self.SPEy_train, percentile)
            
        elif method == 'statistical':
            N = self.T2_train.shape[0]
            k = self.n_components
            
            # T2_CL
            T2_CL = k*(N**2-1)*scipy.stats.f.ppf(1-alpha,k,N-k)/(N*(N-k))

            # SPEx_CL
            mean_SPEx_train = np.nanmean(self.SPEx_train)
            var_SPEx_train = np.nanvar(self.SPEx_train)
            
            if mean_SPEx_train == 0:
                mean_SPEx_train = sys.float_info.epsilon
            if var_SPEx_train == 0:
                var_SPEx_train = sys.float_info.epsilon

            g = var_SPEx_train/(2*mean_SPEx_train)
            h = 2*mean_SPEx_train**2/var_SPEx_train
            SPEx_CL = g*scipy.stats.chi2.ppf(1-alpha, h)

            # SPEy_CL
            mean_SPEy_train = np.nanmean(self.SPEy_train)
            var_SPEy_train = np.nanvar(self.SPEy_train)
            
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
    
    def draw_monitoring_charts(self, metrics=None, logScaleY=False, title=''):
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
        if logScaleY:
            plt.yscale('log')
        plt.title(title), plt.show()
        
        # SPEx
        plt.figure()
        plt.plot(metrics[1], color='k', linestyle = ':', marker='o', markerfacecolor = 'C4')
        if hasattr(self, 'SPEx_CL'):
            plt.axhline(self.SPEx_CL, color = "red", linestyle = "-.")
        plt.xlabel('Sample #'), plt.ylabel('SPEx')
        if logScaleY:
            plt.yscale('log')
        plt.title(title), plt.show()
        
        # SPEy
        plt.figure()
        plt.plot(metrics[2], color='k', linestyle = ':', marker='o', markerfacecolor = 'C4')
        if hasattr(self, 'SPEy_CL'):
            plt.axhline(self.SPEy_CL, color = "red", linestyle = "-.")
        plt.xlabel('Sample #'), plt.ylabel('SPEy')
        if logScaleY:
            plt.yscale('log')
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
        
        
        
        