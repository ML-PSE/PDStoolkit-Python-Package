"""
A module for customized PCA class for dynamic process monitoring.
Monitoring methodology is described in our book  'Machine Learning for Process Systems Engineering' (https://mlforpse.com/books/)
============================================================================
"""

# Version: 2023
# Author: Ankur Kumar @ MLforPSE.com
# License: BSD 3 clause

#%% Imports
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

#%% utility functions
def _augument(X, n_lags):
    """ generate augumented matrix with n_lags lagged measurements for each feature
    
    Parameters:
    ---------------------
    X: ndarray of shape (n_samples, n_features)
        n_samples is the number of samples and n_features is the number of features.
    
    n_lags: The number of lags to be used for data augumentation.   
    
    
    Returns:
    ---------------------
    X_augmented: ndarray of shape (n_samples-n_lags, (n_lags+1)*n_features).
        The n_lags+1 values of feature j go into columns (j-1)*(n_lags+1) to j*(n_lags+1)-1.
        The ith row of X_augmented contains data from row i to row i+n_lags from matrix X.
    
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
        
    return X_augmented


#%% define class
class PDS_DPCA(PCA):
    ''' This class wraps Sklearn's PCA class to provide the following additional methods
    
    1) computeMetrics: computes the monitoring indices for the supplied data
    2) computeThresholds: computes the thresholds / control limits for the monitoring indices from training data
    3) draw_monitoring_charts: Draw the monitoring charts for the training or test data
    4) detect_abnormalities: Detects if the observations are abnormal or normal samples
    
    Additional Parameters
    ----------
    n_lags: integer (default 1)
        The number of lags to be used for data augumentation.
        
    
    Additional Attributes
    ----------
    n_lags: integer
        The number of lags used for data augumentation.
        
        
    Original methods of PCA class
    ----------
    Apart from the fit method, the following methods of the original PCA class have been modified to augment the incoming X matrix before executing the original function.
    
    -- fit_transform(X)
    -- score(X)
    -- score_samples(X)
    -- transform(X)

    
    Usage Example
    --------
    >>> import numpy as np
    >>> from PDStoolkit import PDS_DPCA
    >>> from sklearn.preprocessing import StandardScaler
    
    >>> X = np.random.rand(30,5)
    >>> scaler = StandardScaler()
    >>> X_scaled = scaler.fit_transform(X)
    
    >>> dpca = PDS_DPCA(n_lags=2)
    >>> dpca.fit(X_scaled, autoFindNLatents=True)
    
    >>> metrics_train = dpca.computeMetrics(X_scaled, isTrainingData=True)
    >>> T2_CL, SPE_CL = dpca.computeThresholds(method='percentile')
    >>> dpca.draw_monitoring_charts(metrics=metrics_train, title='training data')
    
    >>> X_test = scaler.transform(np.random.rand(30,5))
    >>> abnormalityFlags = dpca.detect_abnormalities(X_test, title='Test data')
    '''
    
    def __init__(self, n_lags=1, **kws):
        
        # sanity check: positive lag
        if n_lags < 1 or (not isinstance(n_lags, int)):
            raise ValueError("Parameters must be positive integers")
            
        super().__init__(**kws)
        self.eig_vals_all = None
        self.n_lags = n_lags
    
    def fit(self, X, autoFindNLatents=False, varianceThreshold=0.9):
        """
        Extends the fit method of Sklearn PCA to allow in-built augmentation of lagged observations and computation of the 'optimal' number of latents.
        
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
        
        # augument data
        X_aug = _augument(X, self.n_lags)
        
        if autoFindNLatents:
            self.n_components = X_aug.shape[1] 
            PCA.fit(self, X_aug)
            self.eig_vals_all = self.explained_variance_ # to be used later for threshold calculations 
            
            # decide # of PCs to retain
            explained_variance = 100*self.explained_variance_ratio_ # in percentage
            cum_explained_variance = np.cumsum(explained_variance) # cumulative % variance explained
            selected_Nlatent = np.argmax(cum_explained_variance >= varianceThreshold*100) + 1
            
            # print message for selected_Nlatent
            print('# of latents selected: ', selected_Nlatent)
            
            # plot cumulative % variance curve
            plt.figure()
            plt.plot(range(1,X_aug.shape[1]+1), cum_explained_variance, 'r+', label = 'cumulative % variance explained')
            plt.plot(range(1,X_aug.shape[1]+1), explained_variance, 'b+' , label = '% variance explained by each PC')
            plt.ylabel('Explained variance (in %)'), plt.xlabel('Principal component number')
            plt.title('Variance explained vs # of PCs'), plt.legend()
            plt.show()
            
            # store the computed optimal_Nlatents
            self.n_components = selected_Nlatent
        
        # call the original fit method of PCA class to fit with n_components components
        PCA.fit(self, X_aug)
        
        # ensure eig_vals_all attribute is assigned
        if not autoFindNLatents:
            if self.n_components == X_aug.shape[1]:
                self.eig_vals_all = self.explained_variance_
            else:
                PCA_temp = PCA().fit(X_aug)
                self.eig_vals_all = PCA_temp.explained_variance_
            
        return self
        
    def transform(self, X):
        """
        Apply augmentation to X and then perform dimensionality reduction.
        Augmented X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Data, where n_samples is the number of samples and n_features is the number of features.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples-n_lags, n_components)
            Transformed values.
        """
        
        # augument data and call original function
        X_aug = _augument(X, self.n_lags)
        return PCA.transform(self, X_aug)
    
    def fit_transform(self, X):
        """
        Augment X, fit the model with augmented X, and apply the dimensionality reduction on augmented X.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples-n_lags, n_components)
            Transformed values.
        """
        
        # augument data and call original function
        X_aug = _augument(X, self.n_lags)
        return PCA.fit_transform(self, X_aug)
    
    def score(self, X):
        """
        Apply augmentation to X and then compute Score. Returns the average log-likelihood of all samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data.

        Returns
        -------
        ll : float
            Average log-likelihood of the samples under the current model.
        """
                
        # augument data and call original function
        X_aug = _augument(X, self.n_lags)
        return PCA.score(self, X_aug)
    
    def score_samples(self, X):
        """
        Apply augmentation to X and then compute score for each augmented sample. Returns the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.

        Returns
        -------
        ll : ndarray of shape (n_samples,)
            The first n_lags entries are nan.
            Log-likelihood of each augmented sample under the current model.
        """
        
        # augument data and call original function
        X_aug = _augument(X, self.n_lags)
        return np.append(np.repeat(np.nan, self.n_lags), PCA.score_samples(self, X_aug))
    
    def inverse_transform(self, X):
        """
        Transform data back to its original augmented space.
        
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_components)
            DPCS score data, where n_samples is the number of samples and n_components is the number of components retained in the DPCA nodel.

        Returns
        -------
        X_augmented_reconstructed : ndarray of shape (n_samples, n_features)
            Reconstructed augmented values.
        """
               
        return PCA.inverse_transform(self, X)
                  
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
        SPE: ndarray of shape (n_samples, ). Also called Q metrics.
            The first n_lags entries are nan.
                        
        T2: ndarray of shape (n_samples, ). Also called Hotelling's T-square.
            The first n_lags entries are nan.
        """
        
        # augument data
        X_aug = _augument(X, self.n_lags)
        
        # compute SPE
        X_aug_scores = self.transform(X)
        X_aug_reconstruct = self.inverse_transform(X_aug_scores) 
        X_aug_error = X_aug - X_aug_reconstruct
        SPE = np.append(np.repeat(np.nan, self.n_lags), np.sum(X_aug_error*X_aug_error, axis = 1))
        
        # compute T2
        if isTrainingData:
            T_cov = np.cov(X_aug_scores.T) # equivalent to np.diag(pca.explained_variance_)
            T_cov_inv = np.linalg.inv(T_cov)
            self.T_cov_inv = T_cov_inv
            
        T2 = np.zeros((X_aug.shape[0],))
        for i in range(X_aug.shape[0]):
            T2[i] = np.dot(np.dot(X_aug_scores[i,:],self.T_cov_inv),X_aug_scores[i,:].T)
        T2 = np.append(np.repeat(np.nan, self.n_lags), T2)
            
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
            T2_CL = np.nanpercentile(self.T2_train, percentile)
            SPE_CL = np.nanpercentile(self.SPE_train, percentile)
                       
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
    
    def draw_monitoring_charts(self, metrics=None, logScaleY=False, title=''):
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
        if logScaleY:
            plt.yscale('log')
        plt.title(title), plt.show()
        
        # SPE
        plt.figure()
        plt.plot(metrics[1], color='k', linestyle = ':', marker='o', markerfacecolor = 'C4')
        if hasattr(self, 'SPE_CL'):
            plt.axhline(self.SPE_CL, color = "red", linestyle = "-.")
        plt.xlabel('Sample #'), plt.ylabel('SPE')
        if logScaleY:
            plt.yscale('log')
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
        abnormalityFlags: ndarray of shape (n_samples, )
            Returns 1 for abnormal samples and 0 for normal samples.
            The first n_lags entries are nan.
        """
        
        T2, SPE = self.computeMetrics(X)
        abnormalityFlags = np.logical_or(T2 > self.T2_CL, SPE > self.SPE_CL)
        print('Number of abnormal sample(s): ', np.sum(abnormalityFlags))
        
        abnormalityFlags = np.append(np.repeat(np.nan, self.n_lags), abnormalityFlags)        
        if drawMonitoringChart:
            self.draw_monitoring_charts(metrics=(T2, SPE), title=title)
                
        return abnormalityFlags
        
        
        
        