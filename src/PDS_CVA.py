"""
A module for CVA class for dynamic process monitoring. 
Monitoring methodology is described is our book  'Machine Learning in Python for Dynamic Process Systems' (https://mlforpse.com/books/)
=======================================================
"""

# Version: 2023
# Author: Ankur Kumar @ MLforPSE.com
# License: BSD 3 clause

#%% Imports
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy
import sys

#%% utility functions

def _get_past_future_vectors(U, Y, l, getBothMatrices=True):
    """ generate matrix with past (p) and future (f) vectors
    
    Parameters:
    ---------------------
    U: ndarray of shape (n_samples, n_features)
        Input/Predictor data, where n_samples is the number of samples and n_features is the number of predictors
        
    Y: ndarray of shape (n_samples, n_targets)
        Output/Target data, where n_samples is the number of samples and n_targets is the number of response variables.
    
    l: integer. The lag order.
    
    getBothMatrices: bool
        If True, then then both pMatrix and fMatrix are computed else only pMatrix is computed
        
    
    Returns:
    ---------------------
    pMatrix: if getBothMatrices is True then ndarray of shape (n_samples-2*l, l*(n_features+n_targets)) 
             if getBothMatrices is False then ndarray of shape (n_samples-l+1, l*(n_features+n_targets)) 
             
    fMatrix: ndarray of shape (n_samples-2*l, (l+1)*n_targets). Computed only when getBothMatrices is True.
    
    """
    
    # sanity check on the number of samples
    
    if getBothMatrices:
        if (U.shape[0] < 2*l+1) or (Y.shape[0] < 2*l+1):
            raise ValueError("Data matrices must have atleast 2*n_lags+1 samples")     
    else:
        if U.shape[0] < l:
            raise ValueError("Input data matrix must have atleast n_lags samples")
    
    N = U.shape[0]
    m = U.shape[1]
    r = Y.shape[1]
    
    if getBothMatrices:
        pMatrix = np.zeros((N-2*l, l*(m+r)))
        fMatrix = np.zeros((N-2*l, (l+1)*r))
        
        for i in range(l,N-l):
            pMatrix[i-l,:] = np.hstack((Y[i-l:i,:].flatten(), U[i-l:i,:].flatten()))
            fMatrix[i-l,:] = Y[i:i+l+1,:].flatten()
    
    else:
        pMatrix = np.zeros((N-l+1, l*(m+r)))
        fMatrix = None
        
        for i in range(l,N+1):
            pMatrix[i-l,:] = np.hstack((Y[i-l:i,:].flatten(), U[i-l:i,:].flatten()))
        
    return pMatrix, fMatrix

#%% define class
class PDS_CVA():
    ''' This class provides the following methods
    
    1) computeMetrics: computes the monitoring indices (Ts2, Te2, Q) for the supplied data
    2) computeThresholds: computes the thresholds/control limits for the monitoring indices from training data
    3) draw_monitoring_charts: Draw the monitoring charts for the training or test data
    4) detect_abnormalities: Detects if the observations are abnormal or normal samples
    
    Parameters
    ----------
    n_lags: integer (default 1)
        The number of lags to be used for data augumentation.
    
    n_components: integer (default 2)
        Number of components to keep.
        
    
    Attributes
    ----------
    n_lags: integer. The number of lags used for data augumentation.
    
    n_components: integer. The number of components.
    
    
    Usage Example
    --------
    >>> import numpy as np
    >>> from PDStoolkit import PDS_CVA
    
    >>> U = np.random.rand(30,5)
    >>> Y = np.random.rand(30,3)
    
    >>> cva = PDS_CVA(n_lags=3, n_components=2)
    >>> cva.fit(U, Y)
    
    >>> metrics_train = cva.computeMetrics(U, Y, isTrainingData=True)
    >>> Ts2_CL, Te2_CL, Q_CL = cva.computeThresholds()
    >>> cva.draw_monitoring_charts(metrics=metrics_train, title='training data')
    
    >>> U_test = np.random.rand(30,5)
    >>> Y_test = np.random.rand(30,3) 
    >>> abnormalityFlags = cva.detect_abnormalities(U_test, Y_test, title='Test data')
    '''
    
    def __init__(self, n_lags=1, n_components=2):
        # sanity check (positive lag and order)
        if n_lags < 1 or n_components < 1 or isinstance(n_lags, float) or isinstance(n_components, float):
            raise ValueError("Parameters must be positive integers")
        
        self.n_lags = n_lags
        self.n_components = n_components
        self.pScaler = StandardScaler(with_std=False)
        self.fScaler = StandardScaler(with_std=False)
        
    
    def fit(self, U, Y, autoFindNcomponents=False, ratioThreshold=0.1, HankelPlot=True):
        """
        Computes the transformation matrices. Optionally finds the 'optimal' number of states / model order.
        
        Parameters:
        ---------------------
        U: ndarray of shape (n_samples, n_features)
            Training input data, where n_samples is the number of samples and n_features is the number of predictors
            
        Y: ndarray of shape (n_samples, n_targets)
            Training output data, where n_samples is the number of samples and n_targets is the number of response variables.
                
        autoFindNcomponents: bool, optional (default False)
            Decides whether to compute the 'optimal' model order. 
            If True, the Hankel singular values are used. Model order, n, is chosen such that the normalized values of the (n+1)th onwards singular
                     values are below ratioThreshold.
            
        ratioThreshold: float, optional (default 0.1)
            Real value between 0 and 1.
            Used for determination of model order using normalized Hankel singular values.
            
        Returns
        ---------------------
        self: object
            fitted model
        """
        
        # sanity check (positive lag and order)
        if U.ndim != 2 or Y.ndim != 2:
            raise ValueError("U and Y matrices must be 2 dimensional arrays")
        
        # generate past (p) and future (f) vectors for training dataset
        pMatrix_train, fMatrix_train = _get_past_future_vectors(U, Y, self.n_lags, getBothMatrices=True)
        
        # center data
        pMatrix_train_centered = self.pScaler.fit_transform(pMatrix_train)
        fMatrix_train_centered = self.fScaler.fit_transform(fMatrix_train)
        
        # perform SVD
        sigma_pp = np.cov(pMatrix_train_centered, rowvar=False)
        sigma_ff = np.cov(fMatrix_train_centered, rowvar=False)
        sigma_pf = np.cov(pMatrix_train_centered, fMatrix_train_centered, rowvar=False)[:len(sigma_pp),len(sigma_pp):]
        
        matrixProduct = np.dot(np.dot(np.linalg.inv(scipy.linalg.sqrtm(sigma_pp).real), sigma_pf), np.linalg.inv(scipy.linalg.sqrtm(sigma_ff).real))
        U, S, V = np.linalg.svd(matrixProduct)
        J = np.dot(np.transpose(U), np.linalg.inv(scipy.linalg.sqrtm(sigma_pp).real))
        self.SingularValues = S
        
        # plot Hankel singular values
        if HankelPlot:
            plt.figure()
            plt.plot(np.arange(1,len(S)+1), S, '*')
            plt.xlabel('Order # '), plt.ylabel('Singular value')
            plt.xlim(1), plt.show()
            
        # find optimal model order via Singular values if specified
        if autoFindNcomponents:
            self.n_components = np.where(S/S[0] > ratioThreshold)[0][-1] + 1    
            # print message with selected model order
            print('Optimal model order selected is: ', self.n_components)
        
        # get the reduced order matrices
        Jn = J[:self.n_components,:]
        Je = J[self.n_components:,:]
        self.Jn = Jn
        self.Je = Je
        
        return self
        
    def computeMetrics(self, U, Y, isTrainingData=False):
        """
        computes the monitoring indices for the supplied data.
        
        Parameters:
        ---------------------
        U: ndarray of shape (n_samples, n_features)
            Input/Predictor vectors, where n_samples is the number of samples and n_features is the number of predictors
            
        Y: ndarray of shape (n_samples, n_targets)
            Output/Target vectors, where n_samples is the number of samples and n_targets is the number of response variables.
            
        isTrainingData: bool, optional (default False)
            If True, then the computed metrics are stored as object properties which are utilized during computation of metric thresholds
                          
        Returns:
        ---------------------
        Ts2: ndarray of shape (n_samples, ).
            The first n_lags-1 entries are nan.
        
        Te2: ndarray of shape (n_samples, ).
            The first n_lags-1 entries are nan.
        
        Q: ndarray of shape (n_samples, ).
            The first n_lags-1 entries are nan.
        """
        
        if not hasattr(self, 'Jn'):
            raise AttributeError('Transformation matrix not found. Run fit method before computing the metrics.')
        
        # generate past (p) and future (f) vector Matrices and center them
        pMatrix, _ = _get_past_future_vectors(U, Y, self.n_lags, getBothMatrices=False)
        pMatrix_centered = self.pScaler.transform(pMatrix)
        
        # Ts2
        Xn = np.dot(self.Jn, pMatrix_centered.T)
        Ts2 = np.append(np.repeat(np.nan, self.n_lags-1), np.array([np.dot(Xn[:,i], Xn[:,i]) for i in range(pMatrix_centered.shape[0])]))
        
        # Te2
        Xe = np.dot(self.Je, pMatrix_centered.T)
        Te2 = np.append(np.repeat(np.nan, self.n_lags-1), np.array([np.dot(Xe[:,i], Xe[:,i]) for i in range(pMatrix_centered.shape[0])]))
        
        # Q
        r = pMatrix_centered.T - np.dot(self.Jn.T, Xn)
        Q = np.append(np.repeat(np.nan, self.n_lags-1), np.array([np.dot(r[:,i], r[:,i]) for i in range(pMatrix_centered.shape[0])]))
        
        # save metrics for training data as attributes
        if isTrainingData:
            self.Ts2_train = Ts2
            self.Te2_train = Te2
            self.Q_train = Q
        
        return Ts2, Te2, Q
    
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
        Ts2_CL: float
            Control limit/threshold for Ts2 metric
        
        Te2_CL: float
            Control limit/threshold for Te2 metric
        
        Q_CL: float
            Control limit/threshold for Q metric
        """
        
        if not hasattr(self, 'Ts2_train'):
            raise AttributeError('Training metrices not found. Run computeMetrics method before computing the thresholds.')
            
        if method == 'percentile':
            Ts2_CL = np.nanpercentile(self.Ts2_train, percentile)
            Te2_CL = np.nanpercentile(self.Te2_train, percentile)
            Q_CL = np.nanpercentile(self.Q_train, percentile)
            
        elif method == 'statistical':
            N = self.T2_train.shape[0]
            k = self.n_components
            
            # Ts2_CL
            Ts2_CL = k*(N**2-1)*scipy.stats.f.ppf(1-alpha,k,N-k)/(N*(N-k))
            
            # Te2_CL
            z = self.Je.shape[0]
            Te2_CL = z*(N**2-1)*scipy.stats.f.ppf(1-alpha,z,N-z)/(N*(N-z))

            # Q_CL
            mean_Q_train = np.nanmean(self.Q_train)
            var_Q_train = np.nanvar(self.Q_train)
            
            if mean_Q_train == 0:
                mean_Q_train = sys.float_info.epsilon
            if var_Q_train == 0:
                var_Q_train = sys.float_info.epsilon

            g = var_Q_train/(2*mean_Q_train)
            h = 2*mean_Q_train**2/var_Q_train
            Q_CL = g*scipy.stats.chi2.ppf(1-alpha, h)
        
        else:
            raise ValueError('Incorrect choice of method parameter')
            
        # save control limits as attributes
        self.Ts2_CL = Ts2_CL
        self.Te2_CL = Te2_CL
        self.Q_CL = Q_CL
        
        return Ts2_CL, Te2_CL, Q_CL
    
    def draw_monitoring_charts(self, metrics=None, logScaleY=False, title=''):
        """
        Draw the monitoring charts for the training or test data.
        The control limits are plotted as red dashed line. 
        
        Parameters:
        ---------------------
        metrics: (optional) list or tuple of monitoring metrics (1D numpy arrays). Should follow the order (Ts2, Te2, Q).
            If not specified, then the object's stored metrics from training data are used.
            
        title: str, optional 
            Title for the charts                      
        """
        
        if metrics is None:
            metrics = (self.Ts2_train, self.Te2_train, self.Q_train)
        
        # Ts2
        plt.figure()
        plt.plot(metrics[0], color='k', linestyle = ':', marker='o', markerfacecolor = 'C4')
        if hasattr(self, 'Ts2_CL'):
            plt.axhline(self.Ts2_CL, color = "red", linestyle = "-.")
        plt.xlabel('Sample #'), plt.ylabel('Ts2')
        if logScaleY:
            plt.yscale('log')
        plt.title(title), plt.show()
                
        # Te2
        plt.figure()
        plt.plot(metrics[1], color='k', linestyle = ':', marker='o', markerfacecolor = 'C4')
        if hasattr(self, 'Te2_CL'):
            plt.axhline(self.Te2_CL, color = "red", linestyle = "-.")
        plt.xlabel('Sample #'), plt.ylabel('Te2')
        if logScaleY:
            plt.yscale('log')
        plt.title(title), plt.show()
        
        # Q
        plt.figure()
        plt.plot(metrics[2], color='k', linestyle = ':', marker='o', markerfacecolor = 'C4')
        if hasattr(self, 'Q_CL'):
            plt.axhline(self.Q_CL, color = "red", linestyle = "-.")
        plt.xlabel('Sample #'), plt.ylabel('Q')
        if logScaleY:
            plt.yscale('log')
        plt.title(title), plt.show()
    
    def fit_4_monitoring(self, U, Y, autoFindNcomponents=False, ratioThreshold=0.1, HankelPlot=False, method='percentile', percentile=99, alpha=0.01):
        """
        A utility method that calls the following model-training related methods in succession.
        
        fit(U, Y, autoFindNcomponents, ratioThreshold, HankelPlot)
        computeMetrics(U, Y, isTrainingData=True)
        computeThresholds(method, percentile, alpha)
        draw_monitoring_charts(title='training data')  
        
        Note: Check doc string of individual methods for details on the method parameters.
        """
                
        self.fit(U, Y, autoFindNcomponents, ratioThreshold, HankelPlot)
        self.computeMetrics(U, Y, isTrainingData=True)
        self.computeThresholds(method, percentile, alpha)
        self.draw_monitoring_charts(title='training data')
        
        return self
    
    def detect_abnormalities(self, U, Y, drawMonitoringChart=True, title=''):
        """
        Detects if the observations are abnormal or normal. 
        Detection is based on the logic that for a 'normal' sample, the monitoring metrics should lie below their thresholds.
        
        Parameters:
        ---------------------
        U: ndarray of shape (n_samples, n_features)
            Inputs/Predictor vectors, where n_samples is the number of samples and n_features is the number of predictors
            
        Y: ndarray of shape (n_samples,) or (n_samples, n_targets)
            Output/Target vectors, where n_samples is the number of samples and n_targets is the number of response variables.
            
        drawMonitoringChart: bool, optional (default True)
            If True, then the monitoring charts are also drawn.
        
        title: str, optional 
            Title for the charts      
                              
        Returns:
        ---------------------
        abnormalityFlags: ndarray of shape (n_samples,)
            Returns True for abnormal samples and False for normal samples
        """
        
        Ts2, Te2, Q = self.computeMetrics(U, Y)
        abnormalityFlags = np.logical_or.reduce((Ts2 > self.Ts2_CL, Te2 > self.Te2_CL, Q > self.Q_CL))
        print('Number of abnormal sample(s): ', np.sum(abnormalityFlags))
        
        if drawMonitoringChart:
            self.draw_monitoring_charts(metrics=(Ts2, Te2, Q), title=title)
                
        return abnormalityFlags
        
        
        
        