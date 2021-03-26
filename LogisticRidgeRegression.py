import numpy as np
from kernels import LinearKernel
from ridge_utils import solveKLRR

class KernelLogisticRidge():
    
    
    def __init__(self, kernel=LinearKernel(), l=0.01):
        '''
        Kernel Logistic Ridge Regression class
        
        Parameters
        ----------
        kernel : Kernel Class, optional
            This is the Kernel class used for the problem.
            The default is LinearKernel().
        l : float, optional
            Ridge penalty. The default is 0.01.
        '''
        self.l = l
        self.alpha = None
        self.kernel = kernel
        
    def fit(self, X=None, y=None, K=None):
        '''
        Learning the Kernel Logistic Ridge Regression weights

        Either X or K should be given
        Parameters
        ----------
        X : array_like (n, m), optional
            Features Data 
        y : array_like (n)
            Label data
        K : arrau_like (n, n), optional
            Precomputed training data kernel. The default is None.
        '''
        assert (X is not None) or (K is not None), 'Give either X or K should'
        assert y is not None, 'Please give the labels'
        
        if X is not None:
            self.X_train = X.copy()
        else:
            self.X_train = None
            
        if K is None:
            K = self.kernel.gram(X)
            
        self.alpha = solveKLRR(y, K, self.l)
        
    def project(self, X, pairwise_K = None):
        '''
        Predict y of X

        Parameters
        ----------
        X : array_like (n_test, m)
            Predicted data.
        pairwise_K : array_like (n_test, n), optional
            Pairwise kernel K(i, j) = K(X(i), X_train(j)).

        Returns
        -------
        y_predict : array_like (n_test)
            Predicted labels

        '''
        assert (self.X_train is not None) or (pairwise_K is not None), 'Give\
            the pairwise kernel'
        
        if self.alpha is None:
            raise Exception('Model has not been trained yet.', 'alpha not learned')
            
        y_predict = np.zeros(len(X))
        if pairwise_K is None:
            pairwise_K = self.kernel.pairwise_kernel(X, self.X_train)
        else:
            assert pairwise_K.shape[0] == X.shape[0], 'Pairwise K should be \
            computed as pairwise_K(i, j) = K(X[i], X_train[j] )'
        y_predict = np.sum(pairwise_K * self.alpha, axis=1)
        return y_predict
            
    def predict(self, X, pairwise_K = None):
        '''
        Predicting the labels of X

        Parameters
        ----------
        X : array_like (n_test, m)
            Predicted data.
        pairwise_K : array_like (n_test, n), optional
            Pairwise kernel K(i, j) = K(X(i), X_train(j)).

        Returns
        -------
        y_predict : array_like (n_test)
            Predicted labels

        '''
        return np.sign(self.project(X, pairwise_K = pairwise_K))