import numpy as np
import cvxopt.solvers
from kernels import LinearKernel
import time
from sklearn.metrics import accuracy_score
from cvxopt import spmatrix, sparse, matrix


cvxopt.solvers.options['show_progress'] = True

# Class for Soft Margin SVM with kernel function
# Implementation with cvxopt
class SVM(object):
    
    def __init__(self, kernel=LinearKernel, C=None, intercept = False, **kernel_params):
        """
        Parameters
        ----------
        kernel : callable, optional
            DESCRIPTION. The default is linear_kernel.
        C : float, optional
            Parameter for the soft margin control 
            (0 or None for hard SVM) . The default is None.

        Returns
        -------
        None.

        """
        if isinstance(kernel, type):
            #uninitialized kernel
            self.kernel = kernel(**kernel_params)
        else:
            self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)
        self.intercept = intercept
    
    def fit(self, X=None, y=None, K=None):
        """

        Parameters
        ----------
        X : array-like of dim (n,k), n = number of samles, k = number of features
            Data
        y : array-like (n,)
            Labels
        """
        
        assert (X is not None) or (K is not None), 'Give either X or K should'
        assert y is not None, 'Please give the labels'
        
        n = len(y)
        
        
        # Gram matrix
        if K is None:
            t=time.time()
            print('Building Gram matrice')
            K = self.kernel.gram(X) 
            print(f'Gram matrice built in {time.time() - t}s')
            
        q = -matrix(y, (n, 1), tc='d')
        h = matrix(np.concatenate([np.ones(n)/(2*self.C*n), np.zeros(n)]).reshape((2*n, 1)))
        P = matrix(K)
        Gtop = spmatrix(y, range(n), range(n))
        G = sparse([Gtop, -Gtop])

        solution = cvxopt.solvers.qp(P, q, G, h)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        # We take all vectorss
        sv = [True]*len(a)
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        if X is not None:
            self.sv = X[sv]
        self.sv_y = y[sv]
        #print('{0} support vectors out of {1} points'.format(len(self.a), n))

        # Intercept
        self.b = 0
        if self.intercept:
            for n in range(len(self.a)):
                self.b += self.sv_y[n]
                self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
            self.b /= len(self.a)

    
    def project(self, X = None, pairwise_K = None):
        if X is not None:
            n = len(X)
        else:
            n = len(pairwise_K)
        y_predict = np.zeros(n)
        if pairwise_K is None:
            pairwise_K = self.kernel.pairwise_kernel(X, self.sv)
        
        y_predict = np.sum(self.a * pairwise_K, axis=1)
        return y_predict + self.b

    def predict(self, X = None, pairwise_K = None):
        return np.sign(self.project(X, pairwise_K = pairwise_K))
    
    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)