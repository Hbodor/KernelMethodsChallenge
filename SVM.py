import numpy as np
import cvxopt
import cvxopt.solvers
from kernels import LinearKernel
import time

cvxopt.solvers.options['show_progress'] = True

#Class for Soft Margin SVM with kernel function
# Implementation with cvxopt
class SVM(object):
    
    def __init__(self, kernel=LinearKernel() , C=None):
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
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)
    
    def fit(self, X, y):
        """

        Parameters
        ----------
        X : array-like of dim (n,k), n = number of samles, k = number of features
            Data
        y : array-like (n,)
            Labels
        """
        
        n, k = X.shape
        t=time.time()
        print('Building Gram matrice')
        # Gram matrix
        K = self.kernel.gram(X) 
        print(f'Gram matrice built in {time.time() - t}s')
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n) * -1)
        A = cvxopt.matrix(y, (1,n))
        b = cvxopt.matrix(0.0)
        
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n) * -1))
            h = cvxopt.matrix(np.zeros(n))
        else:
            tmp1 = np.diag(np.ones(n) * -1)
            tmp2 = np.identity(n)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n)
            tmp2 = np.ones(n) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problemlinear_kernel
        A = cvxopt.matrix(A, (1, n), 'd')
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        #print('{0} support vectors out of {1} points'.format(len(self.a), n))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel.name == 'Linear':
            self.w = np.zeros(k)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
    
    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            K = self.kernel.pairwise_kernel(X, self.sv)
            y_predict = np.sum(self.a * self.sv_y * K, axis=1)
        return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))