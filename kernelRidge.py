import numpy as np
from kernels import LinearKernel

class KernelRidge():
    def __init__(self, kernel=LinearKernel(), l=0.01):
        self.l = l
        self.alpha = None
        self.kernel = kernel
        
    def fit(self, X, y):
        self.X_train = X.copy()
        K = self.kernel.gram(X)
        n = len(X)
        self.alpha = np.linalg.pinv(K + self.l * n * np.ones_like(K) ) @ y
        
    def project(self, X):
        if self.alpha is None:
            raise NotImplementedError
            
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            y_predict[i] += np.sum(self.kernel(self.X_train, X[i])*self.alpha)
        return y_predict
            
    def predict(self, X):
        return np.sign(self.project(X))