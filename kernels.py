# File containing all kernels that will be considered
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist


#Inspired from https://github.com/raamana/kernelmethods/blob/master/kernelmethods/numeric_kernels.py
# And http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#sigmoid
class BaseKernel():
    def __init__(self):
        pass
    
    def __call__(self, x, y):
        pass
    
    def gram(self, X):
        K = squareform(pdist(X, metric=self))
        for i in range(len(X)):
            K[i, i] = self(X[i], X[i])
        return K
        
    def pairwise_kernel(self, X_1, X_2):
        K = cdist(X_1, X_2, metric=self)
        return K
        
class LinearKernel(BaseKernel):

    def __init__(self):
        super().__init__()
        self.name = 'Linear'
    
    def __call__(self,x,y):
        """
        Parameters
        ----------
        x : array-like
        y : array-like
    
        Returns
        -------
        float
            Formula : <x,y>
        """
        return x.dot(y.T)

    def gram(self, X):
        """
        Parameters
        ----------
        X : array-like
    
        Returns
        -------
        Gram matrice: array-like
        """
        return X.dot(X.T)
    
    def pairwise_kernel(self, X_1, X_2):
        return X_1.dot(X_2.T)

class GaussianKernel(BaseKernel):
    
    def __init__(self, sigma = 1):
        super().__init__()
        self.sigma = sigma
        self.name = 'Gaussian'

    def __call__(self, x, y):
        """
        Parameters
        ----------
        x : array-like
        y : array-like
    
        sigma : float, optional
             The default is 1.0.
    
        Returns
        -------
        float
            Formula : exp( - || x-y ||² / (2*sigma²))
    
        """
        return np.exp(- (np.linalg.norm(x - y, ord=2, axis=-1) ** 2) / \
                      ( 2 * self.sigma ** 2 ) )

    def gram(self, X):
        """
        Parameters
        ----------
        x : array-like
    
        sigma : float, optional
             The default is 1.0.
    
        Returns
        -------
        Gram matrice            
    
        """
        pairwise_dists = squareform(pdist(X, 'sqeuclidean'))
        K = np.exp(- pairwise_dists / self.sigma ** 2)
        return K
    
    def pairwise_kernel(self, X_1, X_2):
        pairwise_dists = cdist(X_1, X_2, metric='sqeuclidean')
        K = np.exp( -pairwise_dists/self.sigma**2)
        return K

class PolyKernel(BaseKernel):
    
    def __init__( self, gamma = 1, b = 0, degree = 2):
        """Polynomial kernel function    

        Parameters
        ----------
        x : array-like
        y : array-like
        gamma : float, optional
             The default is 1.0.
        b : float, optional
             The default is 0.
        degree : float, optional
                The defaukt is 2
        """
        super().__init__()
        self.gamma = gamma
        self.b = b
        self.degree = degree
        self.name = 'Polynomial'
        
    def __call__(self, x, y,):
        """Polynomial kernel function    
    
        Parameters
        ----------
        x : array-like
        y : array-like
        
        Returns
        -------
        float
            Formula:: K(x, y) = ( b + gamma*<x, y> )^degree
            
        """
        return (self.b + self.gamma * np.dot(x, y)) ** self.degree

        
    def gram(self, X):
        """Polynomial kernel function    
    
        Parameters
        ----------
        x : array-like
        y : array-like
        
        Returns
        -------
        float
            Formula:: K(x, y) = ( b + gamma*<x, y> )^degree
            
        """
        return (self.b + self.gamma * X.dot(X.T)) ** self.degree
    
    def pairwise_kernel(self, X_1, X_2):
        return (self.b + self.gamma * X_1.dot(X_2.T)) ** self.degree


class  HadamardKernel(BaseKernel):
    
    def __init__(self, alpha = 1):
        super().__init__()
        self.alpha = alpha
        self.name = 'Hadamard'

    def __call__(self, x, y):
        """Hadamard kernel function
        
        Parameters
        ----------
        x : array-like
        y : array-like
        alpha : int, optional
             The default is 1.
    
        Returns
        -------
        TYPE
            Formula::  K_a(x, y) = \Sum_k {|x_k|^a * |y_k|^a} / {2*(|x_k|^a + |y_k|^a)}
           
        """
        abs_x_a = np.power(np.abs(x), self.alpha)
        abs_y_a = np.power(np.abs(y), self.alpha)
    
        return np.dot((abs_x_a * abs_y_a), 2 * (abs_x_a + abs_y_a))

class LaplacianKernel(BaseKernel):
    
    def __init__(self, gamma = 1.0):
        """Laplacian kernel function
            Parameters
        ----------
        gamma : float, optional
             The default is 1.0.
    
        """
        super().__init__()
        self.gamma = gamma
        self.name = 'Laplacian'
    
    def __call__(self, x, y):
        """Laplacian kernel function
            Parameters
        ----------
        x : array-like
        y : array-like
        gamma : float, optional
             The default is 1.0.
    
        Returns
        -------
        float
            Formula::  K_a(x, y) = -gamma* \sum( |x_i - y_i| )
        """   
        return np.exp(-self.gamma * np.sum(np.abs(x - y)))


class Chi2Kernel(BaseKernel):
    
    def __init__(self, gamma = 1.0):
        super().__init__()
        self.gamma = gamma
        self.name = 'Chi2'

    def __call__(self, x, y):
        """Chi-squared kernel function
        This kernel is implemented as::
            k(x, y) = exp(-gamma * Sum [(x - y)^2 / (x + y)])
        """
        return np.exp(-self.gamma * np.nansum(np.power(x - y, 2) / (x + y)))



class SigmoidKernel(BaseKernel):
    
    def __init__(self, alpha = 1.0, c = 1.0):
        super().__init__()
        self.alpha = alpha
        self.c = c
        self.name = 'Sigmoid'

    def __call__(x, y, alpha=1.0, c=1.0):
        """SigmoidKernel
            This kernel is implemented as::
            k(x, y) = tanh(alpha * <x,y> + c)
        """
        return np.tanh(c + alpha * np.dot(x, y))

    def gram(self, X):
        """SigmoidKernel
            This kernel is implemented as::
            k(x, y) = tanh(alpha * <x,y> + c)
        """
        return np.tanh(self.c + self.alpha * X.dot(X.T))
    
    def pairwise_kernel(self, X_1, X_2):
        return np.tanh(self.c + self.alpha * X_1.dot(X_2.T))
