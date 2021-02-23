# File containing all kernels that will be considered
import numpy as np



#Inspired from https://github.com/raamana/kernelmethods/blob/master/kernelmethods/numeric_kernels.py
# And http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#sigmoid


def LinearKernel(x,y):
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

def GaussianKernel(x, y, sigma=1.0):
    """

    Parameters
    ----------
    x : array-like
    y : array-like

    sigma : float, optional
         The default is 1.0.

    Returns
    -------
    TYPE
        Formula : exp( - || x-y ||² / (2*sigma²))

    """
    return np.exp(- (np.linalg.norm(x - y, ord=2) ** 2) / ( 2 * sigma ** 2 ) )

def PolyKernel(x, y, gamma = 1, b = 0, degree = 2):
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
    Returns
    -------
    TYPE
        Formula:: K(x, y) = ( b + gamma*<x, y> )^degree
        

    """
    return (b + gamma * np.dot(x, y)) ** degree


def  HadamardKernel(x, y, alpha = 1):
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
    abs_x_a = np.power(np.abs(x), alpha)
    abs_y_a = np.power(np.abs(y), alpha)

    return np.dot((abs_x_a * abs_y_a), 2 * (abs_x_a + abs_y_a))

def LaplacianKernel(x, y, gamma=1.0):
    """Laplacian kernel function
        Parameters
    ----------
    x : array-like
    y : array-like
    gamma : float, optional
         The default is 1.0.

    Returns
    -------
    TYPE
        Formula::  K_a(x, y) = -gamma* \sum( | x_i - y_i|)
    """   
    return np.exp(-gamma * np.sum(np.abs(x - y)))


def Chi2Kernel(x,y, gamma = 1.0):
    """Chi-squared kernel function
    This kernel is implemented as::
        k(x, y) = exp(-gamma * Sum [(x - y)^2 / (x + y)])
    """
    return np.exp(-gamma * np.nansum(np.power(x - y, 2) / (x + y)))



def SigmoidKernel(x, y, alpha=1.0, c=1.0):
    """SigmoidKernel
        This kernel is implemented as::
        k(x, y) = tanh(alpha * <x,y> + c)
    """
    return np.tanh(c + alpha * np.dot(x, y))
