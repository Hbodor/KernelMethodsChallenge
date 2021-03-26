# -*- coding: utf-8 -*-
import numpy as np

def solveKRR(K, y, l):
    '''
    Solves Kernel Ridge Regression system
    (K + \lambda * n * I_n) alpha = y
    Parameters
    ----------
    K : array_like (n, n)
        Training PD Kernel
    y : array_like (n)
        Training labels
    l : float
        Ridge penalty 

    Returns
    -------
    alpha : array_like (n)
       Regression coefficients
    '''
    n = len(K)
    A = K + l*n*np.eye(n)
    alpha = np.linalg.solve(A, y)
    return alpha
    
def solveWKRR(K, y, l, w):
    '''
    Solves Weighted Kernel Ridge Regression system
    \sqrt(w) * K + \lambda * n * I_n) alpha = \sqrt(w)*y
    Parameters
    ----------
    K : array_like (n, n)
        Training PD Kernel
    y : array_like (n)
        Training labels
    l : float
        Ridge penalty
    w : array_like (n)
        Weights vector

    Returns
    -------
    alpha : array_like (n)
       Regression coefficients
    '''
    alpha = solveKRR(np.sqrt(w)*K, np.sqrt(w)*y, l)
    return alpha


def solveKLRR(K, y, l):
    '''
    Solves Kernel Logistic Ridge Regression problem 
    using IRLS solver
    Parameters
    ----------
    K : array_like (n, n)
        Training PD Kernel
    y : array_like (n)
        Training labels
    l : float
        Ridge penalty 

    Returns
    -------
    alpha : array_like (n)
       Regression coefficients
    '''
    L = 100
    eps = 1e-3
    sigmoid = lambda a: 1/(1+np.exp(-a))
    n = len(K)
    
    alpha = np.zeros(n)
    
    for k in range(L):
        alpha_old = alpha
        f = K.dot(alpha_old)
        w = sigmoid(f) * sigmoid(-f)
        z = f + y / sigmoid(y*f)
        alpha = solveWKRR(K, z, 2*l, w)
        if np.linalg.norm(alpha - alpha_old)**2 < eps:
            break
        
    return alpha