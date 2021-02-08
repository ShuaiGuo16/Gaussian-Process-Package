# import all packages 
import numpy as np

class GaussianProcess:
    """A class that trains a Gaussian Process model
    to approximate functions"""

    def __init__(self, n_restarts, optimizer, inital_point, kernel):
        """Initialize a Gaussian Process model

        Input
        ------
        n_restarts (int): number of restarts of the local optimizer
        optimizer (string): algorithm of local optimization
                   (see scipy.optimize.minimize methods)
        inital_point (array): user-specified starting point
        kernel (string): kernel type"""

        self.n_restarts = n_restarts
        self.optimizer = optimizer
        self.init_point = inital_point
        self.kernel = kernel

    def Corr(self, X1, X2, theta):
        """Construct the correlation matrix between X1 and X2
        based on specified kernel function

        Input
        -----
        X1, X2 (2D array): shape (n_samples, n_features)
        theta (array): correlation legnths for different dimensions

        Output
        ------
        K: the correlation matrix
        """

        # Initialize correlation matrix
        K = np.zeros((X1.shape[0], X2.shape[0]))

        # Compute entries of the correlation matrix
        if self.kernel == 'Gaussian':
            # Gaussian kernel
            for i in range(X1.shape[0]):
                K[i,:] = np.exp(-np.sum(theta*(X1[i,:]-X2)**2, axis=1))
