# import all packages
import numpy as np

class GaussianProcess:
    """A class that trains a Gaussian Process model
    to approximate functions"""

    def __init__(self, n_restarts, opt, inital_point,
    kernel, trend, nugget):
        """Initialize a Gaussian Process model

        Input
        ------
        n_restarts (int): number of restarts of the local optimizer
        opt(dict): specify optimization parameters
                   (see scipy.optimize.minimize methods)
                   {'optimizer': str, 'jac': bool}
        inital_point (array): user-specified starting points
        kernel (string): kernel type
        nugget (float): nugget term"""

        self.n_restarts = n_restarts
        self.opt = opt
        self.init_point = inital_point
        self.kernel = kernel
        self.trend = trend
        self.nugget = nugget

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

        elif self.kernel == 'Matern-3_2':
            # Matern-3/2 kernel
            for i in range(X1.shape[0]):
                comp = np.sqrt(3)*theta*np.abs(X1[i,:]-X2)
                K[i,:] = np.prod(1+comp, axis=1)*np.exp(-np.sum(comp, axis=1))

        elif self.kernel == 'Matern-5_2':
            # Matern-5/2 kernel
            for i in range(X1.shape[0]):
                comp = np.sqrt(5)*theta*np.abs(X1[i,:]-X2)
                K[i,:] = np.prod(1+comp+comp**2/3, axis=1)*np.exp(-np.sum(comp, axis=1))
        elif self.kernel == 'Cubic':
            # Cubic kernel
            for i in range(X1.shape[0]):
                comp = np.zeros_like(X2)
                diff = theta*np.abs(X1[i,:]-X2)
                # Filter values - first condition
                bool_table = (diff<1) & (diff>0.2)
                comp[bool_table] = 1.25*(1-diff[bool_table])**3
                # Filter values - second condition
                bool_table = (diff<=0.2) & (diff>=0)
                comp[bool_table] = 1-15*diff[bool_table]**2+30*diff[bool_table]**3
                # Construct kernel matrix
                K[i,:] = np.prod(comp, axis=1)

        return K
