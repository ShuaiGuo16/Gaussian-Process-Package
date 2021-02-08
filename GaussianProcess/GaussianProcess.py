# import all packages and set plots to be embedded inline
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
