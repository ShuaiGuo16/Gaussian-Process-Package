# import all packages and set plots to be embedded inline
import numpy as np

class GaussianProcess:
    """A class that trains a Gaussian Process model
    to approximate functions"""

    def __init__(self, n_restarts, optimizer, inital_point):
        """Initialize a Gaussian Process model

        Input
        ------
        n_restarts: number of restarts of the local optimizer
        optimizer: algorithm of local optimization
        inital_point: user-specified starting point"""

        self.n_restarts = n_restarts
        self.optimizer = optimizer
        self.init_point = inital_point
