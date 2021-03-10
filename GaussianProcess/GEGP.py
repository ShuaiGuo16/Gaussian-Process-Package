# import all packages
import numpy as np
import numpy.matlib
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.linalg import cho_solve
from pyDOE import lhs

class GEGP():
    """A class that trains a Gradient-enhanced
    Gaussian Process model to interpolate functions"""

    def __init__(self, n_restarts=20, opt='L-BFGS-B',
    inital_point=None, verbose=False,
    kernel='Gaussian', trend='Const', nugget=1e-10):

        """Initialize a Gradient-enhanced Gaussian Process model

        Input
        ------
        n_restarts (int): number of restarts of the local optimizer
        opt(str): specify optimization method
                   (see scipy.optimize.minimize methods)
        inital_point (array): user-specified starting points
        verbose (bool): display optimization process
        kernel (str): kernel type
        trend (str): trend type
        nugget (float): nugget term"""

        self.n_restarts = n_restarts
        self.opt = opt
        self.init_point = inital_point
        self.kernel = kernel
        self.trend = trend
        self.nugget = nugget


    def Diff(self, X):
        """Construct the difference matrix for each column of matrix X

        Input
        -----
        X (2D array): shape (n_samples, n_features)

        Output
        ------
        diff_list (list): each element is a difference matrix for
        each column of matrix X"""

        diff_list = []

        for i in range(X.shape[1]):
            temp = np.tile(X[:,[i]], (1, X.shape[0]))
            diff_list.append(temp-temp.T)

        return diff_list
