# import all packages
import numpy as np
import numpy.matlib
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.linalg import cho_solve
from pyDOE import lhs
from GaussianProcess import GaussianProcess

class GPRegressor(GaussianProcess):
    """A class that trains a Gaussian Process model
    to regress noisy data"""

    def __init__(self, n_restarts=20, opt='L-BFGS-B',
    inital_point=None, verbose=False,
    kernel='Gaussian', trend='Const', nugget=1e-10):

        # Display optimization log
        self.verbose = verbose

        super().__init__(n_restarts, opt, inital_point,
        kernel, trend, nugget)
