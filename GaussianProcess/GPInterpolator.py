# import all packages
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.linalg import cho_solve
from pyDOE import lhs
from GaussianProcess import GaussianProcess

class GPInterpolator(GaussianProcess):
    """A class that trains a Gaussian Process model
    to interpolate functions"""

    def __init__(self):

    def Neglikelihood(self, theta):
        """Negative log-likelihood function

        Input
        -----
        theta (array): correlation legnths for different dimensions

        Output
        ------
        NegLnLike: Negative log-likelihood value"""

        theta = 10**theta    # Correlation length
        n = self.X.shape[0]  # Number of training instances

        if self.trend == 'Const':
            F = np.ones((n,1))

        # Construct correlation matrix
        K = self.Corr(self.X, self.X, theta) + np.eye(n)*self.nugget
        L = np.linalg.cholesky(K)

        # Mean estimation
        mu = (F.T @ (cho_solve((L, True), self.y))) / \
            (F.T @ (cho_solve((L, True), F)))

        # Variance estimation
        SigmaSqr = (self.y-mu*F).T @ (cho_solve((L, True), self.y-mu*F)) / n

        # Compute log-likelihood
        LnDetK = 2*np.sum(np.log(np.abs(np.diag(L))))
        LnLike = -(n/2)*np.log(SigmaSqr) - 0.5*LnDetK

        # Update attributes
        self.K, self.F, self.L, self.mu, self.SigmaSqr = K, F, L, mu, SigmaSqr

        return -LnLike.flatten()
