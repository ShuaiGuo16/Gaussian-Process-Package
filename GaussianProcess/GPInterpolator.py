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
        pass

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

    def fit(self, X, y):
        """GP model training

        Input
        -----
        X (array): shape (n_samples, n_features)
        y (array): shape (n_samples, 1)
        """

        self.X, self.y = X, y
        lb, ub = -3, 2

        # Generate random starting points (Latin Hypercube)
        lhd = lhs(self.X.shape[1], samples=self.n_restarts)

        # Scale random samples to the given bounds
        initial_points = (ub-lb)*lhd + lb
        initial_points = np.vstack((initial_points, self.init_point))

        # Create A Bounds instance for optimization
        bnds = Bounds(lb*np.ones(X.shape[1]),ub*np.ones(X.shape[1]))

        # Run local optimizer on all points
        opt_para = np.zeros((self.n_restarts, self.X.shape[1]))
        opt_func = np.zeros((self.n_restarts, 1))
        for i in range(self.n_restarts):
            res = minimize(self.Neglikelihood, initial_points[i,:], method=self.optimizer,
                bounds=bnds)
            opt_para[i,:] = res.x
            opt_func[i,:] = res.fun

        # Locate the optimum results
        self.theta = opt_para[np.argmin(opt_func)]

        # Update attributes
        self.NegLnlike = self.Neglikelihood(self.theta)
