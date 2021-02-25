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

    def Neglikelihood(self, theta, tau):
        """Negative log-likelihood function

        Input
        -----
        theta (array): correlation legnths for different dimensions
        tau (float): ratio between noise variance and total variance

        Output
        ------
        NegLnLike: Negative log-likelihood value"""

        theta = 10**theta    # Correlation length
        n = self.X.shape[0]  # Number of training instances

        if self.trend == 'Const':
            F = np.ones((n,1))
        elif self.trend == 'Linear':
            F = np.hstack((np.ones((n,1)), self.X))
        elif self.trend == 'Quadratic':
            # Problem dimensionality
            dim = self.X.shape[1]
            # Initialize F matrix
            F = np.ones((n,1))
            # Fill in linear part
            F = np.hstack((F, self.X))
            # Fill in quadratic part
            for i in range(dim):
                    F = np.hstack((F, self.X[:, [i]]*self.X[:,i:]))
        else:
            F = self.trend


        # Construct correlation matrix
        K = (1-self.tau)*self.Corr(self.X, self.X, theta) + self.tau*np.eye(n)
        L = np.linalg.cholesky(K)

        # Mean estimation
        mu = np.linalg.solve(F.T @ (cho_solve((L, True), F)),
        F.T @ (cho_solve((L, True), self.y)))

        # Total variance estimation
        SigmaSqr = (self.y-F@mu).T @ (cho_solve((L, True), self.y-F@mu)) / n

        # Compute log-likelihood
        LnDetK = 2*np.sum(np.log(np.abs(np.diag(L))))
        NegLnLike = (n/2)*np.log(SigmaSqr) + 0.5*LnDetK

        # Update attributes
        self.K, self.F, self.L, self.mu, self.SigmaSqr = K, F, L, mu, SigmaSqr

            return NegLnLike.flatten()
            

        def fit(self, X, y, noise='auto'):
            """GP model training

            Input
            -----
            X (array): shape (n_samples, n_features)
            y (array): shape (n_samples, 1)
            noise (string/array): noise matrix.
                                  --> 'auto': unknown homogeneous noise,
                                              estimated by GPRegressor
                                  --> array: known homogeneous/heterogeneous noise
            """

            self.X, self.y = X, y
            self.noise = noise
            lb, ub = -3, 2

            # Generate random starting points (Latin Hypercube)
            lhd = lhs(self.X.shape[1], samples=self.n_restarts)

            # Scale random samples to the given bounds
            initial_points = (ub-lb)*lhd + lb

            # Expand initial points if user specified them
            if self.init_point is not None:
                initial_points = np.vstack((initial_points, self.init_point))

            # Create A Bounds instance for optimization
            bnds = Bounds(lb*np.ones(X.shape[1]),ub*np.ones(X.shape[1]))

            # Run local optimizer on all points
            opt_para = np.zeros((self.n_restarts, self.X.shape[1]))
            opt_func = np.zeros(self.n_restarts)
            for i in range(self.n_restarts):
                res = minimize(self.Neglikelihood,
                initial_points[i,:],
                jac=self.opt['jac'],
                method=self.opt['optimizer'],
                bounds=bnds)

                opt_para[i,:] = res.x
                opt_func[i] = res.fun

                # Display optimization progress in real-time
                if self.verbose == True:
                    print('Iteration {}: Likelihood={} \n'
                    .format(str(i+1), np.min(opt_func[:i+1])))

            # Locate the optimum results
            self.theta = opt_para[np.argmin(opt_func)]

            # Update attributes
            if self.opt['jac'] is False:
                self.NegLnlike = self.Neglikelihood(self.theta)
            else:
                self.NegLnlike, self.NegLnLikeDev = self.Neglikelihood(self.theta)
