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

    def __init__(self, n_restarts=10, optimizer='L-BFGS-B',
    inital_point=None, verbose=False,
    kernel='Gaussian', trend='Const', nugget=1e-10):

        # Display optimization log
        self.verbose = verbose

        super().__init__(n_restarts, optimizer, inital_point,
        kernel, trend, nugget)

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
        K = self.Corr(self.X, self.X, theta) + np.eye(n)*self.nugget
        L = np.linalg.cholesky(K)

        # Mean estimation
        mu = np.linalg.solve(F.T @ (cho_solve((L, True), F)),
        F.T @ (cho_solve((L, True), self.y)))
        # mu = (F.T @ (cho_solve((L, True), self.y))) / \
            # (F.T @ (cho_solve((L, True), F)))

        # Variance estimation
        SigmaSqr = (self.y-F@mu).T @ (cho_solve((L, True), self.y-F@mu)) / n

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

        # Expand initial points if user specified them
        if self.init_point is not None:
            initial_points = np.vstack((initial_points, self.init_point))

        # Create A Bounds instance for optimization
        bnds = Bounds(lb*np.ones(X.shape[1]),ub*np.ones(X.shape[1]))

        # Run local optimizer on all points
        opt_para = np.zeros((self.n_restarts, self.X.shape[1]))
        opt_func = np.zeros(self.n_restarts)
        for i in range(self.n_restarts):
            res = minimize(self.Neglikelihood, initial_points[i,:],
            method=self.optimizer, bounds=bnds)

            opt_para[i,:] = res.x
            opt_func[i] = res.fun

            # Display optimization progress in real-time
            if self.verbose == True:
                print('Iteration {}: Likelihood={} \n'
                .format(str(i+1), np.min(opt_func[:i+1])))

        # Locate the optimum results
        self.theta = opt_para[np.argmin(opt_func)]

        # Update attributes
        self.NegLnlike = self.Neglikelihood(self.theta)

    def predict(self, X_test, trend=None, cov_return=False):
        """GP model predicting

        Input
        -----
        X_test (array): test set, shape (n_samples, n_features)
        trend: trend values at test sites, shape (n_samples, n_functions)
        cov_return (bool): return/not return covariance matrix

        Output
        ------
        f: GP predictions
        SSqr: Prediction variances"""

        # Construct correlation matrix between test and train data
        k = self.Corr(self.X, X_test, 10**self.theta)

        # Mean prediction
        n = X_test.shape[0]  # Number of training instances
        dim = X_test.shape[1]  # Problem dimension

        if self.trend == 'Const':
            f = self.mu + k.T @ (cho_solve((self.L, True), self.y-self.F@self.mu))

        elif self.trend == 'Linear':
            obs = np.hstack((np.ones((n,1)), X_test))
            f = obs @ self.mu + k.T @ (cho_solve((self.L, True), self.y-self.F@self.mu))

        elif self.trend == 'Quadratic':
            obs = np.ones((n,1))
            obs = np.hstack((obs, X_test))
            for i in range(dim):
                    obs = np.hstack((obs, X_test[:, [i]]*X_test[:,i:]))
            f = obs @ self.mu + k.T @ (cho_solve((self.L, True), self.y-self.F@self.mu))

        else:
            f = trend @ self.mu + k.T @ (cho_solve((self.L, True), self.y-self.F@self.mu))


        # Variance prediction
        SSqr = self.SigmaSqr*(1 - np.diag(k.T @ (cho_solve((self.L, True), k))))

        # Calculate covariance
        if cov_return == 'True':
            Cov = self.SigmaSqr*(self.Corr(X_test, X_test, 10**self.theta)
             - k.T @ (cho_solve((self.L, True), k)))

            # Return values
            return f.flatten(), SSqr.flatten(), Cov

        else:
            # Return values
            return f.flatten(), SSqr.flatten()


    def score(self, X_test, y_test):
        """Calculate root mean squared error

        Input
        -----
        X_test (array): test set, shape (n_samples, n_features)
        y_test (array): test labels

        Output
        ------
        RMSE: the root mean square error"""

        y_pred, SSqr = self.predict(X_test)
        RMSE = np.sqrt(np.mean((y_pred-y_test.flatten())**2))

        return RMSE
