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
    inital_point=None, kernel='Gaussian', trend='Const', nugget=1e-10):

        super().__init__(n_restarts, optimizer,inital_point,
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
            res = minimize(self.Neglikelihood, initial_points[i,:],
            method=self.optimizer, bounds=bnds)

            opt_para[i,:] = res.x
            opt_func[i,:] = res.fun

        # Locate the optimum results
        self.theta = opt_para[np.argmin(opt_func)]

        # Update attributes
        self.NegLnlike = self.Neglikelihood(self.theta)

    def predict(self, X_test, cov_return=False):
        """GP model predicting

        Input
        -----
        X_test (array): test set, shape (n_samples, n_features)
        cov_return (bool): return/not return covariance matrix

        Output
        ------
        f: GP predictions
        SSqr: Prediction variances"""

        # Construct correlation matrix between test and train data
        k = self.Corr(self.X, X_test, 10**self.theta)

        # Mean prediction
        f = self.mu + k.T @ (cho_solve((self.L, True), self.y-self.mu*self.F))

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
