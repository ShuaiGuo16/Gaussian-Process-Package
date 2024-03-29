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
        self.verbose = verbose
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
        k = self.X.shape[1]  # Number of dimensions

        if self.trend == 'Const':
            F = np.vstack((np.ones((n,1)), np.zeros((n*k,1))))
        else:
            print('Other trends are currently not available, switch to "Const" instead')
            F = np.vstack((np.ones((n,1)), np.zeros((n*k,1))))

        # Construct correlation matrix
        PsiDot = np.zeros(((k+1)*n, (k+1)*n))

        # 1-Build normal Psi matrix
        Psi = np.zeros((n,n))
        for i in range(n):
            Psi[i,:] = np.exp(-np.sum(theta*(self.X[i,:]-self.X)**2, axis=1))
        Psi = Psi + np.eye(n)*self.nugget
        # To avoid duplicate addition
        PsiDot[:n,:n]=Psi/2;

        # 2-Build dPsidX
        for i in range(k):
            PsiDot[:n, (i+1)*n:(i+2)*n] = 2*theta[i]*self.diff_list[i]*Psi

        # 3-Build d2PsidX2
        for i in range(k):
            # To avoid duplicate addition
            PsiDot[(i+1)*n:(i+2)*n, (i+1)*n:(i+2)*n] = \
            (2*theta[i]-4*theta[i]**2*self.diff_list[i]**2)*Psi/2

        # 4-Build d2PsidXdX
        for i in range(k-1):
            for j in range(i+1, k):
                PsiDot[(i+1)*n:(i+2)*n, (j+1)*n:(j+2)*n] = \
                -4*theta[i]*theta[j]*self.diff_list[i]*self.diff_list[j]*Psi

        # 5-Compile PsiDot
        PsiDot = PsiDot+PsiDot.T
        L = np.linalg.cholesky(PsiDot)

        # Mean estimation
        mu = np.linalg.solve(F.T @ (cho_solve((L, True), F)),
        F.T @ (cho_solve((L, True), np.vstack((self.y, self.grad)))))

        # Variance estimation
        SigmaSqr = (np.vstack((self.y, self.grad))-F@mu).T @ \
        (cho_solve((L, True), np.vstack((self.y, self.grad))-F@mu)) / ((k+1)*n)

        # Compute log-likelihood
        LnDetK = 2*np.sum(np.log(np.abs(np.diag(L))))
        NegLnLike = ((k+1)*n/2)*np.log(SigmaSqr) + 0.5*LnDetK

        # Update attributes
        self.PsiDot, self.F, self.L, self.mu, self.SigmaSqr = PsiDot, F, L, mu, SigmaSqr

        return NegLnLike.flatten()

    def fit(self, X, y, grad):
        """GEGP model training

        Input
        -----
        X (array): shape (n_samples, n_features)
        y (array): shape (n_samples, 1)
        grad (array): shape(n_samples*n_features, 1)
        """
        self.X, self.y, self.grad = X, y, grad
        self.diff_list = self.Diff(self.X)
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
            method=self.opt,
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
        self.NegLnlike = self.Neglikelihood(self.theta)

    def predict(self, X_test):
        """GEGP model predicting

        Input
        -----
        X_test (array): test set, shape (n_samples, n_features)

        Output
        ------
        f: GP predictions
        SSqr: Prediction variances"""

        n = self.X.shape[0]  # Number of training instances
        k = self.X.shape[1]  # Problem dimension
        pred_num = X_test.shape[0]  # Number of predicting samples
        theta = 10**self.theta  # Correlation length

        # Construct correlation matrix
        psi=np.zeros((n+k*n, pred_num))

        for i in range(n):

            # Configure the nominal part
            X_temp = np.tile(self.X[[i],:], (pred_num, 1))
            psi_temp = np.exp(-np.sum((X_temp-X_test)**2*theta, axis=1, keepdims=True))
            psi[i,:] = psi_temp.T

            # Configure the gradient parts
            Dpsi_temp = (X_test-X_temp)*2 @ np.diag(theta)
            for j in range(k):
                psi[(j+1)*n+i,:] = np.transpose(Dpsi_temp[:,[j]]*psi_temp)

        # Mean prediction
        f = self.mu + psi.T @ (cho_solve((self.L, True),
        np.vstack((self.y, self.grad))-self.F@self.mu))

        # Variance prediction
        SSqr = self.SigmaSqr*(1 - np.sum(psi.T * np.transpose(cho_solve((self.L, True), psi)),
        axis=1))

        # Return values
        return f.flatten(), SSqr.flatten()

    def predict_only(self, X_train, y_train, grad_train, theta, X_test):
        """Predict-only mode, with given theta value

        Input:
        -----
        X_train (array): shape (n_samples, n_features)
        y_train (array): shape (n_samples, 1)
        grad (array): shape(n_samples*n_features, 1)
        theta (array): correlation legnths for different dimensions
        X_test (array): test set, shape (n_samples, n_features)

        Output
        ------
        f: GEGP predictions
        SSqr: Prediction variances
        grad: Gradient predictions"""

        # Assign training data
        self.X, self.y, self.grad = X_train, y_train, grad_train
        self.diff_list = self.Diff(self.X)

        # Assign theta value
        self.theta = theta

        # Update relevant attributes
        self.NegLnlike = self.Neglikelihood(self.theta)

        # Nominal value predictions
        f, SSqr = self.predict(X_test)

        # Gradient predictions
        grad = self.predict_grad(X_test)

        return f, SSqr, grad
        

    def predict_grad(self, X_test):
        """GEGP model predicting gradients

        Input
        -----
        X_test (array): test set, shape (n_samples, n_features)

        Output
        ------
        grad: Gradient predictions"""

        sample_num = X_test.shape[0]  # Number of predicting samples
        problem_dim = X_test.shape[1]  # Problem dimension
        theta = 10**self.theta  # Correlation length

        # Calculate the constant part of the gradients
        gradConst = cho_solve((self.L, True),
        np.vstack((self.y, self.grad))-self.F@self.mu)

        # Calculate the full gradients
        grad = np.zeros((sample_num,problem_dim))

        for n in range(sample_num):
            X_test_temp = np.tile(X_test[[n],:], (self.X.shape[0], 1))
            r = np.exp(-np.sum((self.X-X_test_temp)**2*theta, axis=1, keepdims=True))

            # Calculate the first-order terms
            R_D_1st = np.zeros((self.X.shape[0], problem_dim))
            for i in range(problem_dim):
                R_D_1st[:,[i]] = 2*theta[i]*r*(self.X[:,[i]]-X_test[n,i])

            # Calculate the second-order terms
            for i in range(problem_dim):
                R_D_2nd = np.zeros((self.X.shape[0], problem_dim))
                for k in range(problem_dim):
                    if k == i:
                        R_D_2nd[:,[k]] = R_D_1st[:,[k]]*(-2*theta[k])*(self.X[:,[k]]-
                        X_test[n,k]) + 2*theta[k]*r
                    else:
                        R_D_2nd[:,[k]] = -2*theta[k]*(self.X[:,[k]]-
                        X_test[n,k])*R_D_1st[:,[i]]

                # Calculate gradient matrix
                grad[n,i] = np.vstack((R_D_1st[:,[i]],
                R_D_2nd.reshape((-1,1), order='F'))).T @ gradConst

        return grad
