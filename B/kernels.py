import numpy as np
import scipy


class GaussianKernel:
    def __init__(self):
        self.name = self.__class__.__name__

    def eval_points(self, x: np.ndarray, y: np.ndarray, gamma: np.ndarray):
        return np.exp(-gamma * np.square(x - y))

    def eval_points_firstdiff(self, x: np.ndarray, y: np.ndarray, gamma: np.ndarray):
        return -2.0 * gamma * (x - y) * self.eval_points(x, y, gamma)

    def eval_points_seconddiff(self, x: np.ndarray, y: np.ndarray, gamma: np.ndarray):
        return 2.0 * gamma * self.eval_points(x, y, gamma) + 2.0 * gamma * (x - y) * self.eval_points_firstdiff(x, y, gamma)

    def eval(self, X: np.ndarray, Xprime: np.ndarray, theta: list[float]):
        '''

        :param X: D x D array
        :type X: np.ndarray
        :param Xprime: D x D array
        :type Xprime: np.ndarray
        :param theta: model parameters, here `[gamma]`
        :type theta: list
        '''
        assert(len(theta) == 1)

        X = np.asarray(X).reshape(-1, 1)
        Xprime = np.asarray(Xprime).reshape(-1, 1)

        gamma = theta[0]
        gamma = float(np.asarray(theta).reshape(-1)[0])

        dists = scipy.spatial.distance.cdist(X, Xprime, metric = 'sqeuclidean')

        return np.exp(-gamma*dists)

    def eval_k(self, X: np.ndarray, Xprime: np.ndarray, gamma: float) -> np.ndarray:
        return self.eval(X, Xprime, [gamma])

    def eval_k1(self, X: np.ndarray, Xprime: np.ndarray, gamma: float):
        N = len(X)
        assert(N == len(Xprime))
        out = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                x = X[i]
                y = Xprime[j]
                out[i, j] = self.eval_points_firstdiff(x, y, gamma)

        return out

    def eval_k2(self, X: np.ndarray, Xprime: np.ndarray, gamma: float):
        N = len(X)
        assert(N == len(Xprime))
        out = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                x = X[i]
                y = Xprime[j]
                out[i, j] = self.eval_points_seconddiff(x, y, gamma)

        return out

    def eval_cov(self, X: np.ndarray, Xprime: np.ndarray, gamma: float):
        '''
        Evaluate the covariance matrix
        ```
        |  K(X)    K1(X) |
        |                |
        | K1(X)    K2(X) |
        ```
        :param X: first vector
        :type X: np.ndarray
        :param Xprime: second vector
        :type Xprime: np.ndarray
        :param gamma: gaussian model parameter
        :type gamma: float
        '''
        n = len(X)
        assert n == len(Xprime)
        cov = np.zeros((2*n,2*n))
        k = self.eval_k(X, Xprime, gamma)
        k1 = self.eval_k1(X, Xprime, gamma)
        k2 = self.eval_k2(X, Xprime, gamma)
        cov[:n,:n] = k
        cov[n:,:n] = k1
        cov[:n,n:] = k1.T
        cov[n:,n:] = k2
        return cov


    def opt_ranges_simple(self):
        '''
        Return the optimization range of the kernel's parameter
        '''
        return ((1e-4, 10.0), )

    def opt_ranges_advanced(self):
        '''
        Return the optimization range of the kernel's parameter
        '''
        return ((1e-4, 20.0), )


class SpecialKernel:
    def __init__(self):
        self.name = self.__class__.__name__

    def eval(self, X: np.ndarray, Xprime: np.ndarray, theta: list[float]):
        '''

        :param X: D x D array
        :type X: np.ndarray
        :param Xprime: D x D array
        :type Xprime: np.ndarray
        :param theta: model parameters
        :type theta: list
        '''
        assert(len(theta) == 2)
        a = theta[0]
        b = theta[1]

        K = (1 + X @ Xprime.T) ** 2 + a * np.multiply.outer(
            np.sin(2 * np.pi * X.reshape(-1) + b),
            np.sin(2 * np.pi * Xprime.reshape(-1) +b ))

        return K

    def opt_ranges(self):
        '''
        Return the optimization range of the kernel's parameters
        '''
        return ((1e-4, 10.0), (1e-4, 10.0))
