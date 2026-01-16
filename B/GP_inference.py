import numpy as np


class GPSimple:
    def __init__(self, kernel):
        self.kernel = kernel

    '''
        Computes the kernel matrix of the GP (K(S))
    '''
    def compute_K(self, X, Xprime, theta, delta):
        assert(delta == None)
        return self.kernel.eval(X, Xprime, theta)
    
    '''
        Computes k(S, x*)
    '''
    def compute_K_S_x_star(self, S, x_star, theta, delta):
        N = len(S)
        out = np.zeros(N)

        for i in range(N):
            out[i] = self.kernel.eval_points(S[i], x_star, theta)
        
        return out
    
    '''
        Computes k(x*, x*)
    '''
    def compute_K_x_star_x_star(self, x_star, theta, delta):
        if theta.shape == (1, ):
            theta = theta[0]

        return self.kernel.eval_points(x_star, x_star, theta)
    

class GPAdvanced:

    def __init__(self, kernel):
        self.kernel = kernel 

    '''
        Computes the kernel matrix of the GP
    '''
    def compute_K(self, X, Xprime, theta, delta):
        gamma = theta

        k = self.kernel.eval_k(X, Xprime, gamma)
        k1 = self.kernel.eval_k1(X, Xprime, gamma)
        k2 = self.kernel.eval_k2(X, Xprime, gamma)

        D = np.diag(delta)

        return k - D @ k1 - k1.T @ D + D@k2@D.T
    
    '''
        Computes k(S, x*)
    '''
    def compute_K_S_x_star(self, S, x_star, theta, delta):
        N = len(S)
        out = np.zeros(N)

        for i in range(N):
            out[i] = self.kernel.eval_points(S[i], x_star, theta) - delta[i] * self.kernel.eval_points_firstdiff(S[i], x_star, theta)
        
        return out
    
    '''
        Computes k(x*, x*)
    '''
    def compute_K_x_star_x_star(self, x_star, theta, delta):
        if theta.shape == (1, ):
            theta = theta[0]

        return self.kernel.eval_points(x_star, x_star, theta)
    

def conditional(X_train, y_train, X_predict, noise_var, theta, delta, gp):
    KS = gp.compute_K(X_train, X_train, theta, delta)
    l = KS.shape[0]

    N_train = len(X_train)

    if isinstance(gp, GPSimple):
        jitter = 1e-6
    elif isinstance(gp, GPAdvanced):
        jitter = 1e-6
    else:
        raise Exception("Invalid GP class type")

    mat = KS + (noise_var + jitter) * np.eye(l)
    
    L = np.linalg.inv(np.linalg.cholesky(mat))
    G = np.dot(L.T,L)
    alpha = np.dot(G, y_train)

    assert(G.shape == (N_train, N_train))
    assert(alpha.shape == (N_train,))

    MU = np.zeros(X_predict.shape[0])
    VAR = np.zeros(X_predict.shape[0])

    for i in range(len(X_predict)):
        x_star = X_predict[i]
        K_S_xstar = gp.compute_K_S_x_star(X_train, x_star, theta, delta)
        K_xstar_xstar = gp.compute_K_x_star_x_star(x_star, theta, delta)

        assert(K_S_xstar.shape == (N_train, ))
        assert(K_xstar_xstar.shape == ())

        mu_star = np.dot(K_S_xstar, alpha)
        var_star = K_xstar_xstar - np.dot(np.dot(K_S_xstar.T, G), K_S_xstar)

        MU[i] = mu_star
        VAR[i] = var_star
    
    return MU, VAR

def model_new_data(X, Y, X_predict, noise, theta, delta, gp):
    mu_star, var_star = conditional(X, Y, X_predict, noise, theta, delta, gp)
    sigma_star = np.sqrt(var_star)

    return mu_star, sigma_star
