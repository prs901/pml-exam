import scipy.optimize as opt
import numpy as np
import scipy

from GP_inference import GPSimple, GPAdvanced

# ---------------------------------------------
# Negative log-likeligood and posterior
# --------------------------------------------- 
def negLogLikelihood(X_train, y_train, noise_y, theta, delta, gp):
    """
    eta:
      - B (gaussian_kernel): [gamma]
      - C (special_kernel):  [a, b]
        noise_y is variance sigma_y^2.
    """
    # kernel matrix
    X_train_reshaped = np.asarray(X_train).reshape((len(X_train), 1))
    y_train_reshaped = np.asarray(y_train).reshape((len(y_train), 1)) 

    K = gp.compute_K(X_train_reshaped, X_train_reshaped, theta, delta)

    l = K.shape[0]

    if isinstance(gp, GPSimple):
        jitter = 1e-6
    elif isinstance(gp, GPAdvanced):
        jitter = 1e-10
    else:
        raise Exception("Unknown GP Class")

    mat = K + (noise_y + jitter) * np.eye(l)
    
    # Cholesky-based log likelihood (https://gregorygundersen.com/blog/2019/09/12/practical-gp-regression/)
    
    try: # check if matrix is PSD
        L = np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        return 1e25  # returning very large value to continue search

    alpha = scipy.linalg.cho_solve((L, True), y_train_reshaped)

    m1 = -0.5 * (y_train_reshaped.T @ alpha)
    m2 = -0.5 * (2.0 * np.sum(np.log(np.diag(L))))
    m3 = -l / 2.0 * np.log(np.sqrt(2.0 * np.pi))

    log_likelihood = m1 +  m2 + m3
    return (-log_likelihood).item()

def optimize_params_free_noise(X_train, y_train, ranges, delta, gp, Ngrid):
    obj = lambda params: negLogLikelihood(X_train, y_train, noise_y = params[0], theta = params[1:],delta = delta , gp = gp)
    opt_params = opt.brute(obj, ranges, Ns=Ngrid, finish=None) 
    opt_params = np.asarray(opt_params).reshape(-1)
    noise_var = opt_params[0]
    eta = opt_params[1:]
    return noise_var, eta

def optimize_params_fixed_noise(X_train, y_train, ranges, delta, gp, Ngrid, noise_y):
    obj = lambda params: negLogLikelihood(X_train, y_train, noise_y = noise_y, theta = params, delta = delta, gp = gp)
    opt_params = opt.brute(obj, ranges, Ns=Ngrid, finish=None) 
    opt_params = np.asarray(opt_params).reshape(-1)
    return opt_params