
import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from kernels import GaussianKernel
from common import RANDOM_SEED
from dataloading import get_data
from torch.distributions.multivariate_normal import MultivariateNormal as TorchMVN
import torch
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC

import matplotlib

from GP_inference import GPAdvanced
import os

import pyro.distributions as pdist

import arviz

from GP_inference import conditional

from plotting import StaticPlotter

from B1 import f_ref_vec

def get_noise_x():
    return 0.01

def get_sigma_y():
    return 0.05035075376884422

def get_gamma():
    return 2.91465829


def model(X, Y = None):
    kernel = GaussianKernel()
    gp = GPAdvanced(kernel)

    noise_x = get_noise_x()
    sigma_y = get_sigma_y()
    gamma = get_gamma()
    dim = len(X)

    delta = pyro.sample("Delta", pdist.MultivariateNormal(loc = torch.zeros(dim), covariance_matrix=noise_x * torch.eye(dim)))

    K = gp.compute_K(X, X, gamma, delta.detach().cpu().numpy())
    C = K + sigma_y * np.eye(dim)
    C = torch.tensor(C)

    y = pyro.sample("y", pdist.MultivariateNormal(loc = torch.zeros(dim), covariance_matrix=C), obs = torch.tensor(Y))

    return y


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    X_noisy, Y, delta_true = get_data()
    warmup_steps = 1000
    num_samples = 1000

    nuts_kernel = NUTS(model, jit_compile = True, ignore_jit_warnings = True, step_size = 1.0)
    mcmc = MCMC(nuts_kernel, num_samples = num_samples, warmup_steps = warmup_steps, num_chains = 4)
    mcmc.run(X_noisy, Y)
    posterior_parameter_samples = mcmc.get_samples()['Delta']

    print("Posterior Samples Shape = {}".format(posterior_parameter_samples.shape))
    nuts_inference = arviz.from_pyro(mcmc)

    arviz.plot_posterior(nuts_inference)
    plt.title("Posterior over Delta")
    plt.tight_layout()
    plt.savefig("B23_Posterior_Plot.png")
    plt.close()

    arviz.plot_trace(nuts_inference, compact = False)
    plt.title("Trace over Delta")
    plt.tight_layout()
    plt.savefig("B23_Trace_Plot.png")
    plt.close()

    delta_9_samples = posterior_parameter_samples[:, 9].numpy()
    delta_10_samples = posterior_parameter_samples[:, 10].numpy()

    plt.title("Delta 9 Scatter Plot")
    plt.scatter(delta_9_samples, delta_10_samples, color = "blue")
    plt.xlabel("Delta 9")
    plt.ylabel("Delta 10")
    plt.tight_layout()
    plt.savefig("B23_ScatterPlot")
    #plt.show()
    plt.clf()

    #num_total_samples = posterior_parameter_samples.shape[0]
    num_approx_samples = 100

    kernel = GaussianKernel()
    gp = GPAdvanced(kernel)

    N_X_grid = 100

    X_predict = np.linspace(start = -1.0, stop = 1.0, num = N_X_grid)

    MU_avg = np.zeros(N_X_grid)
    STD_avg = np.zeros(N_X_grid)

    for i in range(num_approx_samples):
        sigma_y = get_sigma_y()
        gamma = get_gamma()
        delta_sample = posterior_parameter_samples[i, :].numpy()

        mu, std = conditional(X_noisy, Y, X_predict, sigma_y, theta = np.array([gamma]), delta = delta_sample, gp = gp)

        MU_avg += mu
        STD_avg += std 
    
    MU_avg = MU_avg / num_approx_samples
    STD_avg = STD_avg / num_approx_samples

    f_true_vals = f_ref_vec(X_predict)

    plotter = StaticPlotter(kernel_name = "Gaussian")
    plotter.GP_plot(X_noisy, Y, X_predict, f_true_vals, MU_avg, STD_avg, description = "B23_f", path = "B23_f.png")










        






if __name__ == "__main__":
    main()