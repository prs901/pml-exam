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

from GP_inference import GPAdvanced
import os

import pyro.distributions as pdist

import arviz

from torch.distributions.multivariate_normal import MultivariateNormal

'''
    Some code has been taken from the "sampling_pyro.py" file.
    Some of the arviz code is taken from our solution to A2.
'''

#note: running parallel chains is broken in some jupyter notebook environments. just run the scripts in your terminal.
def sample_likelihood(log_likelihood_fn, init_samples, warmup_steps, num_samples, step_size):
    
    def potential_fn(params): #the potential functions used by NUTS/HMC are the negative log-likelihood. 
        return -log_likelihood_fn(params['Delta']) #parameters need to be dicts
    num_chains = init_samples.shape[0]
    mcmc_kernel = NUTS(
        potential_fn=potential_fn, #cannot be a lambda function when using parallel chains
        adapt_step_size=True,
        adapt_mass_matrix=True,
        full_mass=False, #sometimes this is unstable
        step_size = step_size
    )
    mcmc = MCMC(
        mcmc_kernel,
        initial_params={'Delta':init_samples}, #parameters need a name
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
        
    )
    mcmc.run()
    posterior_samples = mcmc.get_samples(group_by_chain=False)['Delta']
    print("Posterior Samples Shape = {}".format(posterior_samples.shape))
    #todo: need arviz

    nuts_inference = arviz.from_pyro(mcmc)

    arviz.plot_posterior(nuts_inference)
    plt.title("Posterior over Delta")
    plt.tight_layout()
    plt.savefig("B23_Posterior_Plot.png")
    plt.clf()

    arviz.plot_trace(nuts_inference, compact = False)
    plt.title("Trace over Delta")
    plt.tight_layout()
    plt.savefig("B23_Trace_Plot.png")
    plt.clf()

    delta_9_samples = posterior_samples[:, 8].numpy()
    delta_10_samples = posterior_samples[:, 9].numpy()

    plt.title("Delta 9 Scatter Plot")
    plt.scatter(delta_9_samples, delta_10_samples, color = "blue")
    plt.xlabel("Delta 9")
    plt.ylabel("Delta 10")
    plt.savefig("B23_ScatterPlot")
    plt.clf()
    return posterior_samples


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    X_noisy, Y, delta_true = get_data()
    gamma = 2.91465829
    sigma_y = 0.05035075376884422

    noise_x = 0.01

    warmup_steps = 100
    num_samples = 100
    
    dim = len(X_noisy)

    print("dim = {}".format(dim))

    kernel = GaussianKernel()
    gp = GPAdvanced(kernel)

    # MODIFIED: Modify the covariance matrix.
    #delta = pyro.sample("Delta", pdist.Normal(0, 1).expand([len(X_noisy)]).to_event(1))
    delta = pyro.sample("Delta", pdist.MultivariateNormal(loc = torch.zeros(dim), covariance_matrix=noise_x * torch.eye(dim)), obs = delta_true)
    K = gp.compute_K(X_noisy, X_noisy, gamma, delta)
    C = K + sigma_y * np.eye(dim)

    normal_dist = TorchMVN(torch.zeros(dim,dtype=torch.float64), covariance_matrix = torch.tensor(C))

    #we want to use 4 parallel chains so we crease 4 initial samples
    #init_samples = torch.randn(4,dim,dtype=torch.float64)
    #init_samples = torch.
    samples = sample_likelihood(normal_dist.log_prob, init_samples, warmup_steps, num_samples, 0.1)

    num_total_samples = samples.shape[0]

    for i in range(num_total_samples):
        delta_sample = samples[i, :].numpy()




if __name__ == "__main__":
    main()