import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


from kernels import GaussianKernel
from GP_inference import GPAdvanced
from GP_inference import GPSimple

from GP_inference import model_new_data

from optimization import optimize_params_free_noise

import os

from dataloading import get_data

from B1 import f_ref_vec

from plotting import StaticPlotter


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    X_noisy, Y, Delta = get_data()
    # Part 2
    kernel = GaussianKernel()
    gp = GPAdvanced(kernel)

    Ngrid = 200

    ranges = ((1e-4, 10), *kernel.opt_ranges_advanced())
    noise_y, theta = optimize_params_free_noise(X_noisy, Y, ranges, Delta, gp, Ngrid)
    print("\n(a)\n")
    print("noise_y = {}".format(noise_y))
    print("Optimal Params = {}".format(theta))

    N_X_grid = 100
    # TODO: Make it include 1.0
    X_predict = np.linspace(start = -1.0, stop = 1.0, num = N_X_grid)
    
    mu_star, sigma_star = model_new_data(X_noisy, Y, X_predict, noise_y, theta[0], Delta, gp)

    f_true_vals = f_ref_vec(X_predict)

    # Plot the seperate images with the static plotter:
    gp_static_plotter = StaticPlotter(kernel_name = kernel.name, title_font_size = 16, axis_font_size = 12)

    gp_static_plotter.GP_plot(X_noisy, Y, X_predict, f_true_vals, mu_star, sigma_star, description = "Advanced GP", path = "B2Plots/Advanced_GP.png")








