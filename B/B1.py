import numpy as np

from kernels import GaussianKernel, SpecialKernel

from optimization import optimize_params_fixed_noise, optimize_params_free_noise
import os

from plotting import StaticPlotter, DynamicPlotter
from GP_inference import model_new_data
from GP_inference import GPSimple
from dataloading import get_data

import sys

# NOTE: A lot of code has been taken from our GP solution code for A3.

'''
    Reference function:
        x : a float
'''
def f_ref(x):
    return -(x**2.0) + 2 * (1.0 / (1 + np.exp(-10.0 * x)))

'''
    Reference function vectorized:
        X: A numpy array of floats
'''
def f_ref_vec(X):
    return np.vectorize(f_ref)(X)


'''
    To get results for gaussian kernel, run program as: python B1.py Gaussian
    To get results for special kernel, run program as: python B1.py Special
'''
def main():
    if len(sys.argv) < 2:
        raise Exception("Program was run with no kernel identifier. Provide Gaussian or Special as an argument")
    elif len(sys.argv) > 2:
        raise Exception("Program received too many arguments")

    kernel_identifier = sys.argv[1]

    if kernel_identifier == "Gaussian":
        kernel = GaussianKernel()
        Ngrid = 200
    elif kernel_identifier == "Special":
        kernel = SpecialKernel()
        Ngrid = 50
    else:
        raise Exception("Program was given unknown kernel identifier")
    
    
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    X_noisy, Y, Delta = get_data()

    # B1.2

    gp = GPSimple(kernel)

    # (a)
    ranges = ((1e-4, 10.0), *kernel.opt_ranges_simple())
    noise_y_a, theta_a = optimize_params_free_noise(X_noisy, Y, ranges, delta = None, gp = gp, Ngrid = Ngrid)
    print("\n(a)\n")
    print("noise_y = {}".format(noise_y_a))
    print("Optimal Params = {}".format(theta_a))

    # (b)
    ranges = kernel.opt_ranges_simple()
    noise_y_b = 0.0025
    theta_b = optimize_params_fixed_noise(X_noisy, Y, ranges, delta = None, gp = gp, Ngrid = Ngrid, noise_y = noise_y_b)
    print("\n(b)\n")
    print("Optimal Params = {}".format(theta_b))

    # (c)
    ranges = kernel.opt_ranges_simple()
    noise_y_c = 0.0025
    theta_c = optimize_params_fixed_noise(X_noisy - Delta, Y, ranges, delta = None, gp = gp, Ngrid = Ngrid, noise_y = noise_y_c)
    print("\n(c)\n")
    print("Optimal Params = {}".format(theta_c))

    # B1.3
    N_X_grid = 100
    # TODO: Make it include 1.0
    X_predict = np.linspace(start = -1.0, stop = 1.0, num = N_X_grid)
    f_true_vals = f_ref_vec(X_predict)

    # Plot the seperate images with the static plotter:
    gp_static_plotter = StaticPlotter(kernel_name = kernel.name, title_font_size = 16, axis_font_size = 12)

    # For (a)
    mu_star_a, sigma_star_a = model_new_data(X_noisy, Y, X_predict, noise_y_a, theta_a, delta = None, gp = gp)
    gp_static_plotter.GP_plot(X = X_noisy, Y = Y, X_predict = X_predict, f_true_vals = f_true_vals, 
            mu_star = mu_star_a, Stds = sigma_star_a, description = "Model (a)".format(kernel.name), path = "B1Plots/Model(a)_{}".format(kernel.name))

    # For (b)
    mu_star_b, sigma_star_b = model_new_data(X_noisy, Y, X_predict, noise_y_b, theta_b, delta = None, gp = gp)
    gp_static_plotter.GP_plot(X = X_noisy, Y = Y, X_predict = X_predict, f_true_vals = f_true_vals, 
            mu_star = mu_star_b, Stds = sigma_star_b, description = "Model (b)".format(kernel.name), path = "B1Plots/Model(b)_{}".format(kernel.name))
    
    # For (c)
    mu_star_c, sigma_star_c = model_new_data(X_noisy, Y, X_predict, noise_y_c, theta_c, delta = None, gp = gp)
    gp_static_plotter.GP_plot(X = X_noisy - Delta, Y = Y, X_predict = X_predict, f_true_vals = f_true_vals, 
            mu_star = mu_star_c, Stds = sigma_star_c, description = "Model (c)".format(kernel.name), path = "B1Plots/Model(c)_{}".format(kernel.name))
    
    # Plot image with all in one with the dynamic plotter:
    gp_dynamic_plotter = DynamicPlotter(plot_shape = (3, 1), kernel_description = kernel.name)

    gp_dynamic_plotter.set_title("GP Results with {}".format(kernel.name))

    gp_dynamic_plotter.add_to_plot(X = X_noisy, Y = Y, X_predict = X_predict, f_true_vals = f_true_vals, 
            mu_star = mu_star_a, Stds = sigma_star_a, description = "Model (a)", index = 1)
    
    gp_dynamic_plotter.add_to_plot(X = X_noisy, Y = Y, X_predict = X_predict, f_true_vals = f_true_vals, 
            mu_star = mu_star_b, Stds = sigma_star_b, description = "Model (b)", index = 2)
    
    gp_dynamic_plotter.add_to_plot(X = X_noisy, Y = Y, X_predict = X_predict, f_true_vals = f_true_vals, 
            mu_star = mu_star_c, Stds = sigma_star_c, description = "Model (c)", index = 3)
    
    gp_dynamic_plotter.save_plot()
    
if __name__ == "__main__":
    main()