import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from kernels import GaussianKernel
from common import RANDOM_SEED
from B1 import get_data


def estimate_f_prime(f: np.ndarray, x: np.ndarray):
    assert f.shape == x.shape
    return (f[2:] - f[:-2]) / (x[2:] - x[:-2])


def mse(a: np.ndarray, b: np.ndarray):
    return np.mean(np.square(a - b))


if __name__ == '__main__':
    # Part 1
    gamma = 0.9 # from B1.2(c)
    kernel = GaussianKernel()
    n = 100
    x = np.linspace(-1, 1, n)

    mean = np.zeros(2*n)
    cov = kernel.eval_cov(x, x, gamma)

    generator = np.random.default_rng(RANDOM_SEED)
    points = generator.multivariate_normal(mean, cov)

    f = points[:n]
    f_prime = points[n:]
    f_tilde = estimate_f_prime(f, x)

    discrepancy = f_prime[1:-1] - f_tilde
    # print('f_prime - f_tilde =', discrepancy)
    print('mse(f_prime, f_tilde) =', mse(f_prime[1:-1], f_tilde))

    plots_path = Path('B2Plots')
    plots_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, f, label='f')
    ax.plot(x, f_prime, label='f\'')
    fig.legend()
    fig.suptitle('GP on even grid')
    fig.savefig(plots_path / 'b2.1.png')

    x_real, y_real, delta_real = get_data()
    n_real = len(x_real)
    mean_real = np.zeros(2*n_real)
    cov_real = kernel.eval_cov(x_real, x_real, gamma)

    points_real = generator.multivariate_normal(mean_real, cov_real)

    f_real = points_real[:n_real]
    f_prime_real = points_real[n_real:]
    f_tilde_real = estimate_f_prime(f_real, x_real)

    discrepancy_real = f_prime_real[1:-1] - f_tilde_real
    print('(real data) mse(f_prime, f_tilde) =', mse(f_prime_real[1:-1], f_tilde_real))

    fig, ax = plt.subplots(1,1)
    ax.scatter(x_real, y_real, label='data')
    ax.scatter(x_real, y_real, label='data')
    ax.plot(x_real, f_real, label='f')
    ax.plot(x_real, f_prime_real, label='f\'')
    fig.legend()
    fig.suptitle('GP on real data')
    fig.savefig(plots_path / 'b2.1_real.png')

    eta = generator.random(n_real)
    y_estimate = f_real - (delta_real * f_prime_real) + eta
    print(y_estimate)

    fig, ax = plt.subplots(1,1)
    ax.plot(x_real, y_real, label='real')
    ax.plot(x_real, y_estimate, label='estimate')
    # ax.plot(x_real, f_real, label='f')
    # ax.plot(x_real, f_prime_real, label='f\'')
    fig.legend()
    fig.suptitle('')
    fig.savefig(plots_path / 'b2.1_estimate.png')
