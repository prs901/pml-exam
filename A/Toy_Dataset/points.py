import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt

from common import RANDOM_SEED

import torch


class PointsGenerator:
    def __init__(self, seed = RANDOM_SEED):
        self.seed = seed
        self.generator = np.random.default_rng(RANDOM_SEED)


    def generate_points(
        self,
        n: int,
        mean: _ArrayLikeFloat_co,
        cov: _ArrayLikeFloat_co
    ) -> np.ndarray:
        assert n > 0
        return self.generator.multivariate_normal(mean, cov, size=n)


    def generate_point_clusters(
        self,
        n: _ArrayLikeInt,
        k: int,
        means: _ArrayLikeFloat_co,
        covs: _ArrayLikeFloat_co
    ):
        assert k > 0
        assert len(means) == k
        assert len(covs) == k

        if isinstance(n, int):
            assert n > 0
            return [self.generate_points(n, means[i], covs[i]) for i in range(k)]
        else:
            assert all((n_ > 0 for n_ in n))
            return [self.generate_points(n[i], means[i], covs[i]) for i in range(k)]


    def plot_points(self, points: np.ndarray):
        assert points.ndim == 2
        assert points.shape[-1] == 2
        fig, ax = plt.subplots()
        x, y = points[:,0], points[:,1]
        ax.scatter(x, y)
        return fig, ax


    def plot_point_clusters(self, clusters: list[np.ndarray]):
        return self.plot_points(np.concat(clusters))

'''
    A pytorch dataset wrapper, for easy integration with pytorch
'''
class Point_Dataset(torch.utils.data.Dataset):
    def __init__(self, ns, means, covs):
        generator = PointsGenerator()
        k = len(ns)
        k_points = generator.generate_point_clusters(n = ns, k = k, means = means, covs = covs)
        points_torch = torch.tensor(np.concat(k_points), dtype = torch.float32)

        self.N = points_torch.shape[0]
        self.torch_points = points_torch 
        self.clusters = k_points
        self.generator = generator

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.torch_points[idx, :]
    
    def plot(self, path):
        #fig, _ = self.generator.plot_point_clusters(self.clusters)
        #fig.savefig(path)
        np_points = self.torch_points.numpy()
        plt.scatter(np_points[:, 0], np_points[:, 1])
        plt.savefig(path)
        plt.clf()
        

    '''
        Should return normalization mean and std, so it is possible to denormalize later.
    '''
    def normalize(self):
        mean = torch.zeros(2)
        std = torch.zeros(2)

        mean[0] = torch.mean(self.torch_points[:, 0])
        std[0] = torch.std(self.torch_points[:, 0])

        mean[1] = torch.mean(self.torch_points[:, 1])
        std[1] = torch.std(self.torch_points[:, 1])

        self.torch_points[:, 0] = (self.torch_points[:, 0] - mean[0]) / std[0]
        self.torch_points[:, 1] = (self.torch_points[:, 1] - mean[1]) / std[1]

        return mean, std


if __name__ == '__main__':
    '''Just for testing'''

    n = 1_000
    mean = np.array([0,0])
    cov = np.array([[1,0], [0,1]])

    ns = [100, 1_000]
    k = 2
    means = np.array([[-5, 5], [5, -5]])
    covs = np.array([cov, 2*cov])

    generator = PointsGenerator()

    points = generator.generate_points(n, mean, cov)
    fig, _ = generator.plot_points(points)
    fig.savefig('test.png')

    k_points = generator.generate_point_clusters(ns, k, means, covs)
    fig, _ = generator.plot_points(np.concat(k_points))
    fig.savefig('test2.png')

    # 4 clusters in 3D
    d3_means = np.array(
        [[-5, 0, 5],
         [5, -5, 0],
         [0, 5, -5],
         [0, 0, 0]]
    )
    d3_cov = np.array(
        [[1,0,0],
         [0,1,0],
         [0,0,1]]
    )
    d3_covs = np.array([d3_cov, 3*d3_cov, 2*d3_cov, 0.5*d3_cov])

    d3_ns = [1_100, 900, 400, 1_200]
    d3_k = 4
    d3_k_points = generator.generate_point_clusters(d3_ns, d3_k, d3_means, d3_covs)
