import pickle
import numpy as np
from pathlib import Path

from points import PointsGenerator
from images import ImageGenerator, save_image


class DatasetGenerator:
    def __init__(self, output_path: str | Path):
        self.output_path = output_path


    def images(self, n: int):
        w = 28
        h = 28
        sigma = None
        generator = ImageGenerator(width=w, height=h, sigma=sigma)
        n_pixels = 100
        meanx = 0
        meany = 5
        stdx = 2.45
        stdy = 1.87
        rho = -0.65
        for i in range(n):
            image = generator.generate_scatter_image(n_pixels, meanx=meanx, meany=meany, stdx=stdx, stdy=stdy, rho=rho)
            path = self.output_path / f'scatter_{i:03}.png'
            save_image(path, image)


    def points(self, n: int, k: int):
        ns = [100, 1_000]
        means = np.array([[-5, 5], [5, -5]])
        cov = np.array([[1,0], [0,1]])
        covs = np.array([cov, 2*cov])

        generator = PointsGenerator()
        results = list()
        for i in range(n):
            k_points = generator.generate_point_clusters(ns, k, means, covs)
            results.append(k_points)
            path = self.output_path / f'points_{i:03}.png'
            fig, _ = generator.plot_points(np.concat(k_points))
            fig.savefig(path)

        points_path = self.output_path / 'points.pickle'
        with points_path.open('wb') as fd:
            pickle.dump(results, fd)


if __name__ == '__main__':
    output_path = Path('toydatasets')
    output_path.mkdir(exist_ok=True)

    data_generator = DatasetGenerator(output_path)
    data_generator.images(n=5)
    data_generator.points(n=5, k=2)
