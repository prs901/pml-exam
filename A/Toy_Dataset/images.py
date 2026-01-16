import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imsave
from pathlib import Path

from common import RANDOM_SEED


def save_image(path: str | Path, image: np.ndarray):
    imsave(path, image)


class ImageGenerator:
    def __init__(self, width: int, height: int, sigma: float | None = None, seed=RANDOM_SEED):
        self.w = width
        self.h = height
        self.sigma = sigma
        self.seed = seed
        self.generator = np.random.default_rng(self.seed)


    def _filter(self, image: np.ndarray):
        '''
        Apply gaussian filter on image with `sigma=self.sigma`
        if this is not none. Assumes pixel values in `[0,1]`

        :param image: image to filter
        :type image: np.ndarray
        '''
        if self.sigma is None:
            image_ = 255 * image
        else:
            image_ = gaussian_filter(image, sigma=self.sigma)
            image_ /= image_.max()
            image_ *= 255
        return image_.astype(np.uint8)


    def generate_normal_image(self, k: int):
        '''
        Generate `k` points on an image

        :param k: number of points
        :type k: int
        '''
        image = np.zeros((self.w, self.h))
        x = self.generator.integers(0, self.w, size=k)
        y = self.generator.integers(0, self.h, size=k)
        image[x,y] = 1
        return self._filter(image)


    def generate_scatter_image(
        self,
        n: int,
        meanx: float = 0,
        meany: float = 0,
        stdx: float = 1,
        stdy: float = 1,
        rho: float = 1
    ):
        '''
        Generate an image with `n` white pixels scattered on it,
        according to a multivariate normal distribution with
        covariance matrix `cov`

        # Notes
        See https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case

        :param n: number of white pixels
        :type n: int
        :param stdx: standard deviation in the horizontal direction
        :type stdx: float
        :param stdx: standard deviation in the vertical direction
        :type stdx: float
        :param rho: correlation between horizontal and vertical directions
        '''
        assert n > 0
        assert stdx > 0
        assert stdy > 0
        mean = [meanx, -meany] # image indices go from top to bottom
        cov = [
            [stdx**2, rho*stdx*stdy],
            [rho*stdx*stdy, stdy**2]
        ]
        points = self.generator.multivariate_normal(mean, cov, size=n)
        points = np.rint(points).astype(np.int_)
        x = points[:,1]
        y = points[:,0]
        y *= -1 # image indices go from top to bottom
        x += self.w // 2 # centering
        y += self.h // 2 # centering
        image = np.zeros((self.w, self.h))
        image[x,y] = 1
        return self._filter(image)


if __name__ == '__main__':
    '''Just for testing'''

    image_generator = ImageGenerator(28, 28, sigma=1.0)

    test_image = image_generator.generate_normal_image(4)
    imsave('normal_image.png', test_image)

    test_image_2 = image_generator.generate_scatter_image(100, meanx=0, meany=0, stdx=2.45, stdy=1.87, rho=-0.65)
    imsave('scatter_image.png', test_image_2)
