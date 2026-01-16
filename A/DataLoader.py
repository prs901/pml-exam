from torchvision import datasets, transforms, utils
import torch

import numpy as np

class DataLoader:

    def __init__(self):
        transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),    # Dequantize pixel values
        transforms.Lambda(lambda x: (x-0.5)*2.0),                    # Map from [0,1] -> [-1, 1]
        transforms.Lambda(lambda x: x.flatten())
        ])

        mnist_train = datasets.MNIST('./mnist_data', download=True, train=True, transform=transform)
        mnist_test = datasets.MNIST('./mnist_data', download=True, train=False, transform=transform)

        self.train_data = mnist_train
        self.test_data = mnist_test

        self.eval_sample_size = 1000

        self.eval_train_sample = self.sample_dataset(self.train_data, self.eval_sample_size)
        self.eval_test_sample = self.sample_dataset(self.test_data, self.eval_sample_size)

    def sample_dataset(self, data_set, size):
        data = torch.zeros((size, 1,  28, 28))
        sample_indices = np.random.choice(a = len(data_set), replace = False, size = size)

        for i, sample_index in enumerate(sample_indices):
            img, _ = data_set[sample_index]
            # Convert from [-1, 1] to [0, 1]
            img = (img + 1.0) / 2.0
            img = img.clamp(0.0, 1.0)
            data[i, 0, :, :] = img.reshape((28, 28))
        
        return data

    def fetch_train_data(self):
        return self.train_data
    
    def fetch_test_data(self):
        return self.test_data
    
    def get_eval_sample_size(self):
        return self.eval_sample_size
    
    def get_eval_train_sample(self):
        return self.eval_train_sample
    
    def get_eval_test_sample(self):
        return self.eval_test_sample