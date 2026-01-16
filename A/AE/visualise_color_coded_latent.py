import torch
import sys
from AE_Model import AE as AE_Constructor
import matplotlib.pyplot as plt
import os

sys.path.append('../')

from DataLoader import DataLoader

sys.path.append('AE/')

'''
    NOTE: Function Added to compute the latent training data
'''
@torch.no_grad()
def compute_latent_dataset(AE, dataloader, N, device):
    AE.eval()
    latents = torch.zeros((N, AE.get_latent_dim()))
    classes = torch.zeros(N)

    counter = 0
    for i, (x, y) in enumerate(dataloader):
        N_batch = x.shape[0]
        
        x = x.to(device)
        z, x_hat = AE(x)

        latents[i * N_batch:(i + 1) * N_batch] = z
        classes[i * N_batch:(i + 1) * N_batch] = y

        counter += N_batch

    return latents, classes

'''
    NOTE: Function Added to represent the latent training data in a pytorch Dataset class
'''
# https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Latent_Dataset(torch.utils.data.Dataset):
    def __init__(self, latents, classes):
        self.latents = latents 
        self.classes = classes 

        self.N = len(classes)

        assert(len(latents) == self.N)

    def __len__(self):
        return self.N 
    
    def __getitem__(self, idx):
        return (self.latents[idx, :], self.classes[idx])

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    run_number = 4
    dataloader = DataLoader()

    path = "AE_Models/model_weights_baseline_SSIM_{}.pth".format(run_number)

    batch_size = 128

    latent_dim = 2
    mnist_train = dataloader.fetch_train_data()
    N_train = len(mnist_train)

    # Download and transform train dataset
    dataloader_train_MNIST = torch.utils.data.DataLoader(mnist_train,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    AE = AE_Constructor(device = device, latent_dim = latent_dim, loss_desc = "SSIM")
    # Loads the just trained AE. 
    AE.load_state_dict(torch.load(path, weights_only=True))
    AE.eval()
    AE = AE.to(device)

    latents, classes = compute_latent_dataset(AE, dataloader_train_MNIST, N_train, device) 

    plt.title("Latent Space of Training Data")
    plt.scatter(latents[:, 0].detach().cpu().numpy(), latents[:, 1].detach().cpu().numpy(), c = classes.detach().cpu().numpy())
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()