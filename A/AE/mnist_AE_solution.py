import torch
import torch.nn as nn

from AE_Model import AE as AE_Constructor

import os

from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import math

import torch.nn.functional as F
import numpy as np

import time

from train_AE import main as train_AE

# https://www.geeksforgeeks.org/python/python-import-from-parent-directory/
import sys

sys.path.append('../')

from utils import FID
from utils import inception_score


'''
    This is the original notebook (in a .py file), modified to work as a DDPM using a pretrained AE.

    Important logic unchanged:
        - The DDPM class is unchanged. 
        - The DDPM parameters are unchanged.
        - Training loop equavalent (i.e. only monitoring has been changed)
        - The dataloading is unchanged.

    Important Modifications:
        - The DDPM network, has been changed from the Unet to the standard net from week 5. As the latent dimensionality 
        is very low (2), the UNET is not appropriate for approximating the latent distribution, and we thus opted for the
        simpler net in week 5.

        - The sampling methods for evaluation (e.g. generate) has been modified, to generate with Decoder(p(z)), where 
            p(z) is the trained DDPM
        
        - The training monitoring has been modified, to monitor both the DDPM sampled latent space, and samples from
            Decoder(p(z)) every epoch.
'''

#neural network to learn epsilon(x_t,t). you can play around with it, but we added it for convenience
#note that this networks expects normalized values of t in 0 to 1. so divide your time index by T.

'''
    NOTE: Using the Simple Net from Week 5 instead of the Unet, as this is suited for lower dimensional
    data.
'''
class Simple_Net(nn.Module):
    def __init__(self, input_dimension , num_latent_features=128, num_embedding_features=64):
        super().__init__()
        self.num_embedding_features=num_embedding_features
        self.num_latent_features = num_latent_features
        self.fc_in = nn.Linear(input_dimension, num_latent_features)
        self.fc_emb = nn.Linear(num_embedding_features, num_latent_features)
        self.fc1 = nn.Linear(num_latent_features, num_latent_features)
        self.fc2 = nn.Linear(num_latent_features, num_latent_features)
        self.fc_out = nn.Linear(num_latent_features, input_dimension)

    def embedding(self, t):
        #We assume that t is a n-dimensional vector with values in [0,1]
        #each element in t gives the time for each element in the batch
        num_frequencies = self.num_embedding_features // 2
        frequencies = torch.logspace(0,3,num_frequencies,device=t.device, dtype=t.dtype)
        cos_feats = torch.cos(2*np.pi*frequencies.unsqueeze(0)*t.unsqueeze(1))
        sin_feats = torch.sin(2*np.pi*frequencies.unsqueeze(0)*t.unsqueeze(1))
        return torch.hstack([cos_feats,sin_feats])
    
    # NOTE: This function predicts the noise (see slide 32)
    def forward(self, x, t):
        t_embedding = self.embedding(t)
        #Create sinusoidal features and apply the silu activation
        t_embedding = F.silu(self.embedding(t))

        #print("t_Embedding shape = {}".format(t_embedding.shape))

        x_in = self.fc_in(x)
        t_emb = self.fc_emb(t_embedding)

        #print("x_in shape = {}".format(x_in.shape))
        #print("t_emb shape = {}".format(t_emb.shape))

        #linear combination of space and time features
        x = F.silu(x_in + t_emb)
        #now apply a simple feed forward network with relu activations.
        #note: we skip dropout for simplicity.
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        #transform back
        x = self.fc_out(x)
        return x


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


class DDPM(nn.Module):

    def __init__(self, network, T=100, beta_1=1e-4, beta_T=2e-2):
        """
        Initialize Denoising Diffusion Probabilistic Model

        Parameters
        ----------
        network: nn.Module
            The inner neural network used by the diffusion process. Typically a Unet.
        beta_1: float
            beta_t value at t=1 
        beta_T: [float]
            beta_t value at t=T (last step)
        T: int
            The number of diffusion steps.
        """
        
        super(DDPM, self).__init__()

        # Normalize time input before evaluating neural network
        # Reshape input into image format and normalize time value before sending it to network model
        self._network = network
        # NOTE: Modified to work with simple net.
        self.network = lambda x, t: (self._network(x, 
                                                   (t.squeeze()/T))
                                    )

        # Total number of time steps
        self.T = T

        # Registering as buffers to ensure they get transferred to the GPU automatically
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T+1))
        self.register_buffer("alpha", 1-self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))
        

    def forward_diffusion(self, x0, t, epsilon):
        '''
        q(x_t | x_0)
        Forward diffusion from an input datapoint x0 to an xt at timestep t, provided a N(0,1) noise sample epsilon. 
        Note that we can do this operation in a single step

        Parameters
        ----------
        x0: torch.tensor
            x value at t=0 (an input image)
        t: int
            step index 
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t
        ''' 

        mean = torch.sqrt(self.alpha_bar[t])*x0
        std = torch.sqrt(1 - self.alpha_bar[t])
        
        return mean + std*epsilon

    def reverse_diffusion(self, xt, t, epsilon):
        """
        p(x_{t-1} | x_t)
        Single step in the reverse direction, from x_t (at timestep t) to x_{t-1}, provided a N(0,1) noise sample epsilon.

        Parameters
        ----------
        xt: torch.tensor
            x value at step t
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t-1
        """

        mean =  1./torch.sqrt(self.alpha[t]) * (xt - (self.beta[t])/torch.sqrt(1-self.alpha_bar[t])*self.network(xt, t)) 
        std = torch.where(t>0, torch.sqrt(((1-self.alpha_bar[t-1]) / (1-self.alpha_bar[t]))*self.beta[t]), 0)
        
        return mean + std*epsilon

    
    @torch.no_grad()
    def sample(self, shape):
        """
        Sample from diffusion model (Algorithm 2 in Ho et al, 2020)

        Parameters
        ----------
        shape: tuple
            Specify shape of sampled output. For MNIST: (nsamples, 28*28)

        Returns
        -------
        torch.tensor
            sampled image            
        """
        
        # Sample xT: Gaussian noise
        xT = torch.randn(shape).to(self.beta.device)

        xt = xT
        for t in range(self.T, 0, -1):
            noise = torch.randn_like(xT) if t > 1 else 0
            t = torch.tensor(t).expand(xt.shape[0], 1).to(self.beta.device)            
            xt = self.reverse_diffusion(xt, t, noise)

        return xt

    
    def elbo_simple(self, x0):
        """
        ELBO training objective (Algorithm 1 in Ho et al, 2020)

        Parameters
        ----------
        x0: torch.tensor
            Input image

        Returns
        -------
        float
            ELBO value            
        """

        # Sample time step t
        t = torch.randint(1, self.T, (x0.shape[0],1)).to(x0.device)
        
        # Sample noise
        epsilon = torch.randn_like(x0)

        # TODO: Forward diffusion to produce image at step t
        xt = self.forward_diffusion(x0, t, epsilon)
        
        return -nn.MSELoss(reduction='mean')(epsilon, self.network(xt, t))

    
    def loss(self, x0):
        """
        Loss function. Just the negative of the ELBO.
        """
        return -self.elbo_simple(x0).mean()

'''
    NOTE: Modified to pass the AE, epoch, and device to the call back function for monitoring the training process
'''
def train(model, AE, optimizer, scheduler, dataloader, epochs, device, ema=True, per_epoch_callback=None, run_number = -1):
    """
    Training loop
    
    Parameters
    ----------
    model: nn.Module
        Pytorch model
    optimizer: optim.Optimizer
        Pytorch optimizer to be used for training
    scheduler: optim.LRScheduler
        Pytorch learning rate scheduler
    dataloader: utils.DataLoader
        Pytorch dataloader
    epochs: int
        Number of epochs to train
    device: torch.device
        Pytorch device specification
    ema: Boolean
        Whether to activate Exponential Model Averaging
    per_epoch_callback: function
        Called at the end of every epoch
    """

    # Setup progress bar
    total_steps = len(dataloader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    if ema:
        ema_global_step_counter = 0
        ema_steps = 10
        ema_adjust = dataloader.batch_size * ema_steps / epochs
        ema_decay = 1.0 - 0.995
        ema_alpha = min(1.0, (1.0 - ema_decay) * ema_adjust)
        ema_model = ExponentialMovingAverage(model, device=device, decay=1.0 - ema_alpha)                
    
    time_total = 0
    for epoch in range(epochs):

        # Switch to train mode
        time_start = time.time()
        model.train()

        global_step_counter = 0
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}", lr=f"{scheduler.get_last_lr()[0]:.2E}")
            progress_bar.update()

            if ema:
                ema_global_step_counter += 1
                if ema_global_step_counter%ema_steps==0:
                    ema_model.update_parameters(model)                
        
        time_end = time.time()

        time_total += time_end - time_start 
        if per_epoch_callback:
            per_epoch_callback(ema_model.module if ema else model, AE, epoch, device, run_number)
    
    return time_total / epochs

'''
    NOTE: Modified to monitor the training process of the DDPM_AE. 
'''
@torch.no_grad()
def reporter(model, AE, epoch, device, run_number):
    """Callback function used for plotting images during training"""
    # Switch to eval mode
    model.eval()
    AE.eval()

    nsamples = 60000
    samples = model.sample((nsamples,AE.get_latent_dim())).cpu().detach().numpy()

    plt.title("Sampling Distribution at epoch = {}".format(epoch))
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.savefig("AE/DDPM_Training_Monitoring/run={}_Distribution_epoch{}.png".format(run_number, epoch))
    plt.clf()

    num_sampled_images = 10

    sample_indices = np.random.choice(a = len(samples), replace = False, size = num_sampled_images)

    samples_selected = samples[sample_indices, :]
    samples_selected = torch.tensor(samples_selected).to(device)

    images = AE.decode_from_latent(samples_selected).detach().cpu()
    images = (images+1.0)/2.0 
    images = images.clamp(0.0, 1.0)

    images = images.reshape((-1, 1, 28, 28))

    grid = utils.make_grid(images, nrow=num_sampled_images)
    plt.gca().set_axis_off()
    plt.imshow(transforms.functional.to_pil_image(grid), cmap="gray")
    plt.savefig(f"AE/DDPM_Training_Monitoring/run={run_number}_MNIST_Samples_{epoch}.png")
    plt.clf()

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


'''
    NOTE: Modified to sample from Decoder(p(z))
'''
@torch.no_grad()
def generate(model, AE, num_samples, run_number):
    time_start = time.time()
    latent_samples = model.sample((num_samples,2))
            
    images = AE.decode_from_latent(latent_samples).detach().cpu()
    time_end = time.time()

    images = (images+1.0)/2.0 
    images = images.clamp(0.0, 1.0)
    

    plotted_samples = images[0:10, :]

    # Plot in grid
    grid = utils.make_grid(plotted_samples.reshape(-1, 1, 28, 28), nrow = num_samples)
    plt.figure(figsize=(10, 2))
    plt.axis("off")
    plt.imshow(transforms.functional.to_pil_image(grid), cmap="gray")
    plt.savefig("AE/Generations/Numbers_{}.png".format(run_number), bbox_inches="tight")
    plt.close()

    return (time_end - time_start) / num_samples, images.reshape((num_samples, 1, 28, 28))

def plot_generated_latent_space(ddpm, run_number):
    num_samples = 10000
    latent_samples = ddpm.sample((num_samples,2)).detach().cpu()

    plt.title("Latent Space")
    plt.scatter(latent_samples[:, 0], latent_samples[:, 1])
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.savefig("AE/Generations/Generated_Latent_Distribution_{}.png".format(run_number))
    plt.clf()

'''
    NOTE: DDPM parameters are unchanged.
'''
def main(run_number, dataloader, num_epochs_AE, num_epochs_ddpm, use_reporter):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if use_reporter:
        rep = reporter
    else:
        rep = None

    # Trains a new AE:
    AE_train_time = train_AE(run_number, dataloader, num_epochs_AE, use_reporter)

    # Parameters
    T = 1000
    learning_rate = 1e-3
    # TODO: Change back to 100
    epochs = num_epochs_ddpm
    batch_size = 256


    mnist_train = dataloader.fetch_train_data()
    N_train = len(mnist_train)

    # Download and transform train dataset
    dataloader_train_MNIST = torch.utils.data.DataLoader(mnist_train,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # NOTE: Added to define the AE and compute the Latent Dataset
    latent_dim = 2
    AE = AE_Constructor(device = device, latent_dim = latent_dim, loss_desc = "SSIM")
    # Loads the just trained AE. 
    AE.load_state_dict(torch.load("AE/AE_Models/model_weights_baseline_SSIM_{}.pth".format(run_number), weights_only=True))
    AE.eval()
    AE = AE.to(device)
    latent_net = Simple_Net(input_dimension = latent_dim)
   
    latents, classes = compute_latent_dataset(AE, dataloader_train_MNIST, N_train, device) 

    plt.title("Latent Space")
    plt.scatter(latents[:, 0].detach().cpu().numpy(), latents[:, 1].detach().cpu().numpy(), c = classes.detach().cpu().numpy())
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.savefig("AE/AE_Training_Monitoring/run={}_True_Latent_Distribution.png".format(run_number))
    plt.clf()

    latent_train = Latent_Dataset(latents, classes)
    dataloader_train = torch.utils.data.DataLoader(latent_train,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    # Construct model
    model = DDPM(latent_net, T=T).to(device)

    # Construct optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Setup simple scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)

    # Call training loop
    avg_epoch_train_time = train(model, AE, optimizer, scheduler, dataloader_train, 
    epochs=epochs, device=device, ema=True, per_epoch_callback=rep, run_number = run_number)

    N = dataloader.get_eval_sample_size()

    # Added for DDPM_AE:
    plot_generated_latent_space(model, run_number = run_number)

    ref_train = dataloader.get_eval_train_sample()
    ref_test = dataloader.get_eval_test_sample()

    generation_time, model_samples = generate(model, AE, num_samples = N, run_number = run_number)
    fid_train = FID(X_true = ref_train, X_approx = model_samples)
    fid_test = FID(X_true = ref_test, X_approx = model_samples)
    inception = inception_score(model_samples)


    return avg_epoch_train_time * epochs + AE_train_time, generation_time, fid_train, fid_test, inception
