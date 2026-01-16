import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F

from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import math

import time

from points import Point_Dataset

'''

    Change: Changed latent and embedding features of the network.
'''
class Simple_Net(nn.Module):
    def __init__(self, input_dimension , num_latent_features=512, num_embedding_features=256):
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
    
# ExponentialMovingAverage implementation as used in pytorch vision
# https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159

# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016, 
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    
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


def train(model, optimizer, scheduler, dataloader, epochs, device, n_samples, ema=True, per_epoch_callback=None, case_num = 0, mu_norm = None, std_norm = None):
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

    Returns the average time per epoch in seconds
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
        # CHANGE: Modified from (x,y) to x
        for i, x in enumerate(dataloader):
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
            per_epoch_callback(ema_model.module if ema else model, epoch, n_samples, case_num, mu_norm, std_norm)
    
    return time_total / epochs

def reporter(model, epoch, nsamples, case_num, mu_norm, std_norm):
    """Callback function used for plotting images during training"""
    # Switch to eval mode

    # Just quick fix to speed up training
    if epoch % 50 != 0:
        return
    
    normalize = mu_norm is not None and std_norm is not None
    
    model.eval()

    samples = model.sample((nsamples,2)).cpu().detach().numpy()

    if normalize:
        samples[:, 0] = samples[:, 0] * std_norm[0] + mu_norm[0]
        samples[:, 1] = samples[:, 1] * std_norm[1] + mu_norm[1]

    plt.title("Sampling Distribution at epoch = {}".format(epoch))
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig("DDPM_Training_Monitoring_{}/Distribution_epoch{}_case{}_Normalize{}.png".format(case_num,epoch, case_num, normalize))
    plt.clf()

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    case_num = 3
    normalize = True

    # Parameters
    T = 100
    learning_rate = 1e-3
    epochs = 501
    batch_size = 256

    ns = [1500, 1500]
    
    if case_num == 0:
        k = 1.0
        means = np.array([[k, k], [-k, -k]])
        covs = np.array([0.1 * k * np.eye(2), 0.1* k * np.eye(2)])
    elif case_num == 1:
        k = 3.0
        means = np.array([[k, k], [-k, -k]])
        covs = np.array([0.1 * k * np.eye(2), 0.1* k * np.eye(2)])
    elif case_num == 2:
        k = 5.0
        means = np.array([[k, k], [-k, -k]])
        covs = np.array([0.1 * k * np.eye(2), 0.1* k * np.eye(2)])
    elif case_num == 3:
        k = 50.0
        means = np.array([[k, k], [-k, -k]])
        covs = np.array([0.1 * k * np.eye(2), 0.1* k * np.eye(2)])

    train_set = Point_Dataset(ns = ns, means = means, covs = covs)

    train_set.plot("DDPM_Training_Monitoring_{}/RefData_{}_{}.png".format(case_num, case_num, normalize))


    if normalize:
        mu_norm, std_norm = train_set.normalize()
        mu_norm = mu_norm.detach().cpu().numpy()
        std_norm = std_norm.detach().cpu().numpy()
        train_set.plot("DDPM_Training_Monitoring_{}/RefData_{}_{}_Normalized.png".format(case_num, case_num, normalize))
    else:
        mu_norm, std_norm = None, None

    
    n_samples = len(train_set)

    # Download and transform train dataset
    dataloader_train = torch.utils.data.DataLoader(train_set,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Construct Unet
    # The original ScoreNet expects a function with std for all the
    # different noise levels, such that the output can be rescaled.
    # Since we are predicting the noise (rather than the score), we
    # ignore this rescaling and just set std=1 for all t.
    simple_net = Simple_Net(input_dimension = 2)

    # Construct model
    model = DDPM(simple_net, T=T).to(device)

    # Construct optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Setup simple scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)

    avg_epoch_train_time = train(model, optimizer, scheduler, dataloader_train, 
      epochs=epochs, device=device, n_samples = n_samples, ema=True, per_epoch_callback=reporter, case_num = case_num, mu_norm = mu_norm, std_norm = std_norm)
    
    print("Average Epoch Training time = {}".format(avg_epoch_train_time))






if __name__ == "__main__":
    main()