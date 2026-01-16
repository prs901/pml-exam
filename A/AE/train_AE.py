import torch
from torchvision import datasets, transforms
from AE_Model import AE as AE_Model

from tqdm import tqdm
import os
import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import time

'''
    training loop very similar to the DDPM one
'''
def train_AE(model, train_dataloader, train_set, optimizer, scheduler, per_epoch_callback, epochs, device, run_number):
    total_steps = len(train_dataloader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    time_total = 0

    for epoch in range(epochs):

        # Switch to train mode
        model.train()
        time_start = time.time()

        for i, (x, _) in enumerate(train_dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            _, x_hat = model(x)
            loss = model.loss(x_hat, x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}", lr=f"{scheduler.get_last_lr()[0]:.2E}")
            progress_bar.update()          
        
        scheduler.step()

        time_end = time.time()

        time_total += time_end - time_start

        if per_epoch_callback:
            per_epoch_callback(model, train_set, device, epoch, run_number)
    
    # https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model.state_dict(), "AE/AE_Models/model_weights_{}_{}.pth".format(model.name, run_number))
    
    return time_total / epochs


'''
    NOTE: Some code has been taken from the handed out "mnist_ddpm_solution.ipynb" notebook
'''
def main(run_number, dataloader, epochs, use_reporter):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if use_reporter:
        rep = epoch_callback
    else:
        rep = None

    batch_size = 128

    train_set = dataloader.fetch_train_data()

    # Download and transform train dataset
    dataloader_train = torch.utils.data.DataLoader(train_set,
                                                    batch_size=batch_size,
                                                    shuffle=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    latent_dim = 2

    model = AE_Model(device = device, latent_dim = latent_dim, loss_desc = "SSIM")

    model = model.to(device)

    lr_start = 1e-3
    lr_end = 1e-5

    num_epochs = epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer = optimizer, T_0 = num_epochs + 1, eta_min = lr_end)

    avg_epoch_train_time = train_AE(model, dataloader_train, train_set, optimizer, scheduler, per_epoch_callback = rep, epochs = num_epochs, device = device, run_number = run_number)

    return avg_epoch_train_time * num_epochs

@torch.no_grad()
def compute_latents(model, dataloader, N_select, device):
    model.eval()
    latents = torch.zeros((N_select, model.get_latent_dim()))
    classes = torch.zeros(N_select)

    counter = 0
    for i, (x, y) in enumerate(dataloader):
        N_batch = x.shape[0]

        if counter + N_batch > N_select:
            break
        
        x = x.to(device)
        z, x_hat = model(x)

        latents[i * N_batch:(i + 1) * N_batch] = z
        classes[i * N_batch:(i + 1) * N_batch] = y

        counter += N_batch

    latents = latents.numpy()
    classes = classes.numpy()

    if model.get_latent_dim() > 2:
        t_sne_model = TSNE(n_components = 2)
        latents = t_sne_model.fit_transform(latents)

    return latents, classes

@torch.no_grad()
def reconstruct_from_data(model, dataset, device):
    N_indices = 4
    indices = np.random.choice(a = len(dataset), size = N_indices, replace = False)
    reconstructions = np.zeros((N_indices, 28, 28))
    for i, index in enumerate(indices):
        img, label = dataset[index]
        img = img.to(device)
        img = img.reshape((1, 28 * 28))
        _, recon = model(img)
        recon = recon.reshape((28, 28))
        
        reconstructions[i, :, :] = recon.detach().cpu().numpy()
    
    return reconstructions

def plot_reconstructions(reconstructions, data_description, epoch, run_number):
    plt.title(data_description)
    plt.axis("off")
    plt.subplot(2, 2, 1)
    plt.imshow(reconstructions[0, :, :], cmap = "gray")
    plt.subplot(2, 2, 2)
    plt.imshow(reconstructions[1, :, :], cmap = "gray")
    plt.subplot(2, 2, 3)
    plt.imshow(reconstructions[2, :, :], cmap = "gray")
    plt.subplot(2, 2, 4)
    plt.imshow(reconstructions[3, :, :], cmap = "gray")

    plt.savefig(f"AE/AE_Training_Monitoring/run={run_number}_Reconstructions_{data_description}_{epoch}.png")
    plt.clf()

@torch.no_grad()
def epoch_callback(model, train_dataset, device, epoch, run_number):
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=128,
                                                    shuffle=True)
    model.eval()

    N_vis = 20000

    latents_train, classes_train = compute_latents(model, train_dataloader, N_vis, device = device)
        
    plt.title("Latent Space at epoch = {} of Train set".format(epoch))
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.scatter(latents_train[:, 0], latents_train[:, 1], c = classes_train)
    plt.savefig("AE/AE_Training_Monitoring/runumber={}_LS_Train_Epoch{}.png".format(run_number, epoch))
    plt.clf()

    reconstructions_train = reconstruct_from_data(model, train_dataset, device)
    plot_reconstructions(reconstructions_train, data_description = "Train", epoch = epoch, run_number = run_number)


