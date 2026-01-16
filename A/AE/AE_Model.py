import torch
import torch.nn as nn
from piqa import SSIM
# https://docs.pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html


# https://stackoverflow.com/questions/53956932/use-pytorch-ssim-loss-function-in-my-model
# How to implement SSIM as loss function
# https://piqa.readthedocs.io/en/stable/api/piqa.ssim.html
# https://pypi.org/project/piqa/
class SSIMLoss(torch.nn.Module):
    '''
        When computing loss, convert to [0,1]
    '''
    def forward(self, x, y):
        x = x.reshape((-1, 1, 28, 28))
        x = (x + 1.0) / 2.0
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))

        y = y.reshape((-1, 1, 28, 28))
        y = (y + 1.0) / 2.0
        y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))

        ssim_class = SSIM(n_channels = 1, value_range = 1.0).cuda()
        ssim_val = ssim_class(x, y)
        return 1.0 - ssim_val

class AE(torch.nn.Module):
    def __init__(self, device, latent_dim = 2, loss_desc = "MSE"):
        super().__init__()
        input_dim = 28 * 28
        self.encoder = Encoder(input_dim = input_dim, latent_dim = latent_dim)
        self.decoder = Decoder(input_dim = input_dim, latent_dim = latent_dim)
        self.latent_dim = latent_dim
        self.name = "baseline_{}".format(loss_desc)
        self.device = device

        self.loss_desc = loss_desc

        if self.loss_desc == "MSE":
            self.loss_func = nn.MSELoss()
        elif self.loss_desc == "SSIM":
            self.loss_func = SSIMLoss()
        else:
            raise Exception("Invalid Loss Description")

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat
    
    def loss(self, x_pred, x_true):
        return self.loss_func(x_true, x_pred)
    
    def decode_from_latent(self, z):
        return self.decoder(z)
    
    def get_latent_dim(self):
        return self.latent_dim

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.l1 = Fc_Layer(dim_in = input_dim, dim_out = input_dim // 2, use_BN = True, activation = nn.LeakyReLU())
        self.l2 = Fc_Layer(dim_in = input_dim // 2, dim_out = input_dim // 4, use_BN = True, activation = nn.LeakyReLU())
        self.l3 = Fc_Layer(dim_in = input_dim // 4, dim_out = latent_dim, use_BN = False, activation = None)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x

class Decoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.l1 = Fc_Layer(dim_in = latent_dim, dim_out = input_dim // 4, use_BN = True, activation = nn.LeakyReLU())
        self.l2 = Fc_Layer(dim_in = input_dim // 4, dim_out = input_dim // 2, use_BN = True, activation = nn.LeakyReLU())
        self.l3 = Fc_Layer(dim_in = input_dim // 2, dim_out = input_dim, use_BN = False, activation = nn.Tanh())
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x
    
class Fc_Layer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, use_BN, activation = None):
        super().__init__()
        self.fc = nn.Linear(in_features = dim_in, out_features = dim_out)
        self.use_BN = use_BN
        
        if self.use_BN:
            self.bn = nn.BatchNorm1d(num_features = dim_out)
        
        self.activation = activation
        self.dropout = nn.Dropout(p = 0.1)
    
    def forward(self, x):
        x = self.fc(x)

        if self.use_BN:
            x = self.bn(x)
        
        if not(self.activation is None):
            x = self.activation(x)

        x = self.dropout(x)
        
        return x
