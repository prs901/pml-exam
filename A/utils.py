from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

import torch

import einops

'''
    Computes FID

    X_true: A pytorch tensor of shape [N, 1, 28, 28] denoting the true image distribution
    X_approx: A pytorch tensor of shape [N, 1, 28, 28] denoting the approximated image distribution

    Assumes that the samples are in the range [0, 1]

    https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
'''
def FID(X_true, X_approx):

    N = X_true.shape[0]

    assert(N == X_approx.shape[0])

    min_true = torch.min(X_true)
    max_true = torch.max(X_true)

    min_approx = torch.min(X_approx)
    max_approx = torch.max(X_approx)

    assert(min_true >= 0.0)
    assert(max_true <= 1.0)
    assert(min_approx >= 0.0)
    assert(max_approx <= 1.0)

    # The inception net is trained on RGB images, so i am replicating the image over the three channels

    # https://stackoverflow.com/questions/57896357/how-to-repeat-tensor-in-a-specific-new-dimension-in-pytorch
    X_true_stacked = einops.repeat(X_true, 'n b h w -> n (repeat b) h w', repeat=3)
    X_approx_stacked = einops.repeat(X_approx, 'n b h w -> n (repeat b) h w', repeat=3)

    # Setting Normalize to true, since FID then expects floats in [0, 1]
    fid = FrechetInceptionDistance(feature = 64, input_img_size = (3, 28, 28), normalize = True)
    fid.update(X_true_stacked, real = True)
    fid.update(X_approx_stacked, real = False)
    res = fid.compute().numpy()

    return float(res)

'''
    Computes inception_score

    X_true: A pytorch tensor of shape [N, 1, 28, 28] denoting the true image distribution
    X_approx: A pytorch tensor of shape [N, 1, 28, 28] denoting the approximated image distribution

    Assumes that the samples are in the range [0, 1]

    The function computes mean inception score and standard deviation of inception score over splits
    of the provided dataset

    https://lightning.ai/docs/torchmetrics/stable/image/inception_score.html
'''
def inception_score(X_approx):
    N = X_approx.shape[0]

    assert(N == X_approx.shape[0])

    min_approx = torch.min(X_approx)
    max_approx = torch.max(X_approx)

    assert(min_approx >= 0.0)
    assert(max_approx <= 1.0)

    # The inception net is trained on RGB images, so i am replicating the image over the three channels

    # https://stackoverflow.com/questions/57896357/how-to-repeat-tensor-in-a-specific-new-dimension-in-pytorch
    X_approx_stacked = einops.repeat(X_approx, 'n b h w -> n (repeat b) h w', repeat=3)

    # Setting Normalize to true, since FID then expects floats in [0, 1]
    inception = InceptionScore(feature = 64, normalize = True, splits = 3)
    inception.update(X_approx_stacked)

    inception_res = inception.compute()
    
    return inception_res



