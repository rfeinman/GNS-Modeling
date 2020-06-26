import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralbpl.utils.mvn import tikhonov, compute_cov2d, gauss_norm_to_orig
from neuralbpl.mixnet.losses import gmm_losses
from neuralbpl.mixnet.sampling import sample_xy_full, adjust_gmm


def conv_block(in_channels, out_channels, dropout=0.):
    block = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(dropout)
    )
    return block

class CNNStart(nn.Module):
    def __init__(self, k, dropout=0., cnn_dropout=0., reg_cov=1e-3, alpha=0.,
                 downsize=False):
        super().__init__()
        d = 2

        # conv & pool layers
        c = 16
        self.features = nn.Sequential(
            conv_block(1,c,dropout=cnn_dropout), # (14,14)
            conv_block(c,2*c,dropout=cnn_dropout), # (7,7)
            conv_block(2*c,4*c,dropout=cnn_dropout), # (3,3)
            conv_block(4*c,8*c,dropout=cnn_dropout) # (1,1)
        ) # final is (1,1,128)

        # output NN
        self.output = nn.Sequential(
            nn.Linear(128+16, 128),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(128, k + 2*k*d + k)
        )

        # previous embedding
        self.prev_embed = nn.Sequential(
            nn.Linear(d, 16),
            nn.Tanh(),
            nn.Dropout(0.2)
        )

        # instance variables
        self.k = k
        self.d = d
        self.reg_cov = reg_cov
        self.alpha = alpha
        self.downsize = downsize

    def unpack_output(self, y):
        k, d = self.k, self.d

        # unpack outputs
        # variables: [mix_probs, means, scales, corrs]
        # sizes: [k, k*d, k*d, k]
        mix_probs, means, scales, corrs = torch.split(
            y, [k, k*d, k*d, k], dim=-1
        )
        mix_probs = torch.softmax(mix_probs, dim=-1) # (n,k)
        scales = torch.exp(scales) # (n,k,d)
        corrs = torch.tanh(corrs) # (n,k)

        # reshape means & scales
        means = means.view(-1,k,d) # (n,k*d) -> (n,k,d)
        scales = scales.view(-1,k,d) # (n,k*d) -> (n,k,d)
        # tikhonov regularization
        if self.reg_cov > 0:
            scales, corrs = tikhonov(scales, corrs, alpha=self.reg_cov)

        return mix_probs, means, scales, corrs

    def forward(self, x, prev):
        prev = self.prev_embed(prev) # (n,16)
        if self.downsize:
            x = F.interpolate(x, size=(28,28), mode='bilinear', align_corners=False)
        x = zscore_canvases(x)
        x = self.features(x) # (n,64,1,1)
        x = x.view(x.size(0), -1) # (n,64)
        x = torch.cat([x,prev],dim=-1) # (n,64+2)
        y = self.output(x) # (n,output)
        y_pred = self.unpack_output(y)

        return y_pred

    def loss_fn(self, x_canv, start, prev):
        # forward model
        start_pred = self.forward(x_canv, prev)

        # gmm loss
        loss_gmm = torch.mean(gmm_losses(start_pred, start, full_cov=True))

        # entropy reg
        entropy = mix_entropy(start_pred[0], eps=0.01) # (n,)
        if self.alpha > 0:
            loss_reg = self.alpha*torch.mean(entropy)
        else:
            loss_reg = torch.tensor(0.)

        # total losses
        loss = loss_gmm + loss_reg
        loss_vals = {
            'total': loss.item(),
            'gmm': loss_gmm.item(),
            'reg': loss_reg.item()
        }

        return loss, loss_vals

    def losses_fn(self, x_canv, start, prev, mean_X=None, std_X=None):
        # forward model
        start_pred = self.forward(x_canv, prev)

        # apply de-normalize
        if (mean_X is not None) and (std_X is not None):
            start_pred, start = self._denormalize(start_pred, start, mean_X, std_X)

        # gmm loss
        losses = gmm_losses(start_pred, start, full_cov=True) # (n,)
        return losses

    def _denormalize(self, start_pred, start, mean_X, std_X):
        assert mean_X.shape == (2,)
        assert std_X.shape == (2,)
        # adjust targets
        start = start*std_X + mean_X
        # adjust mvn parameters
        mix_probs, means, covs = self.mix_params(start_pred)
        means, covs = gauss_norm_to_orig(means, covs, mean_X=mean_X, std_X=std_X)
        start_pred = mix_probs, means, covs

        return start_pred, start

    def mix_params(self, y_pred):
        mix_probs, means, scales, corrs = y_pred
        covs = compute_cov2d(scales, corrs)
        return mix_probs, means, covs

    @torch.no_grad()
    def sample(self, canv, prev, temp=1.):
        self.cpu().eval()
        height,width = canv.shape[-2:]
        if canv.shape == torch.Size([1,height,width]):
            canv = canv.unsqueeze(0)
        else:
            assert canv.shape == torch.Size([1,1,height,width])
        if prev.shape == torch.Size([2]):
            prev = prev.unsqueeze(0)
        else:
            assert prev.shape == torch.Size([1,2])
        y_pred = self.forward(canv, prev)
        if temp != 1.:
            y_pred = adjust_gmm(y_pred, temp)
        mix_probs, means, covs = self.mix_params(y_pred) # (n,k) (n,k,d) (n,k,d,d)
        y = sample_xy_full(mix_probs, means, covs)

        return y

def zscore_canvases(x):
    return (x - 0.0243)/0.1383

def mix_entropy(p, eps):
    """
    :param p: [(n,k) tensor]
    :param eps: [float]
    :return entropy: [(n,) tensor]
    """
    entropy = -torch.sum(p*torch.log(p + eps), dim=-1) # (n,)
    return entropy