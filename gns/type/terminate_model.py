import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from neuralbpl.mixnet.sampling import adjust_bernoulli



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x

def conv_block(in_channels, out_channels, dropout=0., new_version=True):
    if new_version:
        block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
    else:
        block = ConvBlock(in_channels, out_channels, dropout)
    return block

class CNNTerminate(nn.Module):
    """
    CNN receives inputs of size (n,1,105,105)
    """
    def __init__(self, dropout=0., cnn_dropout=0., pos_weight=None,
                 new_version=True, downsize=False):
        super().__init__()

        # conv & pool layers
        self.features = nn.Sequential(
            conv_block(1, 64, cnn_dropout, new_version), # (64,14,14)
            conv_block(64, 64, cnn_dropout, new_version), # (64,7,7)
            conv_block(64, 64, cnn_dropout, new_version), # (64,3,3)
            conv_block(64, 64, cnn_dropout, new_version) # (64,1,1)
        )

        # output NN
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor([pos_weight]))
        else:
            self.pos_weight = None

        self.downsize = downsize

    def forward(self, x):
        if self.downsize:
            x = F.interpolate(x, size=(28,28), mode='bilinear', align_corners=False)
        x = zscore_canvases(x)
        x = self.features(x) # (n,64,1,1)
        x = x.view(x.size(0), -1) # (n,64)
        x = self.net(x) # (n,1)
        x = x.squeeze(-1) # (n,)

        return x

    def loss_fn(self, x, y):
        # forward model
        y_pred = self.forward(x)
        # compute losses
        loss = F.binary_cross_entropy_with_logits(y_pred, y, pos_weight=self.pos_weight)
        # compute accuracy
        num_correct = correct(y_pred, y)
        acc = float(num_correct) / x.size(0)
        loss_vals = {
            'total': loss.item(),
            'acc': acc
        }

        return loss, loss_vals

    def losses_fn(self, x, y):
        y_pred = self.forward(x)
        # compute losses
        losses = F.binary_cross_entropy_with_logits(
            y_pred, y, pos_weight=self.pos_weight, reduction='none'
        )
        return losses

    @torch.no_grad()
    def sample(self, x, temp=1.):
        self.cpu().eval()
        height,width = x.shape[-2:]
        if x.shape == torch.Size([1,height,width]):
            x = x.unsqueeze(0)
        else:
            assert x.shape == torch.Size([1,1,height,width])
        probs = torch.sigmoid(self.forward(x)) # (n,)
        if temp != 1.:
            probs = adjust_bernoulli(probs, temp)
        samples = Bernoulli(probs).sample()

        return samples

def zscore_canvases(x):
    return (x - 0.0574)/0.2084

def correct(y_pred, y):
    y_pred_c = (y_pred > 0.).float()
    num_correct = torch.sum(y_pred_c == y)

    return num_correct.item()