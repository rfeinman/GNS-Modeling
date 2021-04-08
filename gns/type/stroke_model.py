import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from ..utils.mvn import tikhonov, compute_cov2d, gauss_norm_to_orig
from ..mixnet.losses import gmm_losses_seq, end_losses_seq
from ..mixnet.sampling import sample_xy_full, adjust_gmm, adjust_bernoulli



class CNN(nn.Module):
    def __init__(self, new_zscore=True, downsize=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1), # (16,28,28)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1), # (32,28,28)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1), # (64,14,14)
            nn.Tanh()
        )
        self.fmap_size = (64,14,14)
        self.new_zscore = new_zscore
        self.downsize = downsize

    def forward(self, x):
        if self.downsize:
            x = F.interpolate(x, size=(28,28), mode='bilinear', align_corners=False)
        x = zscore_canvases(x, new_stats=self.new_zscore)
        x = self.features(x) # (n,8,5,5)
        x = x.permute(0,2,3,1) # (n,5,5,8)
        x = x.view(x.size(0),-1,x.size(-1)) # (n,5*5,8)
        return x


class AttendSAT(nn.Module):
    """
    Attention mechanism from Show, Attend and Tell (Xu et al., 2016)
    """
    def __init__(self, fmap_dim, hidden_dim, attention_dim):
        super().__init__()
        self.U = nn.Linear(hidden_dim, attention_dim)
        self.W = nn.Linear(fmap_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)
        self.W_a = None

    def register_annotation(self, a):
        self.W_a = self.W(a)

    def forward(self, a, h):
        """
        :param a: (n,locs,feats)
        :param h: (n,hid)
        :return alpha: (n,locs)
        """
        W_a = self.W(a) if (self.W_a is None) else self.W_a # (n,locs,att)
        U_h = self.U(h).unsqueeze(1) # (n,1,att)
        att = torch.tanh(W_a + U_h) # (n,locs,att)
        e = self.v(att) # (n,locs,1)
        e = e.squeeze(-1) # (n,locs)
        alpha = torch.softmax(e, -1) # (n,locs)

        return alpha


class AttendDOT(nn.Module):
    """
    Attention mechanism based on dot products (Luong et al., 2015)
    """
    def __init__(self, fmap_dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, fmap_dim, bias=False)

    def register_annotation(self, a):
        pass

    def forward(self, a, h):
        """
        :param a: (n,locs,feats)
        :param h: (n,hid)
        :return alpha: (n,locs)
        """
        h = self.W(h).unsqueeze(1) # (n,1,feats)
        alpha = torch.sum(a*h, dim=-1) # (n,locs)
        alpha = torch.softmax(alpha, -1) # (n,locs)

        return alpha


class Combine(nn.Module):
    def __init__(self, fmap_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fmap_dim+hidden_dim+32, output_dim),
            nn.Tanh()
        )

    def forward(self, h, z, x, start):
        """
        :param h: (n,hid)
        :param z: (n,feats)
        :param x: (n,16)
        :param start: (n,16)
        :return:
        """
        h_tilde = torch.cat([h,z,x,start], dim=1)
        h_tilde = self.net(h_tilde)

        return h_tilde


class Initialize(nn.Module):
    def __init__(self, fmap_dim, hidden_dim, num_layers, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fmap_dim+16, hidden_dim*num_layers*2),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, a, start):
        """
        :param a: (n,locs,feats)
        :param start: (n,16)
        :return hidden:
            h_0: (layers,n,hid)
            c_0: (layers,n,hid)
        """
        x = torch.cat([a.mean(dim=1), start], dim=1) # (n,feats+16)
        hidden = self.net(x) # (n,h*l*2)
        hidden = hidden.view(-1,self.hidden_dim,self.num_layers,2) # (n,h,l,2)
        hidden = hidden.permute(2,0,1,3) # (l,n,h,2)
        h_0 = hidden[...,0].contiguous()
        c_0 = hidden[...,1].contiguous()

        return h_0, c_0


class LSTMConditioned(nn.Module):
    def __init__(
            self,
            k,
            d=2,
            num_layers=1,
            hidden_dim=100,
            combine_dim=512,
            attention_mode='sat',
            attention_dim=512,
            reg_cov=1e-4,
            dropout=0.,
            recurrent_dropout=0.,
            start_dropout=0.,
            pad_val=1000,
            new_version=True,
            downsize=False
    ):
        super().__init__()

        # CNN
        self.cnn = CNN(new_zscore=new_version, downsize=downsize)
        fmap_dim = self.cnn.fmap_size[0]

        # Attention net modules
        self.f_init = Initialize(fmap_dim, hidden_dim, num_layers, dropout)
        if attention_mode == 'sat':
            self.f_att = AttendSAT(fmap_dim, hidden_dim, attention_dim)
        elif attention_mode == 'dot':
            self.f_att = AttendDOT(fmap_dim, hidden_dim)
        else:
            raise Exception("attention_mode must be either 'sat' or 'dot'.")
        self.f_combine = Combine(fmap_dim, hidden_dim, combine_dim)
        self.f_beta = nn.Sequential(
            nn.Linear(hidden_dim, fmap_dim),
            nn.Sigmoid()
        )

        # embedding modules
        self.x_embed = nn.Sequential(
            nn.Linear(d, 16),
            nn.Tanh()
        )
        self.s_embed = nn.Sequential(
            nn.Linear(d, 16),
            nn.Tanh(),
            nn.Dropout(start_dropout)
        )

        # lstm layer
        self.lstm = nn.LSTM(
            input_size=fmap_dim+16,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0.
        )

        # buffers
        p_hf = (1.-dropout)*torch.ones(hidden_dim)
        p_hr = (1.-recurrent_dropout)*torch.ones(num_layers, hidden_dim)
        p_comb = (1.-dropout)*torch.ones(combine_dim)
        self.register_buffer('p_hf', p_hf)
        self.register_buffer('p_hr', p_hr)
        self.register_buffer('p_comb', p_comb)

        # ouput layer
        # variables: [mix_probs, means, scales, corrs, endings]
        # sizes: [k, k*d, k*d, k, 1]
        self.output = nn.Linear(combine_dim, k + 2*k*d + k + 1)

        # instance variables
        self.k = k
        self.d = d
        self.reg_cov = reg_cov
        self.pad_val = pad_val

    def unpack_output(self, y):
        tsize = len(y.shape)
        assert tsize in [2,3]
        if tsize == 3:
            n, t, _ = y.shape
        else:
            n,_ = y.shape
        d, k = self.d, self.k

        # unpack outputs
        # variables: [mix_probs, means, scales, corrs, endings]
        # sizes: [k, k*d, k*d, k, 1]
        mix_probs, means, scales, corrs, v_logits = torch.split(
            y, [k, k*d, k*d, k, 1], dim=-1
        )
        mix_probs = torch.softmax(mix_probs, dim=-1) # (n,t,k)
        scales = torch.exp(scales) # (n,t,k,d)
        corrs = torch.tanh(corrs) # (n,t,k)
        v_logits = v_logits.squeeze(-1) # (n,t)

        if tsize == 3:
            # reshape (n,t,k*d) -> (n,t,k,d)
            means = means.view(n,t,k,d)
            scales = scales.view(n,t,k,d)
        else:
            # reshape (n,k*d) -> (n,k,d)
            means = means.view(n,k,d)
            scales = scales.view(n,k,d)

        # add tikhonov regularization
        if self.reg_cov > 0:
            scales, corrs = tikhonov(scales, corrs, alpha=self.reg_cov)

        return (mix_probs, means, scales, corrs), v_logits

    def mix_params(self, y_pred):
        mix_probs, means, scales, corrs = y_pred
        covs = compute_cov2d(scales, corrs)
        return mix_probs, means, covs

    def dropout_masks(self, n):
        # reshape
        p_hf = self.p_hf.unsqueeze(0).repeat(n,1) # (n,hid)
        p_hr = self.p_hr.unsqueeze(1).repeat(1,n,1) # (layers,n,hid)
        p_comb = self.p_comb.unsqueeze(0).repeat(n,1) # (n,comb)
        if self.training:
            mask_hf = torch.bernoulli(p_hf)
            mask_hr = torch.bernoulli(p_hr)
            mask_comb = torch.bernoulli(p_comb)
        else:
            mask_hf = p_hf
            mask_hr = p_hr
            mask_comb = p_comb

        return mask_hf, mask_hr, mask_comb

    def forward(self, x, x_canv, start):
        """
        :param x: (n,T,2)
        :param x_canv: (n,1,28,28)
        :param start: (n,2)
        :return:
        """
        n,T,_ = x.shape

        # pad sequences with zero start
        x = sequence_pad(x)

        # get embeddings
        x = self.x_embed(x) # (n,T,16)
        start = self.s_embed(start) # (n,16)

        # init
        a = self.cnn(x_canv) # (n,locs,feats)
        h_t, c_t = self.f_init(a, start) # (layers,n,hid)
        self.f_att.register_annotation(a)
        h = h_t[-1] # (n,hid)

        # dropout masks
        mask_hf, mask_hr, mask_comb = self.dropout_masks(n)

        # execute
        y_out = torch.zeros(n,T,self.k+2*self.k*self.d+self.k+1, device=x.device)
        for t in range(T):
            # compute context
            alpha = self.f_att(a, h) # (n,locs)
            z = torch.sum(a*alpha.unsqueeze(-1), dim=1) # (n,feats)
            beta = self.f_beta(h) # (n,feats)
            z = beta*z # (n,feats)
            # feed input to lstm
            x_input = torch.cat([x[:,t], z], dim=1) # (n,feats+16)
            x_input = x_input.unsqueeze(1) # (n,1,feats+16)
            h, (h_t, c_t) = self.lstm(x_input, (h_t, c_t))
            h = h.squeeze(1) # (n,hid)
            h = mask_hf*h # (n,hid)
            h_t = mask_hr*h_t # (layers,n,hid)
            # compute output
            y = self.f_combine(h, z, x[:,t], start) # (n,hid)
            y = mask_comb*y
            y = self.output(y)
            y_out[:,t] = y

        # compute model outputs
        y_pred, v_logits = self.unpack_output(y_out)

        return y_pred, v_logits

    def loss_fn(self, x, x_canv, start):
        # preliminaries
        mask = torch.any(x != self.pad_val, dim=-1).float() # (n,T)
        v = 1. - mask # (n,T)
        mask_v = terminate_mask(mask)
        # compute forward pass
        x_pred, v_logits = self.forward(x, x_canv, start)
        # compute losses
        loss_seq = torch.mean(gmm_losses_seq(x_pred, x, mask, full_cov=True))
        loss_end = torch.mean(end_losses_seq(v_logits, v, mask_v, logits=True))
        loss = loss_seq + loss_end
        loss_vals = {
            'total': loss.item(),
            'seq': loss_seq.item(),
            'end': loss_end.item()
        }

        return loss, loss_vals

    def losses_fn(self, x, x_canv, start, std_X=None):
        # preliminaries
        mask = torch.any(x != self.pad_val, dim=-1).float() # (n,T)
        v = 1. - mask # (n,T)
        mask_v = terminate_mask(mask)
        # compute forward pass
        x_pred, v_logits = self.forward(x, x_canv, start)
        # apply de-normalize
        if std_X is not None:
            x_pred, x = self._denormalize(x_pred, x, std_X)
        # compute losses
        losses_seq = gmm_losses_seq(x_pred, x, mask, full_cov=True)
        losses_end = end_losses_seq(v_logits, v, mask_v, logits=True)
        losses = losses_seq + losses_end

        return losses

    def _denormalize(self, x_pred, x, std_X):
        assert std_X.shape == (2,)
        mean_X = torch.zeros_like(std_X)
        # adjust targets
        x = x*std_X + mean_X
        # adjust mvn parameters
        mix_probs, means, covs = self.mix_params(x_pred)
        means, covs = gauss_norm_to_orig(means, covs, mean_X=mean_X, std_X=std_X)
        x_pred = mix_probs, means, covs

        return x_pred, x

    @torch.no_grad()
    def sample(self, canv, start, T=50, temp=1.):
        self.cpu().eval()
        height,width = canv.shape[-2:]
        if canv.shape == torch.Size([1,height,width]):
            canv = canv.unsqueeze(0)
        else:
            assert canv.shape == torch.Size([1,1,height,width])
        if start.shape == torch.Size([2]):
            start = start.unsqueeze(0)
        else:
            assert start.shape == torch.Size([1,2])
        s_embed = self.s_embed(start) # (1,16)
        a = self.cnn(canv) # (1,locs,feats)
        h_t, c_t = self.f_init(a, s_embed) # (layers,1,hid)
        self.f_att.register_annotation(a)
        h = h_t[-1] # (1,hid)
        mask_hf, mask_hr, mask_comb = self.dropout_masks(1)
        x = torch.zeros(1,2)
        output = [x]
        for t in range(T):
            # forward pass
            x_embed = self.x_embed(x) # (1,16)
            alpha = self.f_att(a, h) # (1,locs)
            z = torch.sum(a*alpha.unsqueeze(-1), dim=1) # (1,feats)
            beta = self.f_beta(h) # (1,feats)
            z = beta*z # (1,feats)
            x_input = torch.cat([x_embed, z], dim=1) # (1,feats+16)
            x_input = x_input.unsqueeze(1) # (1,1,feats+16)
            h, (h_t, c_t) = self.lstm(x_input, (h_t, c_t)) # (1,1,hid)
            h = h.squeeze(1) # (1,hid)
            h = mask_hf*h # (1,hid)
            h_t = mask_hr*h_t # (layers,1,hid)
            y = self.f_combine(h, z, x_embed, s_embed) # (1,hid)
            y = mask_comb*y # (1,hid)
            y = self.output(y) # (1,output)
            y_pred, v_logits = self.unpack_output(y)
            v_probs = torch.sigmoid(v_logits)
            if temp != 1.:
                y_pred = adjust_gmm(y_pred, temp)
                v_probs = adjust_bernoulli(v_probs, temp)

            # sample
            if t > 0:
                # sample end
                v = Bernoulli(v_probs).sample().item()
                if v == 1.:
                    break
            mix_probs, means, covs = self.mix_params(y_pred)
            x = sample_xy_full(mix_probs, means, covs) # (1,d)
            output.append(x)
        output = torch.cat(output) # (1,T,d)

        return output


def zscore_canvases(x, new_stats=True):
    if new_stats:
        return (x - 0.0243)/0.1383
    else:
        return (x - 0.0574)/0.2084

def sequence_pad(x):
    """
    Pre-process sequences before feeding them to the LSTM; add zero-vector
    to the start of each sequence
    :param x: (n,t,d)
    :return x: (n,t,d)
    """
    n,_,d = x.shape
    pad = torch.zeros(n,1,d, device=x.device)
    x = torch.cat([pad, x[:,:-1]], dim=1)
    return x

def terminate_mask(mask):
    """
    extend mask of each sample to include one more non-zero value
    """
    lengths = torch.sum(mask, dim=-1)
    mask_v = mask.clone()
    index = lengths.long().unsqueeze(-1)
    mask_v.scatter_(1,index,1.)

    return mask_v