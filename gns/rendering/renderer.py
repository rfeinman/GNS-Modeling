import torch
import torch.nn as nn
import torch.nn.functional as F
from pybpl.parameters import Parameters
from pybpl.util.general import fspecial



def drawings_to_cpu(drawings):
    if isinstance(drawings[0], list):
        if drawings[0][0].is_cuda:
            drawings = [[stk.cpu() for stk in drawing] for drawing in drawings]
    elif drawings[0].is_cuda:
        drawings = [stk.cpu() for stk in drawings]
    return drawings

class Renderer(nn.Module):
    def __init__(self, blur_sigma=0.5, epsilon=0., blur_fsize=None, PM=None):
        super().__init__()
        if PM is None:
            PM = Parameters()

        self.painter = Painter(PM)
        self.broaden_and_blur = BroadenAndBlur(blur_sigma, epsilon, blur_fsize, PM)

    def cuda(self, device=None):
        self.painter = self.painter.cpu()
        self.broaden_and_blur = self.broaden_and_blur.cuda(device)
        return self

    def forward(self, drawings, blur_sigma=None, epsilon=None):
        """
        Render each drawing by converting the drawing to image ink
        and then applying broaden & blur filters

        Parameters
        ----------
        drawings : list[list[torch.Tensor]] | list[torch.Tensor]
            Input drawings. Each drawing is a list of tensors
        blur_sigma : float | None
            Sigma parameter for blurring. Only used for adaptive blurring.
            Default 'None' means use the blur_sigma from __init__() call

        Returns
        -------
        pimgs : torch.Tensor
            [n,H,W] Pre-conv image probabilities
        """
        # draw the strokes (this part on cpu)
        if not isinstance(drawings[0], list):
            single = True
            drawings = [drawings]
        else:
            single = False

        pimgs = self.painter(drawings)
        pimgs = self.broaden_and_blur(pimgs, blur_sigma, epsilon) # (n,H,W)

        if single:
            pimgs = pimgs[0]

        return pimgs

    def forward_partial(self, drawing, blur_sigma=None, epsilon=None, concat=False):
        """
        In this version, we include all partial canvas renders in addition
        to the final renders
        """
        if isinstance(drawing[0], list):
            pimgs = [self.painter.forward_partial(d) for d in drawing]
            lengths = [len(p) for p in pimgs]
            pimgs = torch.cat(pimgs)
            pimgs = self.broaden_and_blur(pimgs, blur_sigma, epsilon)
            if concat:
                return pimgs
            pimgs = torch.split(pimgs, lengths, 0)
            return list(pimgs)
        else:
            pimgs = self.painter.forward_partial(drawing)
            pimgs = self.broaden_and_blur(pimgs, blur_sigma, epsilon)
            return pimgs

class Painter(nn.Module):
    def __init__(self, PM=None):
        super().__init__()
        if PM is None:
            PM = Parameters()
        self.ink_pp = PM.ink_pp
        self.ink_max_dist = PM.ink_max_dist
        self.register_buffer('index_mat',
                             torch.arange(PM.imsize[0]*PM.imsize[1]).view(PM.imsize))
        self.register_buffer('space_flip', torch.tensor([-1.,1.]))
        self.imsize = PM.imsize

    @property
    def device(self):
        return self.index_mat.device

    @property
    def is_cuda(self):
        return self.index_mat.is_cuda

    def space_motor_to_img(self, stk):
        return torch.flip(stk, dims=[-1])*self.space_flip

    def check_bounds(self, myt):
        xt = myt[:,0]
        yt = myt[:,1]
        x_out = (torch.floor(xt) < 0) | (torch.ceil(xt) >= self.imsize[0])
        y_out = (torch.floor(yt) < 0) | (torch.ceil(yt) >= self.imsize[1])
        out = x_out | y_out

        return out

    def seqadd(self, D, lind_x, lind_y, inkval):
        lind = self.index_mat[lind_x.long(), lind_y.long()]
        D = D.view(-1)
        D = D.scatter_add(0, lind, inkval)
        D = D.view(self.imsize)
        return D

    def add_stroke(self, pimg, stk):
        stk = self.space_motor_to_img(stk)

        # reduce trajectory to only those points that are in bounds
        out = self.check_bounds(stk) # boolean; shape (neval,)
        ink_off_page = out.any()
        if out.all():
            return pimg, ink_off_page
        stk = stk[~out]

        # compute distance between each trajectory point and the next one
        if stk.shape[0] == 1:
            myink = stk.new_tensor(self.ink_pp)
        else:
            dist = torch.norm(stk[1:] - stk[:-1], dim=-1) # shape (k,)
            dist = dist.clamp(None, self.ink_max_dist)
            dist = torch.cat([dist[:1], dist])
            myink = (self.ink_pp/self.ink_max_dist)*dist # shape (k,)

        # make sure we have the minimum amount of ink, if a particular
        # trajectory is very small
        sumink = torch.sum(myink)
        if sumink < 2.22e-6:
            nink = myink.shape[0]
            myink = (self.ink_pp/nink)*torch.ones_like(myink)
        elif sumink < self.ink_pp:
            myink = (self.ink_pp/sumink)*myink
        assert torch.sum(myink) > (self.ink_pp - 1e-4)

        # share ink with the neighboring 4 pixels
        x = stk[:,0]
        y = stk[:,1]
        xfloor = torch.floor(x).detach()
        yfloor = torch.floor(y).detach()
        xceil = torch.ceil(x).detach()
        yceil = torch.ceil(y).detach()
        x_c_ratio = x - xfloor
        y_c_ratio = y - yfloor
        x_f_ratio = 1 - x_c_ratio
        y_f_ratio = 1 - y_c_ratio
        lind_x = torch.cat([xfloor, xceil, xfloor, xceil])
        lind_y = torch.cat([yfloor, yfloor, yceil, yceil])
        inkval = torch.cat([
            myink*x_f_ratio*y_f_ratio,
            myink*x_c_ratio*y_f_ratio,
            myink*x_f_ratio*y_c_ratio,
            myink*x_c_ratio*y_c_ratio
        ])

        # paint the image
        pimg = self.seqadd(pimg, lind_x, lind_y, inkval)

        return pimg, ink_off_page

    def draw(self, pimg, strokes):
        for stk in strokes:
            pimg, _ = self.add_stroke(pimg, stk)
        return pimg

    def forward(self, drawings):
        assert not self.is_cuda
        drawings = drawings_to_cpu(drawings)
        n = len(drawings)
        pimgs = torch.zeros(n, *self.imsize)
        for i in range(n):
            pimgs[i] = self.draw(pimgs[i], drawings[i])

        return pimgs

    def forward_partial(self, drawing):
        """
        In this version, we include all partial canvas renders in addition
        to the final renders
        """
        assert not self.is_cuda
        drawing = drawings_to_cpu(drawing)
        ns = len(drawing)
        pimgs = torch.zeros(ns+1, *self.imsize)
        canvas = torch.zeros(*self.imsize)
        for i, stk in enumerate(drawing):
            canvas, _ = self.add_stroke(canvas, stk)
            pimgs[i+1] = canvas

        return pimgs


class BroadenAndBlur(nn.Module):
    def __init__(self, blur_sigma=0.5, epsilon=0., blur_fsize=None, PM=None):
        super().__init__()
        if PM is None:
            PM = Parameters()
        if blur_fsize is None:
            blur_fsize = PM.fsize
        assert blur_fsize % 2 == 1, 'blur conv filter size must be odd'
        self.register_buffer('H_broaden', broaden_filter(PM.ink_a, PM.ink_b))
        self.register_buffer('H_blur', blur_filter(blur_fsize, blur_sigma))
        self.nbroad = PM.ink_ncon
        self.blur_pad = blur_fsize//2
        self.blur_sigma = blur_sigma
        self.blur_fsize = blur_fsize
        self.epsilon = epsilon

    @property
    def device(self):
        return self.H_broaden.device

    @property
    def is_cuda(self):
        return self.H_broaden.is_cuda

    def forward(self, x, blur_sigma=None, epsilon=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            [n,H,W] pre-conv image probabilities
        blur_sigma : float | None
            amount of blur. 'None' means use value from __init__ call

        Returns
        -------
        x : torch.Tensor
            [n,H,W] post-conv image probabilities
        """
        if self.is_cuda:
            x = x.cuda()

        if blur_sigma is None:
            H_blur = self.H_blur
            blur_sigma = self.blur_sigma
        else:
            blur_sigma = check_float_tesnor(blur_sigma, self.device)
            H_blur = blur_filter(self.blur_fsize, blur_sigma, device=self.device)

        if epsilon is None:
            epsilon = self.epsilon
        else:
            epsilon = check_float_tesnor(epsilon, self.device)

        # unsqueeze
        x = x.unsqueeze(1)
        # apply broaden
        for i in range(self.nbroad):
            x = F.conv2d(x, self.H_broaden, padding=1)
        x = F.hardtanh(x, 0., 1.)
        # return if no blur
        if blur_sigma == 0:
            x = x.squeeze(1)
            return x
        # apply blur
        for i in range(2):
            x = F.conv2d(x, H_blur, padding=self.blur_pad)
        x = F.hardtanh(x, 0., 1.)
        # apply pixel noise
        if epsilon > 0:
            x = (1-epsilon)*x + epsilon*(1-x)
        # squeeze
        x = x.squeeze(1)

        return x


def broaden_filter(a, b, device=None):
    H = b*torch.tensor(
        [[a/12, a/6, a/12],
         [a/6, 1-a, a/6],
         [a/12, a/6, a/12]],
        dtype=torch.get_default_dtype(),
        device=device
    )
    H = H[None, None]
    return H

def blur_filter(fsize, sigma, device=None):
    H = fspecial(fsize, sigma, ftype='gaussian', device=device)
    H = H[None,None]
    return H

def check_float_tesnor(x, device):
    if torch.is_tensor(x):
        assert x.shape == ()
        x = x.to(device)
    else:
        assert isinstance(x, float)
    return x