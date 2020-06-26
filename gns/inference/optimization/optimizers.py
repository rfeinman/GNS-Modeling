"""
Stochastic gradient descent with noisy gradients (instead of minibatch).
"""
import math
import random
import torch
import torch.optim as optim



class SGD(optim.SGD):
    """
    This version uses multiplicative gradient noise.
    """
    def __init__(self, params, noise=0.1, **kwargs):
        self.noise = noise
        super().__init__(params, **kwargs)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                eps = random.normalvariate(1, self.noise)
                p.grad.data.mul_(eps)
        super().step()

        return loss

class SGDAdditive(optim.SGD):
    """
    This version uses additive gradient noise with annealing.
    """
    def __init__(self, params, noise=50., noise_decay=0.005, **kwargs):
        self._noise_var = noise
        self._noise_decay = noise_decay
        super().__init__(params, **kwargs)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        noise_scale = math.sqrt(self._noise_var)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                eps = noise_scale * torch.randn_like(d_p)
                d_p.add_(eps)
        super().step()
        self._noise_var *= (1 - self._noise_decay)

        return loss

class AdamS(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False,
                 noise_init=50., noise_decay=0.005):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        noise_init=noise_init, noise_decay=noise_decay)
        super(optim.Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            noise_init = group['noise_init']
            noise_decay = group['noise_decay']
            amsgrad = group['amsgrad']
            if noise_init == 0:
                continue
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                if 'noise_buf' not in state:
                    noise = torch.tensor(noise_init, dtype=torch.float)
                    state['noise_buf'] = noise
                else:
                    noise = state['noise_buf']
                    noise.mul_(1 - noise_decay)
                d_p = p.grad.data
                eps = torch.sqrt(noise) * torch.randn_like(d_p)
                d_p.add_(eps)
        super().step()

        return loss

class NewtonsMethod:
    def __init__(self, params, lr=1.):
        self.params = params
        self.shapes = [p.shape for p in params]
        self.sizes = [p.view(-1).size(0) for p in params]
        self.lr = lr

    def step(self, loss):
        grad = self.compute_grad(loss)
        H = self.compute_hessian(grad)
        step, _ = torch.lstsq(H, grad.unsqueeze(1))
        step = step[:,0]
        param_steps = torch.split(step, self.sizes, 0)
        param_steps = [step.view(size) for step,size in zip(param_steps, self.shapes)]
        for p,pstep in zip(self.params, param_steps):
            p.data.add_(-self.lr, pstep)

    def compute_grad(self, loss):
        grad = torch.autograd.grad(loss, self.params, create_graph=True)
        grad = torch.cat([g.view(-1) for g in grad])
        return grad

    def compute_hessian(self, grad):
        H = []
        for dx in grad:
            d2x = torch.autograd.grad(dx, self.params, retain_graph=True)
            d2x = torch.cat([h.view(-1) for h in d2x])
            H.append(d2x)
        H = torch.stack(H)
        return H