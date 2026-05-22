# WGAN‑GP gradient penalty и вспомогательные лоссы

# training/losses.py
import torch
import torch.nn.functional as F
from torch.autograd import grad


def gradient_penalty(D, real, fake, mask, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, 1, 1, device=device)
    interp = alpha * real + (1 - alpha) * fake
    interp.requires_grad_(True)
    d_interp = D(interp, mask)
    grads = grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gp = ((grads.view(grads.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
    return gp


def reconstruction_loss(x_hat, real, mask):
    return F.l1_loss(x_hat * (1 - mask), real * (1 - mask))
