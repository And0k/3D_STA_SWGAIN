
import torch
import gc
import logging
import os
import psutil
from tqdm.auto import tqdm
from .losses import gradient_penalty, reconstruction_loss

def get_mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def train_epoch(G, D, loader, optG, optD, device, epoch, n_critic=5, lambda_gp=10, checkpoint_dir=None):
    G.train(); D.train()
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}')
    
    for i, batch in pbar:
        real = batch['field'].to(device)
        mask = batch['mask'].to(device)
        
        # FIX: Initialize loss_D in batch scope to prevent UnboundLocalError
        loss_D = torch.tensor(0.0, device=device)

        for c_idx in range(n_critic):
            optD.zero_grad()
            z = torch.randn_like(real, device=device)
            fake = G(real * mask, mask, z)
            x_hat = (mask * real + (1 - mask) * fake).detach()
            
            d_real = D(real, mask)
            d_fake = D(x_hat, mask)
            gp = gradient_penalty(D, real, x_hat, mask, device)
            
            loss_D = -(d_real.mean() - d_fake.mean()) + lambda_gp * gp
            loss_D.backward()
            optD.step()
            
            del z, fake, x_hat, d_real, d_fake, gp

        optG.zero_grad()
        z_g = torch.randn_like(real, device=device)
        fake_g = G(real * mask, mask, z_g)
        loss_G = -D(fake_g, mask).mean() + 10.0 * reconstruction_loss(fake_g, real, mask)
        loss_G.backward()
        optG.step()

        pbar.set_postfix({
            'D_loss': f'{loss_D.item():.4f}',
            'G_loss': f'{loss_G.item():.4f}',
            'RSS': f'{get_mem_mb():.0f}MB'
        })

        del real, mask, z_g, fake_g, loss_G
        if i % 2 == 0:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
