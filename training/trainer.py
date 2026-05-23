
import torch
import gc
import logging
import os
import psutil
from .losses import gradient_penalty, reconstruction_loss

logger = logging.getLogger(__name__)

def get_mem_mb():
    """Returns current process Resident Set Size (RSS) in Megabytes for OOM tracking."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def train_epoch(G, D, loader, optG, optD, device, epoch, n_critic=5, lambda_gp=10, checkpoint_dir=None):
    """Executes one training epoch with detailed memory logging and aggressive garbage collection."""
    G.train()
    D.train()
    
    for i, batch in enumerate(loader):
        real = batch['field'].to(device)
        mask = batch['mask'].to(device)
        
        if i == 0:
            logging.info(f'[TRAIN] Epoch {epoch} | Batch 0 start | RSS: {get_mem_mb():.2f} MB')

        # --- Critic Update Loop (WGAN-GP style) ---
        for c_idx in range(n_critic):
            optD.zero_grad()
            
            # Sample noise and generate fake data
            z = torch.randn_like(real, device=device)
            fake = G(real * mask, mask, z)
            x_hat = (mask * real + (1 - mask) * fake).detach()
            
            if i == 0 and c_idx == 0:
                logging.debug(f'[TRAIN] Pre-Critic Forward | RSS: {get_mem_mb():.2f} MB')
            
            # Score real and fake samples
            d_real = D(real, mask)
            d_fake = D(x_hat, mask)
            gp = gradient_penalty(D, real, x_hat, mask, device)
            
            # Wasserstein loss with Gradient Penalty
            loss_D = -(d_real.mean() - d_fake.mean()) + lambda_gp * gp
            loss_D.backward()
            optD.step()

            # Cleanup local tensors to prevent memory accumulation
            del z, fake, x_hat, d_real, d_fake, gp, loss_D
            if i == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # --- Generator Update ---
        optG.zero_grad()
        z_g = torch.randn_like(real, device=device)
        fake_g = G(real * mask, mask, z_g)
        
        # Combined Adversarial and Reconstruction Loss
        loss_G = -D(fake_g, mask).mean() + 10.0 * reconstruction_loss(fake_g, real, mask)
        loss_G.backward()
        optG.step()

        if i == 0:
            logging.info(f'[TRAIN] Batch 0 iteration complete | RSS: {get_mem_mb():.2f} MB')

        # Final cleanup for the batch
        del real, mask, z_g, fake_g, loss_G
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
