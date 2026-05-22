# Организация цикла обучения с n_critic, чекпоинтами, multiple sampling

import os, torch
from .losses import gradient_penalty, reconstruction_loss


def train_epoch(G, D, loader, optG, optD, device, epoch, n_critic=5, lambda_gp=10, checkpoint_dir=None):
    G.train()
    D.train()
    for i, batch in enumerate(loader):
        real = batch["field"].to(device)  # (B,T,C,Z,H,W)
        mask = batch["mask"].to(device)  # (B,T,1,Z,H,W)
        B = real.shape[0]

        # Train critic n_critic times
        for _ in range(n_critic):
            z = torch.randn_like(real, device=device)
            observed = real * mask
            fake = G(observed, mask, z)
            x_hat = mask * real + (1 - mask) * fake

            d_real = D(real, mask)
            d_fake = D(x_hat.detach(), mask)
            gp = gradient_penalty(D, real, x_hat.detach(), mask, device)
            loss_D = -(d_real.mean() - d_fake.mean()) + lambda_gp * gp

            optD.zero_grad()
            loss_D.backward()
            optD.step()

        # Train generator
        z = torch.randn_like(real, device=device)
        observed = real * mask
        fake = G(observed, mask, z)
        x_hat = mask * real + (1 - mask) * fake

        d_fake = D(x_hat, mask)
        adv_loss = -d_fake.mean()
        rec_loss = reconstruction_loss(x_hat, real, mask)
        loss_G = adv_loss + 10.0 * rec_loss

        optG.zero_grad()
        loss_G.backward()
        optG.step()

        if i % 50 == 0:
            print(
                f"Epoch {epoch} Step {i} | D {loss_D.item():.4f} | G {loss_G.item():.4f} | rec {rec_loss.item():.4f}"
            )

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {"G": G.state_dict(), "D": D.state_dict()}, os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch}.pt")
        )
        print("Saved checkpoint", os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch}.pt"))
