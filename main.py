import memory_utils
memory_utils.setup_memory_logging()
import memory_utils
import logging
logger = logging.getLogger(__name__)
# Скрипт запуска: монтирование Drive в Colab, создание даталоадера, инициализация моделей и запуск обучения
from torch.utils.data import DataLoader
from device_utils import init_device

@memory_utils.log_memory_diff('Device Init')
def init_wrapper(): return init_device()
device, IS_XLA, xm, parallel_loader_factory, optimizer_step, save_checkpoint, is_master = init_device()

import torch

from data.dataset import SyntheticOceanDataset
from models.generator import Generator
from models.critic import Critic
from training.trainer import train_epoch


def optimizer_step_fn(opt):
    optimizer_step(opt, IS_XLA, xm)
    if IS_XLA:
        xm.mark_step()


def run_training(device="cuda", checkpoint_dir = "./checkpoints"):
    import sys, logging
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', stream=sys.stdout)
    logging.info('--- Logging Initialized in run_training ---')

    # params
    T = 4
    Z = 8
    H = 32
    W = 32
    C = 2

    per_device_batch = 1
    epochs = 1

    dataset = SyntheticOceanDataset(n_samples=200, T_window=T, Z=Z, H=H, W=W, channels=C)

    # DataLoader creation
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    G = Generator(T, Z, H, W, channels=C, base_dim=16).to(device)
    D = Critic(base_dim=16, T_window=T, Z=Z, H=H, W=W).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for epoch in range(1, epochs + 1):
        train_epoch(
            G, D, loader, optG, optD, device, epoch, n_critic=5, lambda_gp=10, checkpoint_dir=checkpoint_dir
        )


if __name__ == "__main__":
    run_training(device)
