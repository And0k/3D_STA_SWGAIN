# device_utils.py
import torch


def init_device():
    """
    Detect device once.
    Returns (device, IS_XLA, xm, parallel_loader_factory, optimizer_step, save_fn, is_master).
    """
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl

        IS_XLA = True
        device = xm.xla_device()

        def parallel_loader_factory(loader, device):
            return pl.ParallelLoader(loader, [device]).per_device_loader(device)

        def optimizer_step(opt):
            xm.optimizer_step(opt, barrier=True)
            xm.mark_step()

        def save_fn(state, path):
            if xm.is_master_ordinal():
                xm.save(state, path)

        def is_master():
            return xm.is_master_ordinal()

    except Exception:
        # fallback CPU/GPU
        IS_XLA = False
        xm = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def parallel_loader_factory(loader, device):
            # identity: return the loader itself for CPU/GPU
            return loader

        def optimizer_step(opt):
            opt.step()

        def save_fn(state, path):
            torch.save(state, path)

        def is_master():
            return True

    return device, IS_XLA, xm, parallel_loader_factory, optimizer_step, save_fn, is_master
