# визуализация

import numpy as np
import matplotlib.pyplot as plt
import imageio


def plot_depth_slice(field_tensor, depth_idx=0, time_idx=0, quiver_sub=4, savepath=None):
    if hasattr(field_tensor, "detach"):
        arr = field_tensor.detach().cpu().numpy()
    else:
        arr = field_tensor
    # arr: (T,C,Z,H,W)
    u = arr[time_idx, 0, depth_idx]
    v = arr[time_idx, 1, depth_idx]
    plt.figure(figsize=(6, 5))
    plt.imshow(np.sqrt(u**2 + v**2), origin="lower")
    plt.colorbar(label="speed")
    X = np.arange(0, u.shape[1], quiver_sub)
    Y = np.arange(0, u.shape[0], quiver_sub)
    U = u[::quiver_sub, ::quiver_sub]
    V = v[::quiver_sub, ::quiver_sub]
    plt.quiver(X, Y, U.T, V.T, color="white", scale=5)
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()


def make_gif_sequence(frames, out_path, fps=4):
    imgs = []
    for f in frames:
        if isinstance(f, np.ndarray):
            imgs.append((255 * (f - f.min()) / (f.max() - f.min() + 1e-9)).astype(np.uint8))
    imageio.mimsave(out_path, imgs, fps=fps)
    print("Saved gif", out_path)
