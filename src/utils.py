import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_samples(real_data, fake_data, save_path="results/generated_vs_real.png", show=True):
    plt.figure(figsize=(6, 6))
    plt.scatter(real_data[:, 0], real_data[:, 1], c='blue', alpha=0.4, label='Real')
    plt.scatter(fake_data[:, 0], fake_data[:, 1], c='red', alpha=0.4, label='Generated')
    plt.title("Real vs Generated Samples")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(save_path)
    if show:
        plt.show()


def to_tensor(x, device="cpu"):
    return torch.tensor(x, dtype=torch.float32).to(device)


def load_real_data(path="data/real_distribution.npy"):
    return np.load(path)


def save_generated_samples(samples, path="data/generated_samples.npy"):
    np.save(path, samples)


def log_loss(epoch, d_loss, g_loss):
    print(f"[Epoch {epoch:03d}] D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f}")
