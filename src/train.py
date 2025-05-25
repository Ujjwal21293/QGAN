import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from discriminator import get_discriminator
from generator import get_real_space_samples, initialize_params

import os
import matplotlib.pyplot as plt

# configuration
n_qubits = 2
shots = 1000
n_epochs = 250
batch_size = 64
lr_gen = 0.1
lr_disc = 0.001
save_path = "QGANS/data/generated_samples_zne.npy"

# loading real data
real_data = np.load("QGANS/data/real_distribution.npy")
real_data = torch.tensor(real_data, dtype=torch.float32)

# initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator = get_discriminator(device)
criterion = nn.BCELoss()
opt_disc = optim.Adam(discriminator.parameters(), lr=lr_disc)

# quantum generator parameters
theta = torch.tensor(initialize_params(), requires_grad=True, dtype=torch.float32)
opt_gen = optim.Adam([theta], lr=lr_gen)


def sample_real_data(batch_size):
    idx = np.random.choice(real_data.shape[0], batch_size, replace=False)
    return real_data[idx]


def train():
    d_losses = []
    g_losses = []

    for epoch in range(n_epochs):
        # train discriminator
        discriminator.train()
        real_batch = sample_real_data(batch_size).to(device)
        real_labels = torch.ones((batch_size, 1)).to(device)

        # generate fake samples from quantum circuit
        gen_data = get_real_space_samples(theta.detach().numpy(), shots=batch_size)
        fake_batch = torch.tensor(gen_data, dtype=torch.float32).to(device)
        fake_labels = torch.zeros((batch_size, 1)).to(device)

        # BCE Loss
        d_real = discriminator(real_batch)
        d_fake = discriminator(fake_batch)
        d_loss_real = criterion(d_real, real_labels)
        d_loss_fake = criterion(d_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        opt_disc.zero_grad()
        d_loss.backward()
        opt_disc.step()

        # train generator
        def generator_loss():
            fake_samples = get_real_space_samples(theta.detach().numpy(), shots=batch_size)
            x_fake = torch.tensor(fake_samples, dtype=torch.float32).to(device)
            y_pred = discriminator(x_fake)
            return criterion(y_pred, torch.ones((batch_size, 1)).to(device))  

        # loss
        g_loss = generator_loss()

        opt_gen.zero_grad()
        g_loss.backward()
        opt_gen.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"[Epoch {epoch:03d}] D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

    # save generated samples
    gen_final = get_real_space_samples(theta.detach().numpy(), shots=1000)
    np.save(save_path, gen_final)

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("QGAN Training Losses")
    plt.savefig("QGANS/results/training_curves_zne.png")
    plt.show()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    train()
