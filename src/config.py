import torch

# quantum GAN configuration
n_qubits = 2
shots = 1000

# training
n_epochs = 500
batch_size = 64
lr_gen = 0.1
lr_disc = 0.001
seed = 42

# files
real_data_path = "data/real_distribution.npy"
generated_data_path = "data/generated_samples.npy"
plot_path = "results/generated_vs_real.png"
loss_plot_path = "results/training_curves.png"

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
