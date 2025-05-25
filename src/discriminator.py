import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16):
        """MLP including leaky relu and signmoid output"""
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = torch.sigmoid(self.out(x))
        return x


def get_discriminator(device="cpu"):
    """loads the model"""
    model = Discriminator()
    return model.to(device)
