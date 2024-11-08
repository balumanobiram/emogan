# training/generator.py

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 48 * 48 * 3)
        self.batch_norm = nn.BatchNorm1d(256)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = self.batch_norm(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = x.view(-1, 3, 48, 48)  # Reshape to image size
        return x
