import torch
from torch import nn

class ImageDecoder(nn.Module):
    def __init__(self, x_dim:int, h_dim:int, z_dim:int):
        super(ImageDecoder, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.NN = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.x_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        xh = self.NN(z)
        return xh