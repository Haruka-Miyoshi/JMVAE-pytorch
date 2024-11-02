import torch
from torch import nn

class TextDecoder(nn.Module):
    def __init__(self, w_dim:int, h_dim:int, z_dim:int):
        super(TextDecoder, self).__init__()
        self.w_dim = w_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.NN = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.w_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        xh = self.NN(z)
        return xh