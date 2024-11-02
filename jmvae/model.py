import torch
from torch import nn

from .encoder import Encoder
from .image_decoder import ImageDecoder
from .text_decoder import TextDecoder

class Model(nn.Module):
    def __init__(self, x_dim:int, w_dim:int, h_dim:int, z_dim:int):
        super(Model, self).__init__()
        self.x_dim = x_dim
        self.w_dim = w_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.xw_to_z = Encoder(self.x_dim + self.w_dim, self.h_dim, self.z_dim)        
        self.z_to_x = ImageDecoder(self.x_dim, self.h_dim, self.z_dim)
        self.z_to_w = TextDecoder(self.w_dim, self.h_dim, self.z_dim)
    
    def reparameterize(self, mu, logvar, mode):
        if mode:
            s = torch.exp(0.5 * logvar)
            e = torch.rand_like(s)
            return e.mul(s).add_(mu)
        else:
            return mu
    
    def encode(self, x, w, mode):
        xw = torch.cat((x, w), dim=1)
        mu, logvar = self.xw_to_z(xw)
        z = self.reparameterize(mu, logvar, mode)
        return mu, logvar, z
    
    def decode(self, theta):
        xh = self.z_to_x(theta)
        wh = self.z_to_w(theta)
        return xh, wh

    def forward(self, x, w, mode):
        xw = torch.cat((x, w), dim=1)
        mu, logvar = self.xw_to_z(xw)
        z = self.reparameterize(mu, logvar, mode)
        xh = self.z_to_x(z)
        wh = self.z_to_w(z)
        return mu, logvar, z, xh, wh