import torch
from torch import nn
from torch.nn import functional as F

import os
from .model import *

class JMVAE(object):
    def __init__(self, x_dim:int, w_dim:int, h_dim:int, z_dim:int, lr:float=3e-4, train_mode:bool=True, save_path="param", model_path="jmvae_mlda_parameter.path"):
        self.x_dim = x_dim # 入力画像の次元数
        self.w_dim = w_dim # 入力単語の次元数
        self.h_dim = h_dim # 隠れ変数次元数
        self.z_dim = z_dim # 潜在変数次元数

        self.save_path = save_path # パラメータ保存先
        self.model_path = model_path # パラメータファイル名
        self.path = os.path.join(self.save_path, self.model_path) # パス生成

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        self.mode = train_mode # 学習モード

        self.model = Model(self.x_dim, self.w_dim, self.h_dim, self.z_dim).to(device=self.device)

        if not os.path.exists(save_path):
                os.mkdir(self.save_path)

        if not self.mode:
            # パラメータファイルがない場合における処理を追加
            try:
                self.model.load_state_dict(torch.load(self.path))
            except:
                raise("Not Found model paramter file!!")

        self.lr = lr
        self.optim = torch.optim.Adam([{'params':self.model.parameters()}], lr=self.lr)
        self.losses = []
        
    def tensor(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device)
    
    def numpy(self, x):
        if self.device == "cpu":
            return x.detach().numpy()
        else:
            return x.detach().cpu().numpy()
        
    def BCE(self, xh, x):
        loss = F.binary_cross_entropy(xh, x, reduction='sum')
        return loss
    
    def KLD(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss

    def train(self, train_loader, epoch=30):
        self.model.train() # 学習モード
        for e in range(1, epoch + 1):
            total_loss = 0.0
            for i, (batch_data, label) in enumerate(train_loader):
                self.optim.zero_grad()
                x = batch_data.view(batch_data.size(0), self.x_dim).to(device=self.device)
                w = F.one_hot(label, self.w_dim).float().to(device=self.device)
                mu, logvar, z, xh, wh = self.model(x, w, True)
                loss = ( self.KLD(mu, logvar) + self.BCE(xh, x) + self.BCE(wh, w) ).mean()
                loss.backward()
                total_loss += loss.item()
                self.optim.step()
            total_loss /= i
            self.losses.append(total_loss)
            print(f"epoch:{e}, loss:{total_loss}")

        if self.mode:
            torch.save(self.model.state_dict(), self.path)
    
    def word_to_bow(self, word):
        w = F.one_hot(word, self.w_dim).float() 
        return w

    def xw_to_z_to_xhwh(self, x, w):
        self.model.eval() # 推論モード        
        with torch.no_grad():
            mu, logvar, z, xh, wh = self.model(x, w, False)
        return mu, logvar, z, xh, wh
    
    def xw_to_theta(self, x, w):
        self.model.eval()
        with torch.no_grad():
            mu, logvar, z = self.model.encode(x, w, False)
        return mu, logvar, z
    
    def theta_to_xw(self, theta):
        self.model.eval()
        with torch.no_grad():
            xh, wh = self.model.decode(theta)
        return xh, wh