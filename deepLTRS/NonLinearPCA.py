#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:20:07 2019

@author: marco
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# some nice colors
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
pal = sns.color_palette(flatui)

#############################################
## Low and High dimensional simulated data ##
#############################################

np.random.seed(0)
N = 10000
D = 25
from sklearn.datasets import make_moons
ld_dat, Z = make_moons(n_samples = 10000, noise = .05)
col = [pal[idx] for idx in Z]
plt.scatter(ld_dat[:,0], ld_dat[:,1], color = col)

# non-linear projecting in higher dimension (via a neural network!)
A = np.random.normal(size=(D,2))
b = np.random.normal(size = (1,D))
C = np.random.normal(size=(D,D))
d = np.random.normal(size=(1,D))

hd_dat = np.dot(np.tanh(np.dot(ld_dat, A.T) + b), C.T)+ d


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
res = pca.fit_transform(hd_dat)

plt.figure()
plt.plot(res[:,0], res[:,1], 'ro', color = pal[2])




#########
## VAE ##
#########

import torch
from torch.utils.data import Dataset, DataLoader  
from torch.nn import functional as F
from torch import nn, optim
from torch.autograd import Variable


device = "cpu"

# manually splitting the dataset into train and validation
pos = np.random.choice(range(len(hd_dat)), int(0.8*N), replace = False)
not_pos = []
for idx in range(len(hd_dat)):
    if not idx in pos:
        not_pos.append(idx)
train_df = hd_dat[pos, :]
valid_df = hd_dat[not_pos,:]
train_Z = Z[pos]
valid_Z = Z[not_pos]

# CDataset: I am overwriting the methods __init__, __getitem__ and __len__,
class MDataset(Dataset):
    # constructor
    def __init__(self, dat, transform=False):
        self.dat = dat
        self.transform = transform
        self.L = self.dat.shape[0]
        
    def __getitem__(self, item):
        dat = self.dat[item,:]
        if self.transform is not False:
            dat = torch.Tensor(dat)
            if device=="cuda":
                dat = dat.to(device)
        return dat
    
    def __len__(self):
        return self.L

C_train_df = MDataset(train_df, transform = True)    
C_valid_df = MDataset(valid_df, transform = True)

Tdload = DataLoader(C_train_df,
                    batch_size = 125,
                    shuffle = False
                    )
Vdload = DataLoader(C_valid_df,
                    batch_size = 2000,
                    shuffle = False
                    )

# Global parameters
init_dim = D
mid_dim_1 = 10*D
mid_dim_2 = 10*D
int_dim = 2
epochs = 80


# Encoding class 
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        ## inference network
        self.en1 = nn.Linear(init_dim, mid_dim_1)
        self.en2 = nn.Linear(mid_dim_1, mid_dim_2)
        self.mu = nn.Linear(mid_dim_2, int_dim)
        self.logv = nn.Linear(mid_dim_2, int_dim)
       
    def encode(self, x):
        h1 = F.relu(self.en2(F.relu(self.en1(x))))
        mu = self.mu(h1)
        logv = self.logv(h1)
        return mu, logv
        
    def forward(self, x):
        return self.encode(x)
    
 # Decoding class
class Decoder(nn.Module):
    def __init__(self):
          super(Decoder, self).__init__()           
          self.E = Encoder()
          self.D1 = nn.Linear(int_dim, mid_dim_2)
          self.D2 = nn.Linear(mid_dim_2, mid_dim_1)
          self.D3= nn.Linear(mid_dim_1, D)
          self.log_vareps = nn.Parameter(torch.randn(1))
          
    def reparametrize(self, mu, logv):
          std = torch.exp(0.5*logv)
          eps = torch.randn_like(std)
          if device == 'cuda':
              eps = eps.to(device)
          return mu + eps*std
    
    def decode(self, z):
        out = self.D3(F.relu(self.D2(F.relu(self.D1(z)))))
        return out
        
    def forward(self, x):
        mu, logv= self.E(x)
        z = self.reparametrize(mu, logv)
        out = self.decode(z)
        return out, mu, logv, self.log_vareps   
         
# loss function
def lossf(hd_dat, out, mu, logv, log_vareps):
    
        s = torch.exp(logv)
        
        # ** main loss component  (- Gaussian log-likelihood!)
        vareps = torch.exp(log_vareps)
        if device == "cuda":
            vareps = vareps.to("cuda")        
        MC = (0.5/vareps)*(hd_dat-out)*(hd_dat-out)
        MC = torch.sum(MC + 0.5*log_vareps.expand_as(hd_dat))

        # ** computing the KL divergence
        m1 = torch.Tensor(1, int_dim).fill_(0.0) # Nota: requires_grad is set to False by default ==> No optimization with respect to the prior parameters
        v1 = torch.Tensor(1, int_dim).fill_(1.0)
        if device=="cuda":
            m1 = m1.to(device)
            v1 = v1.to(device)
        log_v1 = torch.log(v1)
        
        scale_factor1 = torch.ones(1, int_dim)
        if device=="cuda":
            scale_factor1 = scale_factor1.to(device)
        
        # due to the batch size we need the prior and the posterior pmts to have the same dims
        m1 = m1.expand_as(mu)
        v1 = v1.expand_as(s)
        log_v1 = log_v1.expand_as(logv)
        scale_factor1=scale_factor1.expand_as(mu)
        
        var_division1    = s / v1
        diff1            = mu - m1
        diff_term1       = diff1 * diff1 / v1
        logvar_division1 = log_v1 - logv
                
        KLv = 0.5 * ( torch.sum(var_division1 + diff_term1 + logvar_division1 - scale_factor1))
        
        return  (MC + KLv)
    
## a validation loss on validation (No KL div)    
def o_vlossf(target, out):
        MC = (target-out)*(target-out)
        MC = torch.sum(MC)
        return torch.sqrt(MC/len(target))    

## The model and the optizer
Mod = Decoder()
if device=="cuda":
    Mod = Mod.cuda()
optimizer = optim.Adam(Mod.parameters(), lr=2e-3, betas=(0.99, 0.999))

def train(epoch):
    Mod.train()
    train_loss = 0
    for batch_idx, obs in enumerate(Tdload):
        optimizer.zero_grad()
        x = Variable(obs)
        out, mu, logv, log_vareps = Mod.forward(x) # Mod(x)
        loss = lossf(x.view(-1, init_dim), out, mu, logv, log_vareps)
        # computing the gradient
        loss.backward()
        train_loss += torch.Tensor.item(loss)
        # gradient step to update parameters
        optimizer.step()
#        if batch_idx % 100 == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch_idx * len(x), len(Tdload.dataset),
#                100. * batch_idx / len(Tdload.dataset),
#                torch.Tensor.item(loss) / len(x)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(Tdload.dataset)))
    Mod.eval()
    it = iter(Vdload)
    ix = it.next()
    out, mu, logv, log_vareps = Mod.forward(ix)
    vloss = lossf(ix.view(-1, init_dim), out, mu, logv, log_vareps)
    #vloss = o_vlossf(ix,out)
    print('====> Validation loss: {}'.format(vloss / len(Vdload.dataset)))
    return train_loss, vloss

vloss = np.zeros(epochs)    
tloss = np.zeros(epochs)
for epoch in range(epochs):
    tloss[epoch], vloss[epoch] = train(epoch)
 
plt.figure("Losses")    
plt.plot(tloss/len(Tdload.dataset), color = pal[4], label = 'train')
plt.plot(vloss/len(Vdload.dataset), color = pal[5], label = 'validation')
plt.legend(loc = "upper right")

Mod.eval()
it = iter(Vdload)
ix = it.next()
out, mu, logv, log_vareps = Mod.forward(ix)
nout = out.cpu().data.numpy()
nmu = mu.data.numpy()

plt.figure()
plt.plot(nmu[:,0], nmu[:,1], 'ro' ,color = pal[4])

