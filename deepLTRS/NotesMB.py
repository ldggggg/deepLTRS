#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:09:46 2019

@author: marco
"""
import numpy as np
import torch
from torch import nn, optim
import os
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader  #, Sampler
from torch.autograd import Variable

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

############# Auxiliary Functions #############################################

def ipos(M, P, idx):
    return np.arange(idx*P, idx*P + P)

def jpos(M, P, idy):
    return np.arange(idy, idy + (M*P), P)

###############################################################################
import pandas as pd
#dat = np.load("sim_data_notes.npy")
#dat = dat[:25,:]

# loading notes
#dat = np.load("sim_data_notes.npy")
dat = pd.read_csv('universal_simu.txt')
dat = dat.iloc[:,:3]

# loading labels
clr = np.load('clr.npy')
clc = np.load('clc.npy')

M,P = 75, 60
############## loading and manipulatind docs and vocabulary  ###############



# CDataset: I am overwriting the methods __init__, __getitem__ and __len__,
# no other names allowed
class MDataset(Dataset):
    # constructor
    def __init__(self, dat, transform=False):
        self.dat = dat
        self.transform = transform
        self.L = len(dat)
        
    def __getitem__(self, item):
        dat = self.dat.iloc[item,:]
        if self.transform is not False:
            dat = torch.Tensor(dat)
        return dat
    
    def __len__(self):
        return self.L

train_size = int(0.8*len(dat))
pos = np.random.choice(np.arange(len(dat)), train_size, replace = False)

train_df = dat.iloc[pos,:]

X = dat.copy()
test_df = X.drop(index=pos)
# test_df = dat[-pos,:] ??
## row and column data
        
    
Tdata = MDataset(train_df, transform = True)
test_data = MDataset(test_df, transform = True)

## Loading both

# prop = 0.4
batch_size = 100
Tdload = DataLoader(Tdata,
                    batch_size = batch_size, 
                    shuffle = True)  

## *************************************************
Y = np.zeros(shape = (M,P))
for line in dat.to_numpy() :
    Y[int(line[0]), int(line[1])] = int(line[2])

# Global parameters
init_dim_R = 60
init_dim_C = 75
mid_dim = 50
mid_dim_out = 50
int_dim = 2
# prior_variance = 0.995
epochs =1000
mc_iter = 1
#

# variable status
#status = 'row'
# Encoding class 
class Encoder(nn.Module):
    def __init__(self, type = 'R'):
        super(Encoder, self).__init__()
        
        if type == 'R':
            init_dim = init_dim_R
        elif type == 'C':
            init_dim = init_dim_C
        else:
            raise ValueError('Wrong type assigment.')
        ## 1. inference network
        # working with rows
        self.en1 = nn.Linear(init_dim, mid_dim)
        #self.int = nn.Linear(mid_dim, 1)               # one coeff. for each row
        self.mu = nn.Linear(mid_dim, int_dim)
        self.logv = nn.Linear(mid_dim, int_dim)
        
    
    def encode(self, x):
        h1 = F.relu(self.en1(x))
        mu = self.mu(h1)
        logv =self.logv(h1)
        #intercept = self.int(h1)
        return mu, logv
        
    def forward(self, x):
        return self.encode(x)

# Decoding NN, containing two Encoders
class Decoder(nn.Module):
    def __init__(self):
          super(Decoder, self).__init__()           
          self.E1 = Encoder('R')
          self.E2 = Encoder('C')
          self.D1 = nn.Linear(batch_size, mid_dim_out)
          self.D2 = nn.Linear(mid_dim_out, batch_size)
          self.log_vareps = nn.Parameter(torch.randn(1))
          #self.log_vareps = torch.tensor(-6.90, requires_grad = False)
          
    def reparametrize(self, m, log_v):
        # expanding the mean tensor
        # ta, tb = m.shape
        # m = m.unsqueeze(-1)
        # m = m.expand(ta, tb, mc_iter)
        # expanding the log_v tensor
        # log_v = log_v.unsqueeze(-1)
        # log_v = log_v.expand(ta, tb, mc_iter)
        # sampling
        std = torch.exp(0.5*log_v)
        eps = torch.randn_like(std)
        return m + eps*std
    
    def forward(self, x, y):
        muV, logv_V = self.E1(x)
        muW, logv_W = self.E2(y.transpose(0, 1))
        V = self.reparametrize(muV, logv_V)
        W = self.reparametrize(muW, logv_W)
        # 
        val = torch.mm(V, W.transpose(0,1)) 
        out = self.D2(F.relu(self.D1(val.diag()))) ### diagonal
        return out, muV, logv_V, muW, logv_W, self.log_vareps      
        
             
# loss function
def lossf(target, out, muV, muW, logv_V, logv_W, log_vareps):
    
    SV = torch.exp(logv_V)
    SW = torch.exp(logv_W)
    # main loss component
    vareps = torch.exp(log_vareps)
    
    MC = 0.5/vareps*(target-out)*(target-out)
    MC = torch.sum(MC + 0.5*log_vareps.expand_as(target))

    # ** computing the first KL divergence
    m1 = torch.Tensor(1, int_dim).fill_(0.0) # Nota: requires_grad is set to False by default ==> No optimization with respect to the prior parameters
    v1 = torch.Tensor(1, int_dim).fill_(1.0)
    log_v1 = torch.log(v1)
    
    scale_factor1 = torch.ones(1, int_dim)
    
    # due to the batch size we need the prior and the posterior pmts to have the same dims
    m1 = m1.expand_as(muV)
    v1 = v1.expand_as(SV)
    log_v1 = log_v1.expand_as(logv_V)
    scale_factor1=scale_factor1.expand_as(muV)
    
    var_division1    = SV / v1
    diff1            = muV - m1
    diff_term1       = diff1 * diff1 / v1
    logvar_division1 = log_v1 - logv_V
            
    KLv = 0.5 * ( torch.sum(var_division1 + diff_term1 + logvar_division1 - scale_factor1))
 
    # ** computing the second KL divergence
    m2 = torch.Tensor(1, int_dim).fill_(0.0) # Nota: requires_grad is set to False by default ==> No optimization with respect to the prior parameters
    v2 = torch.Tensor(1, int_dim).fill_(1.0)
    log_v2 = torch.log(v2)
    
    scale_factor2 = torch.ones(1, int_dim)
    
    # due to the batch size we need the prior and the posterior pmts to have the same dims
    m2 = m2.expand_as(muW)
    v2 = v2.expand_as(SW)
    log_v2 = log_v2.expand_as(logv_W)
    scale_factor2=scale_factor2.expand_as(muW)
        
    var_division2    = SW / v2
    diff2            = muW - m2
    diff_term2       = diff2 * diff2 / v2
    logvar_division2 = log_v2 - logv_W
            
    KLw = 0.5 * ( torch.sum(var_division2 + diff_term2 + logvar_division2 - scale_factor2))
        
    return  MC + KLv + KLw   
   
def vlossf(targetN, outN):
        MCN = (targetN-outN)*(targetN-outN)
        MCN = torch.sum(MCN)
        return torch.sqrt(MCN/len(targetN))


# the model
Mod = Decoder()
#optimizer = optim.SGD(Mod.parameters(), lr=2e-9)
optimizer = optim.Adam(list(Mod.parameters())+list(Mod.E1.parameters())+list(Mod.E2.parameters()), lr=2e-3, betas=(0.99, 0.999))

Y_ = torch.Tensor(Y)

def train(epoch):   
    Mod.train()
    train_loss = 0
    for batch_idx, obs in enumerate(Tdload):
        optimizer.zero_grad()
        loc_i = obs[:,0].type(torch.long)
        loc_j = obs[:,1].type(torch.long)
        x = Y_[loc_i,:]
        y = Y_[:,loc_j]
        out, muV, logv_V, muW, logv_W, log_vareps = Mod.forward(x,y)
        stock_muV[loc_i,:] = muV.detach().numpy()
        stock_muW[loc_j,:] = muW.detach().numpy()
        #if epoch == 0:
         # print(" Initial variance: {}".format(torch.exp(log_vareps)))
        loss = lossf(Y_[loc_i, loc_j], out, muV, muW, logv_V, logv_W, log_vareps)
        loss.backward()
        optimizer.step()
        train_loss += loss
        # Mod.eval()
        # validation RMSE loss
        # vloss = vlossf(torch.FloatTensor(dat[val_pos[0], val_pos[1]]), out[val_pos[0], val_pos[1]])
        # loss = test_loss(inA, muV, muW)
        if epoch % 100 == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)))
        return train_loss    
            # print('Validation RMSE {:.6f}'.format(vloss))

   
import matplotlib.pyplot as plt    
stock_muV = np.zeros(shape = (M,int_dim))
stock_muW = np.zeros(shape = (P, int_dim))
total_loss = np.zeros(epochs)
for epoch in range(epochs):
    total_loss[epoch] = train(epoch)
    if epoch % 100 ==0:
        f, ax = plt.subplots(1)
        ax.scatter(stock_muV[:,0], stock_muV[:,1], color = 'red')
        ax.scatter(stock_muW[:,0], stock_muW[:,1], color = 'blue')    
    

import matplotlib.pyplot as plt
plt.plot(total_loss)    
    

# # reconstructing the whole matrix
# rY = np.zeros((M,P))
# Mod.eval()
# for i in range(M):
#     x = Y_[i,:].reshape(1,-1)
#     for j in range(P):
#         y = Y_[:,j].reshape(1,-1)
#         out, muV, logv_V, muW, logv_W, log_vareps = Mod.forward(x,y)
#         rY[i,j] = out.detach().numpy()
        

# from sklearn.metrics import adjusted_rand_score as ari
# from sklearn.cluster import KMeans

# eclr = KMeans(n_clusters = 2).fit(rY)
# eclc = KMeans(n_clusters = 2).fit(rY.T)

# print(ari(clr, eclr.labels_))
# print(ari(eclc.labels_, clc))        


