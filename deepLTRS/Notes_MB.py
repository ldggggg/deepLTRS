#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:09:46 2019
@author: marco
"""
import numpy as np
from random import sample, shuffle
import torch
from torch import nn, optim
import os
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader  #, Sampler
from torch.autograd import Variable
import time

#os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0')

############# Auxiliary Functions #############################################

def ipos(M, P, idx):
    return np.arange(idx*P, idx*P + P)

def jpos(M, P, idy):
    return np.arange(idy, idy + (M*P), P)

###############################################################################

#dat = np.load("sim_data_notes.npy")
#dat = dat[:25,:]

# loading notes
dat = np.load("sim_data_notes_750_600.npy")

# loading labels
clr = np.load('clr_750_600.npy')
clc = np.load('clc_750_600.npy')

M,P = dat.shape

############## loading and manipulatind docs and vocabulary  ###############
X = dat.copy()

## inserting missing values
val = 0.0
import random
# we are storing the positions of validation and test entries!
ipv = jpv = ipt = jpt = np.zeros(shape = (1,1))
# random.seed(0)
# np.random.seed(0)
delete = list()
delete_col = list()
ix = [(row, col) for row in range(X.shape[0]) for col in range(X.shape[1])]
for row, col in random.sample(ix, int(round(0.99*len(ix)))):
    X[row, col] = 0
    delete.append(row*P+col)
    delete_col.append(col*M+row)   
    if np.random.uniform() < 1/1000:    # validation
        ipv = np.vstack((ipv, row))
        jpv = np.vstack((jpv, col))
        X[row, col] = 0
        # delete.append(row*P+col)
        # delete_col.append(col*M+row)
    elif np.random.uniform() > 999/1000:    # test
        ipt = np.vstack((ipt, row))
        jpt = np.vstack((jpt, col))
        X[row, col] = 0 
        # delete.append(row*P+col)
        # delete_col.append(col*M+row)
ipv = ipv[1:,0].astype('int32')
jpv = jpv[1:,0].astype('int32')
ipt = ipt[1:,0].astype('int32')
jpt = jpt[1:,0].astype('int32')

# saving positions of observed values
store = np.where(X != 0)
# validation and test positions
val_pos = (ipv, jpv)
test_pos = (ipt, jpt)
print(len(store[0]), len(val_pos[0]), len(test_pos[0]))

############# mean of users and products ###############    
mean_u = np.zeros((M,1))
for u in range(M):
    nz = len(np.where(dat[u,:]!=0)[0])
    mean_u[u,:] = dat[u,:].sum()/float(nz*2)
if device=="cuda":
    mean_u = mean_u.to(device)
    
mean_p = np.zeros((P,1))
for p in range(P):
    nz = len(np.where(dat.T[p,:]!=0)[0])
    mean_p[p,:] = dat.T[p,:].sum()/float(nz*2)
if device=="cuda":
    mean_p = mean_p.to(device)     
    
###### mean fixed #######
b_u = torch.Tensor(mean_u)
b_p = torch.Tensor(mean_p) 

####### Identity vector ########
bu_i = torch.Tensor(M, 1).fill_(1.0)
bp_j = torch.Tensor(P, 1).fill_(1.0)  

# CDataset: I am overwriting the methods __init__, __getitem__ and __len__,
# no other names allowed
class MDataset(Dataset):
    # constructor
    def __init__(self, dat, transform=False):
        self.dat = dat
        self.transform = transform
        self.L = len(dat)
        
    def __getitem__(self, item):
        dat = self.dat[item,:]
        if self.transform is not False:
            dat = torch.Tensor(dat)
            if device=="cuda":
                dat = dat.to(device) 
        return dat
    
    def __len__(self):
        return self.L

## row and column data
Rdata = MDataset(X, transform = True)
Cdata = MDataset(X.transpose(), transform = True)

batch_size_R = 5
batch_size_C = 3

## Loading both
Rdload = DataLoader(Rdata,
                    batch_size = batch_size_R, 
                    shuffle = True)  

Cdload = DataLoader(Cdata,
                    batch_size = batch_size_C,
                    shuffle = True
                    )

# Global parameters
init_dim_R = P
init_dim_C = M
mid_dim = 50
mid_dim_out = 80
int_dim = 50
# prior_variance = 0.5
epochs = 50

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
        self.en1 = nn.Linear(init_dim, mid_dim)
        self.en2 = nn.Linear(mid_dim, mid_dim)
        self.en2_drop = nn.Dropout(0.2)        
        self.mu = nn.Linear(mid_dim, int_dim)
        self.mu_bn  = nn.BatchNorm1d(int_dim) 
        self.logv = nn.Linear(mid_dim, int_dim)
        self.logv_bn  = nn.BatchNorm1d(int_dim) 
        
    
    def encode(self, x):
        h1 = F.relu(self.en1(x))
        h2 = self.en2_drop(F.softplus(self.en2(h1)))
        mu = self.mu_bn(self.mu(h2))
        logv =self.logv_bn(self.logv(h2))
        return mu, logv
        
    def forward(self, x):
        return self.encode(x)

# Decoding NN, containing two Encoders
class Decoder(nn.Module):
    def __init__(self):
          super(Decoder, self).__init__()           
          self.E1 = Encoder('R')
          self.E2 = Encoder('C')
          self.D1 = nn.Linear(init_dim_R, mid_dim_out)
          self.D2 = nn.Linear(mid_dim_out, init_dim_R)
          self.log_vareps = nn.Parameter(torch.randn(1))
          
    def reparametrize(self, m, log_v):
        std = torch.exp(0.5*log_v)
        eps = torch.randn_like(std)
        if device == 'cuda':
              eps = eps.to(device)
        return m + eps*std
    
    def decode_notes(self, zV, zW, bu, bp, bu_, bp_):
        val = torch.mm(zV, zW.transpose(0,1))
        #print(zV.size(), zW.size(), val.size())
        #val = torch.mm(F.relu(zV), F.relu(zW.transpose(0,1)))
        #out = F.relu(val) # nodec 
        bias = torch.mm(bu, bp.transpose(0,1)) + torch.mm(bu_, bp_.transpose(0,1))
        #print(bu.size(), bp.size(), bias.size())
        out = val + bias
        return out
    
    def forward(self, x, idx, in_row = True):
        if in_row == True:
            muV, logv_V = self.E1(x)
            V = self.reparametrize(muV, logv_V)
            W = self.reparametrize(store_muW, store_logv_W)
            if (idx+1)*batch_size_R < M:
                b = batch_size_R 
                b_u1 = b_u[idx*b:(idx+1)*b,:]
                b_u_ = bu_i[idx*b:(idx+1)*b,:]
            else:    # si pas divisible
                b = M - idx*batch_size_R   # le reste
                b_u1 = b_u[idx*batch_size_R:M,:]
                b_u_ = bu_i[idx*batch_size_R:M,:]
            out = self.decode_notes(V, W, b_u1, bp_j, b_u_, b_p)
            return out, muV, logv_V, self.log_vareps  
        
        else:
            muW, logv_W = self.E2(x)
            V = self.reparametrize(store_muV, store_logv_V)
            W = self.reparametrize(muW, logv_W)
            if (idx+1)*batch_size_C < P:
                b = batch_size_C 
                b_p1 = b_p[idx*b:(idx+1)*b,:]
                b_p_ = bp_j[idx*b:(idx+1)*b,:]
            else:    # si pas divisible
                b = P - idx*batch_size_C   # le reste
                b_p1 = b_p[idx*batch_size_C:P,:]
                b_p_ = bp_j[idx*batch_size_C:P,:]
            out = self.decode_notes(W, V, b_p1, bu_i, b_p_, b_u)
            return out, muW, logv_W, self.log_vareps
        
             
# loss function
def lossf(target, out, mu, logv, log_vareps, in_row = True):
    
    S = torch.exp(logv)
    # main loss component
    vareps = torch.exp(log_vareps)
    if device == "cuda":
        vareps = vareps.to("cuda") 
    
    MC = 0.5/vareps*(target-out)*(target-out)
    MC = torch.sum(MC + 0.5*log_vareps.expand_as(target))

    # ** computing the first KL divergence
    m = torch.Tensor(1, int_dim).fill_(0.0) # Nota: requires_grad is set to False by default ==> No optimization with respect to the prior parameters
    v = torch.Tensor(1, int_dim).fill_(1.0)
    if device=="cuda":
        m = m.to(device)
        v = v.to(device) 
    log_v = torch.log(v)
    
    scale_factor = torch.ones(1, int_dim)
    if device=="cuda":
        scale_factor = scale_factor.to(device)
    
    # due to the batch size we need the prior and the posterior pmts to have the same dims
    m = m.expand_as(mu)
    v = v.expand_as(S)
    log_v = log_v.expand_as(logv)
    scale_factor=scale_factor.expand_as(mu)
    
    var_division    = S / v
    diff            = mu - m
    diff_term       = diff * diff / v
    logvar_division = log_v - logv
            
    KL = 0.5 * ( torch.sum(var_division + diff_term + logvar_division - scale_factor))
 
    # ** computing the second KL divergence
    # m2 = torch.Tensor(1, int_dim).fill_(0.0) # Nota: requires_grad is set to False by default ==> No optimization with respect to the prior parameters
    # v2 = torch.Tensor(1, int_dim).fill_(1.0)
    # log_v2 = torch.log(v2)
    
    # scale_factor2 = torch.ones(1, int_dim)
    
    # # due to the batch size we need the prior and the posterior pmts to have the same dims
    # m2 = m2.expand_as(muW)
    # v2 = v2.expand_as(SW)
    # log_v2 = log_v2.expand_as(logv_W)
    # scale_factor2=scale_factor2.expand_as(muW)
        
    # var_division2    = SW / v2
    # diff2            = muW - m2
    # diff_term2       = diff2 * diff2 / v2
    # logvar_division2 = log_v2 - logv_W
            
    # KLw = 0.5 * ( torch.sum(var_division2 + diff_term2 + logvar_division2 - scale_factor2))
    if in_row == True:
        correct = batch_size_R/M
        return  correct*(MC + KL) 
    else:
        correct = batch_size_C/P  
        return  correct*(MC + KL) 

    
def vlossf(targetN, outN):
        MCN = (targetN-outN)*(targetN-outN)
        MCN = np.sum(MCN)
        return np.sqrt(MCN/len(targetN))

itA = iter(Rdload)
itB = iter(Cdload)
inA = itA.next()
inB = itB.next()

###################################
## Global muV, muW, log_V, log_W ##
###################################
store_muV = torch.normal(mean=0, std = 1, size = [M, int_dim])
store_logv_V = torch.normal(mean=0, std = 1, size = [M, int_dim])
store_muW = torch.normal(mean=0, std = 1, size = [P, int_dim])
store_logv_W = torch.normal(mean=0, std = 1, size = [P, int_dim])


# the model
Mod = Decoder()
if device=="cuda":
    Mod = Mod.cuda()
optimizer = optim.Adam(Mod.parameters(), lr=2e-3, betas=(0.99, 0.999))
#optimizer = optim.Adam(list(Mod.parameters())+list(Mod.E1.parameters())+list(Mod.E2.parameters()), lr=2e-3, betas=(0.99, 0.999))


def train(epoch, store_muV, store_logv_V, store_muW, store_logv_W):    
    Mod.train()
    optimizer.zero_grad() 
    ######################
    ## MB on rows first ##
    ######################
    for batch_idx, obs in enumerate(Rdload):
        #print(batch_idx, obs.numpy())
        out, muV, logv_V, log_vareps = Mod.forward(obs, batch_idx, in_row = True)
        #print(out.data.numpy().shape)
        loss = lossf(obs[obs!=0], out[obs!=0], muV, logv_V, log_vareps, in_row = True)
        loss.backward()
        optimizer.step()
    
    ##############################################################
    ## Updating variational parameters: store_muV, store_logv_V ##                   
    ##############################################################
    Mod.eval()
    inX = torch.tensor(X, dtype = torch.float32)
    if device=="cuda":
        inX = inX.to(device)
    store_muV_, store_logv_V_ = Mod.E1(inX)
    store_muV = store_muV_.detach()
    store_logv_V = store_logv_V_.detach()
    
    if epoch % 1 == 0:
         print('Train Epoch: {} \tRLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)))
    
    ######################
    ##  MB on columns   ##
    ######################    
    Mod.train()
    optimizer.zero_grad() 
    for batch_idx, obs in enumerate(Cdload):
        #print(batch_idx, obs.numpy())
        out, muW, logv_W, log_vareps = Mod.forward(obs, batch_idx, in_row = False)
        #print(out.data.numpy().shape)
        loss = lossf(obs[obs!=0], out[obs!=0], muW, logv_W, log_vareps, in_row = False)
        loss.backward()
        optimizer.step()
        
    ##############################################################
    ## Updating variational parameters: store_muV, store_logv_V ##                   
    ##############################################################
    Mod.eval()
    inX = torch.tensor(X.transpose(), dtype = torch.float32)
    if device=="cuda":
        inX = inX.to(device)
    store_muW_, store_logv_W_ = Mod.E2(inX)
    store_muW = store_muW_.detach()
    store_logv_W = store_logv_W_.detach()
    
    if epoch % 1 == 0:
         print('Train Epoch: {} \tCLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)))

    return store_muV, store_logv_V, store_muW, store_logv_W
    
store_vloss = np.zeros(epochs)
all_muV = torch.zeros(size = [epochs, M, int_dim])
all_logv_V = torch.zeros(size = [epochs, M, int_dim])
all_muW = torch.zeros(size = [epochs, P, int_dim])
all_logv_W = torch.zeros(size = [epochs, P, int_dim])

start = time.process_time()

for epoch in range(epochs):
    store_muV, store_logv_V, store_muW, store_logv_W = train(epoch, store_muV, store_logv_V, store_muW, store_logv_W)    
    ###############
    # valid check #
    ###############
    out = torch.mm(store_muV, store_muW.transpose(0,1))+ torch.mm(b_u, bp_j.transpose(0,1)) + torch.mm(bu_i, b_p.transpose(0,1))
    out = out.numpy()
    perdita = vlossf(dat[val_pos], out[val_pos])
    store_vloss[epoch] = perdita
    all_muV[epoch] = store_muV
    all_logv_V[epoch] = store_logv_V
    all_muW[epoch] = store_muW
    all_logv_W[epoch] = store_logv_W
    if epoch % 1 == 0:
        print('Validation RMSE {:.6f}'.format(perdita)) 
    
end = time.process_time()
print(end-start)    

minloss = store_vloss[0]
minidx = 0
for idx in range(epochs):
    if store_vloss[idx] < minloss:
        minloss = store_vloss[idx]
        minidx = idx
print('minloss: ', minloss, 'minidx: ', minidx)      
best_muV = all_muV[minidx] 
best_logv_V = all_logv_V[minidx] 
best_muW = all_muW[minidx] 
best_logv_W = all_logv_W[minidx]  

## test
out1 = torch.mm(best_muV, best_muW.transpose(0,1))+ torch.mm(b_u, bp_j.transpose(0,1)) + torch.mm(bu_i, b_p.transpose(0,1))
out1 = out1.numpy()

mnout = out1[test_pos[0], test_pos[1]]
minA = dat[test_pos[0], test_pos[1]]

rmse = np.sqrt(np.sum( (mnout - minA)**2  )/len(mnout))
print( 'Our model RMSE: {}'.format(rmse))

###########################################################################
muV = best_muV
muW = best_muW

from  sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans

eclr = KMeans(n_clusters = 2).fit(out1)
eclc = KMeans(n_clusters = 2).fit(out1.T)

print(ari(clr, eclr.labels_))
print(ari(eclc.labels_, clc))
 
colR = []
for idx in range(len(clr)):
    if clr[idx] == 0:
        colR.append('red')
    else:
        colR.append('blue')

##
colC = []
for idx in range(len(clc)):
    if clc[idx] == 0:
        colC.append('yellow')
    else:
        colC.append('green')
        
# plt.plot(store_vloss, color = 'red') 
        
# f, ax = plt.subplots(1)
# ax.scatter(muV[:,0], muV[:,1], color = colR)
# ax.scatter(muW[0,:], muW[1,:], color = colC)

import matplotlib.pyplot as plt   
from sklearn.manifold import TSNE

V_embedded = TSNE(n_components=2,init='pca', perplexity=30,early_exaggeration=100).fit_transform(muV)
W_embedded = TSNE(n_components=2,init='pca', perplexity=30,early_exaggeration=100 ).fit_transform(muW)
        
f, ax = plt.subplots(1,figsize=(15,10))
ax.scatter(V_embedded[:,0], V_embedded[:,1], color = 'yellow', label='User')
ax.scatter(W_embedded[:,0], W_embedded[:,1], color = 'blue', label='Product')
#ax.set_xlabel('Visualization of users and products for simulated data',fontsize=20)
plt.legend(loc='upper right',fontsize=24)
plt.show()
f.savefig("C:/Users/Dingge/Doctoral_projets/Pytorch/visu_simu_no_text.pdf", bbox_inches='tight')

# # emnout = np.round(mnout)
# # eonout = np.round(onout)
# # from sklearn.metrics import accuracy_score
# # print(" missing accuracy score:{} ".format(accuracy_score(emnout, minA)))
# # accuracy_score(eonout, oinA)

# # ##
# # print(" The estimated eta square is: {}".format(torch.exp(log_vareps)))
# # ## RMSE on the test dataset
# # rmse = np.sqrt(np.sum( (mnout - minA)**2  )/len(mnout))
# # print( 'Our model RMSE: {}'.format(rmse))
# print(' ARI in row: {}'.format(ari(clr, eclr.labels_)))
# print(' ARI in col: {}'.format(ari(eclc.labels_, clc)))

