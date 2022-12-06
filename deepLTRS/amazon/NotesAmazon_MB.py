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

############# Auxiliary Functions #############################################

def ipos(M, P, idx):
    return np.arange(idx*P, idx*P + P)

def jpos(M, P, idy):
    return np.arange(idy, idy + (M*P), P)

###############################################################################    
# loading Amazon notes
    
Y = np.genfromtxt('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/amazon/AmazonOrdinal.csv', 
                  skip_header=1, delimiter=",",)
# Y = np.genfromtxt('AmazonOrdinal.csv', 
#                   skip_header=1, delimiter=",",)
dat = Y[:,1:]
M = dat.shape[0] # 1644
P = dat.shape[1] # 1733
clr = np.random.choice(range(2), M) 
clc = np.random.choice(range(2), P)

############## loading and manipulatind docs and vocabulary  ###############   
nonzero_pos = np.where(dat != 0)
nonzero_row = nonzero_pos[0].tolist()
nonzero_col = nonzero_pos[1].tolist()

X = dat.copy()
## inserting missing values
val = 0.0
import random
# we are storing the positions of validation and test entries!
ipv = jpv = ipt = jpt = np.zeros(shape = (1,1))
# random.seed(0)
# np.random.seed(0)
# l = list()
# for index in random.sample(range(0,len(nonzero_row)), len(nonzero_row)):  
#     l.append(index)
# print(len(l))

#for index in range(len(nonzero_row)):
for index in random.sample(range(0,len(nonzero_row)), len(nonzero_row)):    
    if np.random.uniform() < 1/10:    # validation
        ipv = np.vstack((ipv, nonzero_row[index]))
        jpv = np.vstack((jpv, nonzero_col[index]))
        X[nonzero_row[index], nonzero_col[index]] = 0
    elif np.random.uniform() > 3/4:                            # test
        ipt = np.vstack((ipt, nonzero_row[index]))
        jpt = np.vstack((jpt, nonzero_col[index]))
        X[nonzero_row[index], nonzero_col[index]] = 0        
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
        return dat
    
    def __len__(self):
        return self.L

## row and column data
Rdata = MDataset(X, transform = True)
Cdata = MDataset(X.transpose(), transform = True)

batch_size = 200

## Loading both
Rdload = DataLoader(Rdata,
                    batch_size = batch_size, 
                    shuffle = True)  

Cdload = DataLoader(Cdata,
                    batch_size = batch_size,
                    shuffle = True
                    )

# Global parameters
init_dim_R = P
init_dim_C = M
mid_dim = 80
mid_dim_out = 80
int_dim = 10
# prior_variance = 0.5
epochs = 3000

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
          self.D1 = nn.Linear(init_dim_R, mid_dim_out)
          self.D2 = nn.Linear(mid_dim_out, init_dim_R)
          self.log_vareps = nn.Parameter(torch.randn(1))
          
    def reparametrize(self, m, log_v):
        std = torch.exp(0.5*log_v)
        eps = torch.randn_like(std)
        return m + eps*std
    
    def forward(self, x, in_row = True):
        if in_row == True:
            muV, logv_V = self.E1(x)
            # muW, logv_W = self.E2(torch.transpose(x,0,1))
            V = self.reparametrize(muV, logv_V)
            W = self.reparametrize(store_muW, store_logv_W)
            val = torch.mm(V, W.transpose(0,1)) 
            #out = self.D2(F.relu(self.D1(val)))
            out = val
            return out, muV, logv_V, self.log_vareps    
        else:
            muW, logv_W = self.E2(x)
            V = self.reparametrize(store_muV, store_logv_V)
            W = self.reparametrize(muW, logv_W)
            val = torch.mm(W, V.transpose(0,1)) 
            #out = self.D2(F.relu(self.D1(val)))
            out = val
            return out, muW, logv_W, self.log_vareps
        
             
# loss function
def lossf(target, out, mu, logv, log_vareps, in_row = True):
    
    S = torch.exp(logv)
    # main loss component
    vareps = torch.exp(log_vareps)
    
    MC = 0.5/vareps*(target-out)*(target-out)
    MC = torch.sum(MC + 0.5*log_vareps.expand_as(target))

    # ** computing the first KL divergence
    m = torch.Tensor(1, int_dim).fill_(0.0) # Nota: requires_grad is set to False by default ==> No optimization with respect to the prior parameters
    v = torch.Tensor(1, int_dim).fill_(1.0)
    log_v = torch.log(v)
    
    scale_factor = torch.ones(1, int_dim)
    
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
        correct = batch_size/M
        return  correct*(MC + KL) 
    else:
        correct = batch_size/P  
        return  correct*(MC + KL) 

    
def vlossf(targetN, outN):
        MCN = (targetN-outN)*(targetN-outN)
        MCN = np.sum(MCN)
        return np.sqrt(MCN/len(targetN))

# itA = iter(Rdload)
# inA = itA.next()

###################################
## Global muV, muW, log_V, log_W ##
###################################
store_muV = torch.normal(mean=0, std = 1, size = [M, int_dim])
store_logv_V = torch.normal(mean=0, std = 1, size = [M, int_dim])
store_muW = torch.normal(mean=0, std = 1, size = [P, int_dim])
store_logv_W = torch.normal(mean=0, std = 1, size = [P, int_dim])


# the model
Mod = Decoder()
optimizer = optim.Adam(Mod.parameters(), lr=2e-5, betas=(0.99, 0.999))
#optimizer = optim.Adam(list(Mod.parameters())+list(Mod.E1.parameters())+list(Mod.E2.parameters()), lr=2e-3, betas=(0.99, 0.999))


def train(epoch, store_muV, store_logv_V, store_muW, store_logv_W):    
    Mod.train()
    optimizer.zero_grad() 
    ######################
    ## MB on rows first ##
    ######################
    for batch_idx, obs in enumerate(Rdload):
        out, muV, logv_V, log_vareps = Mod.forward(obs, in_row = True)
        loss = lossf(obs[obs!=0], out[obs!=0], muV, logv_V, log_vareps, in_row = True)
        loss.backward()
        optimizer.step()
    
    ##############################################################
    ## Updating variational parameters: store_muV, store_logv_V ##                   
    ##############################################################
    Mod.eval()
    store_muV_, store_logv_V_ = Mod.E1(torch.tensor(X, dtype = torch.float32))
    store_muV = store_muV_.detach()
    store_logv_V = store_logv_V_.detach()
    
    ######################
    ##  MB on columns   ##
    ######################    
    Mod.train()
    optimizer.zero_grad() 
    for batch_idx, obs in enumerate(Cdload):
        out, muW, logv_W, log_vareps = Mod.forward(obs, in_row = False)
        loss = lossf(obs[obs!=0], out[obs!=0], muW, logv_W, log_vareps, in_row = False)
        loss.backward()
        optimizer.step()
        
    ##############################################################
    ## Updating variational parameters: store_muV, store_logv_V ##                   
    ##############################################################
    Mod.eval()
    store_muW_, store_logv_W_ = Mod.E2(torch.tensor(X.transpose(), dtype = torch.float32))
    store_muW = store_muW_.detach()
    store_logv_W = store_logv_W_.detach()

    return store_muV, store_logv_V, store_muW, store_logv_W
            
        
    # Mod.eval()
    # # validation RMSE loss
    # vloss = vlossf(torch.FloatTensor(dat[val_pos[0], val_pos[1]]), out[val_pos[0], val_pos[1]])
    # #loss = test_loss(inA, muV, muW)
    # if epoch % 100 == 0:
    #     print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)/(len(inA)*len(inA.transpose(0,1)))))
    #     print('Validation RMSE {:.6f}'.format(vloss))

    
store_vloss = np.zeros(epochs)

start = time.process_time()

for epoch in range(epochs):
    store_muV, store_logv_V, store_muW, store_logv_W = train(epoch, store_muV, store_logv_V, store_muW, store_logv_W)    
    ###############
    # valid check #
    ###############
    out = torch.mm(store_muV, store_muW.transpose(0,1))
    out = out.numpy()
    perdita = vlossf(dat[val_pos], out[val_pos])
    store_vloss[epoch] = perdita
    print("validation loss: {}".format(perdita))
    
end = time.process_time()
print(end-start)    

## test
Mod.eval()
store_muV, store_logv_V, store_muW, store_logv_W = train(epoch, store_muV, store_logv_V, store_muW, store_logv_W) 
out1 = torch.mm(store_muV, store_muW.transpose(0,1))
out1 = out1.numpy()

# observed
onout = out1[store[0], store[1]]
oinA = dat[store[0], store[1]]

# missing
mnout = out1[test_pos[0], test_pos[1]]
minA = dat[test_pos[0], test_pos[1]]

rmse = np.sqrt(np.sum( (mnout - minA)**2  )/len(mnout))
print( 'Our model RMSE: {}'.format(rmse))

muV = store_muV
muW = store_muW.transpose(0,1)

# =============================================================================
# from  sklearn.metrics import adjusted_rand_score as ari
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# 
# eclr = KMeans(n_clusters = 2).fit(out)
# eclc = KMeans(n_clusters = 2).fit(out.T)
# 
# print(ari(clr, eclr.labels_))
# print(ari(eclc.labels_, clc))
# 
# 
# ## 
# colR = []
# for idx in range(len(clr)):
#     if clr[idx] == 0:
#         colR.append('blue')
#     else:
#         colR.append('blue')
# 
# ##
# colC = []
# for idx in range(len(clc)):
#     if clc[idx] == 0:
#         colC.append('blue')
#     else:
#         colC.append('blue')
#         
plt.plot(store_vloss, color = 'red')   
plt.title('MB_val_loss')
plt.show()
    
# =============================================================================
plt.subplot(211)
plt.scatter(muV[:,0], muV[:,1], color = 'yellow', label = 'MB_user') # user
plt.legend(loc='lower right')
plt.subplot(212)
plt.scatter(muW[0,:], muW[1,:], color = 'green', label = 'MB_product') # product
plt.legend(loc='lower right')

f, ax = plt.subplots(1)
ax.scatter(muV[:,0], muV[:,1], color = 'blue') # user
ax.scatter(muW[0,:], muW[1,:], color = 'red', alpha = 0.5) # product
ax.set_title('MB_user_product')

data = [mnout[minA==1], mnout[minA==2], mnout[minA==3], mnout[minA==4], mnout[minA==5]]
fig7, ax7 = plt.subplots()
ax7.set_title('MB_Prediction')
ax7.boxplot(data)

############################################################################
emnout = np.round(mnout)
eonout = np.round(onout)
from sklearn.metrics import accuracy_score
print('MB_miss_accuracy: {}'.format(accuracy_score(emnout, minA))) # missing
print('MB_obs_accuracy: {}'.format(accuracy_score(eonout, oinA))) # observed

# print(" The estimated eta square is: {}".format(torch.exp(log_vareps)))
# ## RMSE on the test dataset
# rmse = np.sqrt(np.sum( (mnout - minA)**2  )/len(mnout))
# print( 'Our model RMSE: {}'.format(rmse))
# print(' ARI in row: {}'.format(ari(clr, eclr.labels_)))
# print(' ARI in col: {}'.format(ari(eclc.labels_, clc)))

