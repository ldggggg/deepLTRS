#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:09:46 2019

@author: marco
"""

import numpy as np
import torch
import pickle
import torch.cuda
from torch import nn, optim
#import os
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader  #, Sampler
#from torch.autograd import Variable
import time
import os
    
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0')
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
M,P = dat.shape

# loading labels
clr = np.load('clr_750_600.npy')
clc = np.load('clc_750_600.npy')

## inserting missing values in the ordinal data matrix
dat_ = dat.copy()
val = 0.0
import random
# random.seed(0)
# np.random.seed(0)
# we are storing the positions of validation and test entries!
ipv = jpv = ipt = jpt = np.zeros(shape = (1,1))
ix = [(row, col) for row in range(dat_.shape[0]) for col in range(dat_.shape[1])]
for row, col in random.sample(ix, int(round(.5*len(ix)))):
    if np.random.uniform() < 1/4:    # validation
        ipv = np.vstack((ipv, row))
        jpv = np.vstack((jpv, col))
        dat_[row, col] = 0
    elif np.random.uniform() > 3/4:                            # test
        ipt = np.vstack((ipt, row))
        jpt = np.vstack((jpt, col))
        dat_[row, col] = 0        
ipv = ipv[1:,0].astype('int32')
jpv = jpv[1:,0].astype('int32')
ipt = ipt[1:,0].astype('int32')
jpt = jpt[1:,0].astype('int32')

# saving positions of observed values
store = np.where(dat_ != 0)
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
            if device=="cuda":
                dat = dat.to(device)
        return dat
    
    def __len__(self):
        return self.L

## target data, row and column data 
Rdata = MDataset(dat_, transform = True)      # The individual/input matrix
Cdata = MDataset(dat_.transpose(), transform = True)      # The object/input matrix

batch_size = 5

## Loading all datasets
Rdload = DataLoader(Rdata,
                    batch_size = batch_size , 
                    shuffle = False
                    )  
Cdload = DataLoader(Cdata,
                    batch_size = batch_size ,
                    shuffle = False
                    )

# Global parameters
init_dim_R = P 
init_dim_C = M
mid_dim = 50
#add_dim = 150
mid_dim_out = 80

another_dim = 80
nb_of_topics = 50

int_dim = 50
epochs = 10
mc_iter = 1


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
        #self.newen1 = nn.Linear(add_dim, mid_dim)
        self.en2 = nn.Linear(mid_dim, mid_dim)
        self.en2_drop = nn.Dropout(0.2)
        #self.int = nn.Linear(mid_dim, 1)               # one coeff. for each row
        self.mu = nn.Linear(mid_dim, int_dim)
        self.mu_bn  = nn.BatchNorm1d(int_dim)                   # bn for mean        
        self.logv = nn.Linear(mid_dim, int_dim)
        self.logv_bn  = nn.BatchNorm1d(int_dim)                   # bn for mean
        
    
    def encode(self, x):
        h1 = F.softplus(self.en1(x))
        #h0 = F.softplus(self.newen1(h1))
        h2 = self.en2_drop(F.softplus(self.en2(h1)))
        mu = self.mu_bn(self.mu(h2))
        logv =self.logv_bn(self.logv(h2))
        #intercept = self.int(h1)
        return mu, logv
        
    def forward(self, x):
        return self.encode(x)

# Decoding NN reconstructing the notes, containing two Encoders
class Decoder(nn.Module):
    def __init__(self):
          super(Decoder, self).__init__()           
          self.E1 = Encoder('R')
          self.E2 = Encoder('C')
          self.p_drop = nn.Dropout(0.2)
          # decoding layers: notes
          self.D1 = nn.Linear(P, mid_dim_out)
          self.D2 = nn.Linear(mid_dim_out, P) 
          self.log_vareps = nn.Parameter(torch.randn(1))

          
    def reparametrize(self, m, log_v):
          std = torch.exp(0.5*log_v)
          eps = torch.randn_like(std)
          if device == 'cuda':
              eps = eps.to(device)
          return m + eps*std
    
    def decode_notes(self, zV, zW):
        val = torch.mm(zV, zW.transpose(0,1))
        out = val # nodec
        #out = self.D2(F.relu(self.D1(val))) # dec
        #out = [o.data for o in out]
        return out
        
    def forward(self, x, in_row = True):
        if in_row == True:
            muV, logv_V = self.E1(x)
            V = self.reparametrize(muV, logv_V)
            W = self.reparametrize(store_muW, store_logv_W)
            out_notes = self.decode_notes(V, W)
            return out_notes, muV, logv_V, self.log_vareps 

        else:
            muW, logv_W = self.E2(x)
            V = self.reparametrize(store_muV, store_logv_V)
            W = self.reparametrize(muW, logv_W)
            out_notes = self.decode_notes(W, V)
            return out_notes, muW, logv_W, self.log_vareps     
    
             
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
 
    if in_row == True:
        correct = batch_size/M
        return  correct*(MC + KL) 
    else:
        correct = batch_size/P  
        return  correct*(MC + KL) 
    
def vlossf(targetN, outN):
    MCN = (targetN-outN)*(targetN-outN)
    MCN = torch.sum(MCN)
    return torch.sqrt(MCN/len(targetN))

itA = iter(Rdload)
itB = iter(Cdload)
inA = itA.next()
inB = itB.next()

store_muV = torch.normal(mean=0, std = 1, size = [M, int_dim])
store_logv_V = torch.normal(mean=0, std = 1, size = [M, int_dim])
store_muW = torch.normal(mean=0, std = 1, size = [P, int_dim])
store_logv_W = torch.normal(mean=0, std = 1, size = [P, int_dim])

# the models
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
        out, muV, logv_V, log_vareps = Mod.forward(obs, in_row = True)
        #print(out.data.numpy().shape)
        loss = lossf(obs[obs!=0], out[obs!=0], muV, logv_V, log_vareps, in_row = True)
        loss.backward()
        optimizer.step()
    
    ##############################################################
    ## Updating variational parameters: store_muV, store_logv_V ##                   
    ##############################################################
    Mod.eval()
    inX = torch.tensor(dat_, dtype = torch.float32)
    if device=="cuda":
        inX = inX.to(device)
    store_muV_, store_logv_V_ = Mod.E1(inX)
    store_muV = store_muV_.detach()
    store_logv_V = store_logv_V_.detach()
    
    if epoch % 1 == 0:
        print('Train Epoch: {} \tRLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)/(len(inA)*len(inA.transpose(0,1)))))
    
    ######################
    ##  MB on columns   ##
    ######################    
    Mod.train()
    optimizer.zero_grad() 
    for batch_idx, obs in enumerate(Cdload):
        #print(batch_idx, obs.numpy())
        out, muW, logv_W, log_vareps = Mod.forward(obs, in_row = False)
        #print(out.data.numpy().shape)
        loss = lossf(obs[obs!=0], out[obs!=0], muW, logv_W, log_vareps, in_row = False)
        loss.backward()
        optimizer.step()
        
    ##############################################################
    ## Updating variational parameters: store_muV, store_logv_V ##                   
    ##############################################################
    Mod.eval()
    inY = torch.tensor(dat_.transpose(), dtype = torch.float32)
    if device=="cuda":
        inY = inY.to(device)
    store_muW_, store_logv_W_ = Mod.E2(inY)
    store_muW = store_muW_.detach()
    store_logv_W = store_logv_W_.detach()

    if epoch % 1 == 0:
        print('Train Epoch: {} \tCLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)/(len(inB)*len(inB.transpose(0,1)))))

    return store_muV, store_logv_V, store_muW, store_logv_W

store_vloss = np.zeros(epochs)
all_muV = torch.zeros(size = [epochs, M, int_dim])
all_logv_V = torch.zeros(size = [epochs, M, int_dim])
all_muW = torch.zeros(size = [epochs, P, int_dim])
all_logv_W = torch.zeros(size = [epochs, P, int_dim])

start = time.time()

for epoch in range(epochs): 
    store_muV, store_logv_V, store_muW, store_logv_W = train(epoch, store_muV, store_logv_V, store_muW, store_logv_W)    
    ###############
    # valid check #
    ###############
    out = torch.mm(store_muV, store_muW.transpose(0,1))
    indat = torch.Tensor(dat)
    if device=="cuda":
          indat = indat.to(device)
    perdita = vlossf(indat[val_pos], out[val_pos])
    store_vloss[epoch] = perdita
    all_muV[epoch] = store_muV
    all_logv_V[epoch] = store_logv_V
    all_muW[epoch] = store_muW
    all_logv_W[epoch] = store_logv_W
    if epoch % 1 == 0:
        print('Validation RMSE {:.6f}'.format(perdita))   
        
end = time.time()
print(end-start)    

########### save the best store parameters ##########
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
Mod.eval()
store_muV, store_logv_V, store_muW, store_logv_W = train(epoch, store_muV, store_logv_V, store_muW, store_logv_W) 
outN = torch.mm(best_muV, best_muW.transpose(0,1))
noutN = outN.cpu().data.numpy()

mnout = noutN[test_pos[0], test_pos[1]]
minA = dat[test_pos[0], test_pos[1]]

rmse = np.sqrt(np.sum((mnout - minA)**2)/len(mnout))
print( 'Our model (test data) RMSE: {}'.format(rmse))

muV = store_muV.cpu().data.numpy()
muW = store_muW.cpu().data.numpy()

from  sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
# plt.plot(store_vloss, color = 'red') 

eclr = KMeans(n_clusters = 2).fit(noutN)
eclc = KMeans(n_clusters = 2).fit(noutN.T)

print(" ARI (rows):{} ".format(ari(clr, eclr.labels_)))
print(" ARI (cols):{} ".format(ari(eclc.labels_, clc)))

from sklearn.metrics import accuracy_score
emnout = np.round(mnout)
print(" missing accuracy score:{} ".format(accuracy_score(emnout, minA)))

# print(" The estimated eta square is: {}".format(torch.exp(log_vareps)))

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

#from sklearn.decomposition import PCA       
#pca = PCA(n_components = 2) 
#o1 = pca.fit(muV).fit_transform(muV)
#o2 = pca.fit(muW.transpose()).fit_transform(muW.transpose())
#        
f, ax = plt.subplots(1)
ax.scatter(muV[:,0], muV[:,1], color = colR, label='User')
ax.scatter(muW[:,0], muW[:,1], color = colC, label='Product')
#
#
#
### direct k-means on Y
##kclr = KMeans(n_clusters = 2).fit(inA)
##kclc = KMeans(n_clusters = 2).fit(inB)
##
##ari(clr, kclr.labels_)
##ari(kclc.labels_, clc)
#
#emnout = np.round(mnout)
#eonout = np.round(onout)
#from sklearn.metrics import accuracy_score
#accuracy_score(emnout, minA)
#accuracy_score(eonout, oinA)
    

# ######## Hierarchical Poisson Recommendation (for comparison) ##########
# import pandas as pd
# from sklearn.metrics import accuracy_score
# from hpfrec import HPF

# (i,j) = X.nonzero()
# counts_df = pd.DataFrame({
#         'UserId' : i,
#         'ItemId' : j,
#         'Count'  : X[(i,j)].astype('int32')
#         }
#         )

# val_df = pd.DataFrame({
#         'UserId' : ipv,
#         'ItemId' : jpv,
#         'Count'  : dat[val_pos]
#         })

# ## Initializing the model object
# recommender = HPF()


# ## For stochastic variational inference, need to select batch size (number of users)
# recommender = HPF(users_per_batch = 20)

# ## Full function call
# recommender = HPF(
#  	k=30, a=0.3, a_prime=0.3, b_prime=1.0,
#  	c=0.3, c_prime=0.3, d_prime=1.0, ncores=-1,
#  	stop_crit='train-llk', check_every=10, stop_thr=1e-3,
#  	users_per_batch=None, items_per_batch=None, step_size=lambda x: 1/np.sqrt(x+2),
#  	maxiter=100, reindex=True, verbose=True,
#  	random_seed = None, allow_inconsistent_math=False, full_llk=False,
#  	alloc_full_phi=False, keep_data=True, save_folder=None,
#  	produce_dicts=True, keep_all_objs=True, sum_exp_trick=False
# )

# ## Fitting the model while monitoring a validation set
# recommender = HPF(stop_crit='val-llk')
# recommender.fit(counts_df, val_set = val_df )

# ## Fitting the model to the data
# #recommender.fit(counts_df)

# ## Making predictions on the train dataset
# obsout = recommender.predict(user = i, item = j)
# obsout = np.round(obsout)
# accuracy_score(obsout, X[(i,j)])

# ## Making predictions on the test dataset
# mout = recommender.predict(user = test_pos[0], item = test_pos[1])
# #accuracy_score(np.round(mout), dat[test_pos])

# ## computing the RMSE on the test dataset
# rmse_hpf = np.sqrt(np.sum( (mout - dat[test_pos])**2  )/len(mout))
# print( 'Our model RMSE: {}'.format(rmse))
#print( 'HPF RMSE: {}'.format(rmse_hpf))

# ###############

# # def print_top_words_z(beta, vocab, n_top_words = 10):
# #     # averaging  accross all topix
# #     nbeta = beta.mean(axis = 0)
# #     nbeta = beta - nbeta
# #     for i in range(len(nbeta)):
# #                 print('\n--------------Topic{}:-----------------'.format(i+1))
# #                 line = " ".join([vocab[j] 
# #                             for j in beta[i].argsort()[:-n_top_words - 1:-1]])
# #                 print('     {}'.format(line))
# #                 print('--------------End of Topic{}---------------'.format(i+1))


# # beta = Mod.D3.weight.data.numpy().T
# # print_top_words_z(beta, dct)
