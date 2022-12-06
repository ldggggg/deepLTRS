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
device = "cpu"
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

# # import random
# # # we are storing the positions of validation and test entries!
# # ipv = jpv = np.zeros(shape = (1,1))
# # #random.seed(0)
# # #np.random.seed(0)
# # ix = [(row, col) for row in range(X.shape[0]) for col in range(X.shape[1])]
# # for row in range(X.shape[0]):
# #     for col in range(X.shape[1]):
# #         if X[row, col] == 0:
# #             ipv = np.vstack((ipv, row))
# #             jpv = np.vstack((jpv, col))
      
# # ipv = ipv[1:,0].astype('int32')
# # jpv = jpv[1:,0].astype('int32')
# # all_pos = (ipv, jpv)

# # np.save('amazon_valtest_pos', all_pos)

# # row_mean = np.true_divide(X.sum(1),(X!=0).sum(1))
# # for row in range(X.shape[0]):
# #     for col in range(X.shape[1]):
# #         if X[row, col] == 0:
# #             X[row, col] = int(round(row_mean[row]))

# nonzero_pos = np.where(dat != 0)
# zero_pos = np.where(dat == 0)
# zero_pos_row = zero_pos[0].tolist()
# zero_pos_col = zero_pos[1].tolist()

# # seq_rows = np.arange(0,M)
# # shuffle(seq_rows)

# # seq_cols = np.arange(0,P)
# # shuffle(seq_cols)

# # all_row_pos = list()
# # all_col_pos = list()
# # for row in seq_rows:
# #     for col in seq_cols:           
# #         all_row_pos.append(row)
# #         all_col_pos.append(col) 
        
# # shuffle(np.asarray(all_row_pos))
# # shuffle(np.asarray(all_col_pos))
# # all_pos = (np.asarray(all_row_pos), np.asarray(all_col_pos))

# # validation and test positions
# #l = int(round(.15*len(all_pos[0])))
# ll = int(round(.15*len(nonzero_pos[0])))
# lll = int(round(.3*len(nonzero_pos[0])))

# for row in nonzero_pos[0][lll:]:
#     zero_pos_row.append(row)
# for col in nonzero_pos[1][lll:]:
#     zero_pos_col.append(col)
    
# all_pos = (np.asarray(zero_pos_row), np.asarray(zero_pos_col))

# #val_pos = (all_pos[0][:l], all_pos[1][:l])
# val_pos = (nonzero_pos[0][ll:lll], nonzero_pos[1][ll:lll])
# store = (nonzero_pos[0][lll:], nonzero_pos[1][lll:])
# test_pos = (nonzero_pos[0][:ll], nonzero_pos[1][:ll])

####################### process missing value with mean #######################
# row_mean = np.true_divide(X.sum(1),(X!=0).sum(1))
# for row in range(X.shape[0]):
#     for col in range(X.shape[1]):
#         if X[row, col] == 0:
#             X[row, col] = int(round(row_mean[row]))

# seq_rows = np.arange(0,M)
# shuffle(seq_rows)

# seq_cols = np.arange(0,P)
# shuffle(seq_cols)

# train_row = int(round(.75*len(seq_rows)))
# train_col = int(round(.75*len(seq_cols)))

# val_row = int(round(.77*len(seq_rows)))
# val_col = int(round(.77*len(seq_cols)))

# train_row_pos = list()
# train_col_pos = list()
# for row in seq_rows[:train_row]:
#     for col in seq_cols[:train_col]:           
#         train_row_pos.append(row)
#         train_col_pos.append(col)                 
# store = (np.asarray(train_row_pos), np.asarray(train_col_pos))

# val_row_pos = list()
# val_col_pos = list()
# for row in seq_rows[train_row: val_row]:
#     for col in seq_cols[train_col: val_col]:           
#         val_row_pos.append(row)
#         val_col_pos.append(col)                 
# val_pos = (np.asarray(val_row_pos), np.asarray(val_col_pos))

# test_row_pos = list()
# test_col_pos = list()
# for row in seq_rows[val_row:]:
#     for col in seq_cols[val_col:]:           
#         test_row_pos.append(row)
#         test_col_pos.append(col)                 
# test_pos = (np.asarray(test_row_pos), np.asarray(test_col_pos))
 
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
Ndata = MDataset(dat, transform = True) 
Rdata = MDataset(X, transform = True)

## Loading both

# prop = 0.4
Ndload = DataLoader(Ndata,
                    batch_size = Ndata.__len__(),
                    shuffle = False
                    )
Rdload = DataLoader(Rdata,
                    batch_size = Rdata.__len__(), 
                    shuffle = False)  

# Global parameters
init_dim_R = P
init_dim_C = M
mid_dim = 50
mid_dim_out = 80
int_dim = 50
# mid_dim = 80
# mid_dim_out = 80
# int_dim = 10
# prior_variance = 0.995
epochs = 3000
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
          self.D1 = nn.Linear(init_dim_R, mid_dim_out)
          self.D2 = nn.Linear(mid_dim_out, init_dim_R)
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
        if device == 'cuda':
              eps = eps.to(device)
        return m + eps*std
    
    def forward(self, x):
        muV, logv_V = self.E1(x)
        muW, logv_W = self.E2(torch.transpose(x,0,1))
        V = self.reparametrize(muV, logv_V)
        W = self.reparametrize(muW, logv_W)
        # vR, vC, vS = V.shape
        # wR, wC, wS = W.shape     
        # out = torch.zeros(vR,wR,vS)
        # for idx in range(mc_iter):
        #    val = torch.mm(V[:,:,idx], W[:,:,idx].transpose(0,1)) 
        #    out[:,:,idx] = self.D2(F.relu(self.D1(val)))
        val = torch.mm(V, W.transpose(0,1)) 
        out = self.D2(F.relu(self.D1(val)))
        #out = val
        return out, muV, logv_V, muW, logv_W, self.log_vareps      
        
             
# loss function
def lossf(target, out, muV, muW, logv_V, logv_W, log_vareps):
    
    SV = torch.exp(logv_V)
    SW = torch.exp(logv_W)
    # main loss component
    vareps = torch.exp(log_vareps)
    if device == "cuda":
        vareps = vareps.to("cuda")
    
    MC = 0.5/vareps*(target-out)*(target-out)
    MC = torch.sum(MC + 0.5*log_vareps.expand_as(target))

    # ** computing the first KL divergence
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
    if device=="cuda":
        m2 = m2.to(device)
        v2 = v2.to(device)   
    log_v2 = torch.log(v2)
    
    scale_factor2 = torch.ones(1, int_dim)
    if device=="cuda":
        scale_factor2 = scale_factor2.to(device)    
    # due to the batch size we need the prior and the posterior pmts to have the same dims
    m2 = m2.expand_as(muW)
    v2 = v2.expand_as(SW)
    log_v2 = log_v2.expand_as(logv_W)    
    scale_factor2=scale_factor2.expand_as(muW)
    if device=="cuda":
        scale_factor2 = scale_factor2.to(device)
        
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

itA = iter(Rdload) # X
inA = itA.next()

itN = iter(Ndload) # dat
inN = itN.next()

# the model
Mod = Decoder()
if device=="cuda":
    Mod = Mod.cuda()
optimizer = optim.Adam(Mod.parameters(), lr=2e-5, betas=(0.99, 0.999))
#optimizer = optim.Adam(list(Mod.parameters())+list(Mod.E1.parameters())+list(Mod.E2.parameters()), lr=2e-3, betas=(0.99, 0.999))


def train(epoch):    
    Mod.train()
    # Mod.E1.train()
    # Mod.E2.train()
    optimizer.zero_grad()         
    out, muV, logv_V, muW, logv_W, log_vareps = Mod.forward(inA)
    if epoch == 0:
        print(" Initial variance: {}".format(torch.exp(log_vareps)))
    loss = lossf(inN[store[0], store[1]], out[store[0], store[1]], muV, muW, logv_V, logv_W, log_vareps)
    # loss = lossf(inA[seq_rows[:train_row], seq_cols[:train_row]], out[seq_rows[:train_row], seq_cols[:train_row]], 
    #              muV, muW, logv_V, logv_W, log_vareps)
    
    loss.backward()
    optimizer.step()
    Mod.eval()
    # validation RMSE loss
    vloss = vlossf(inN[[val_pos[0], val_pos[1]]], 
                   out[[val_pos[0], val_pos[1]]])
    #loss = test_loss(inA, muV, muW)
    if epoch % 100 == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)/(len(inA)*len(inA.transpose(0,1)))))
        print('Validation RMSE {:.6f}'.format(vloss))

start = time.process_time()

for epoch in range(epochs):
    train(epoch)    

end = time.process_time()
print(end-start)        

# test (not exactly...)

Mod.eval()

out, muV, logv_V, muW, logv_W, log_vareps = Mod.forward(inA)

########################################################
tval = torch.mm(muV, muW.transpose(0,1)) 
out = Mod.D2(F.relu(Mod.D1(tval)))
#out = tval
#######################################################

out = out.reshape(M, P)
nout = out.cpu().data.numpy()

# # observed
# onout = nout[seq_rows[:train_row], seq_cols[:train_row]]
# oinA = dat[seq_rows[:train_row], seq_cols[:train_row]]

# # missing
# mnout = nout[seq_rows[val_row:], seq_cols[val_row:]]
# minA = dat[seq_rows[val_row:], seq_cols[val_row:]]

# observed
onout = nout[store[0], store[1]]
oinN = inA[store[0], store[1]]

# missing
mnout = nout[test_pos[0], test_pos[1]]
minA = dat[test_pos[0], test_pos[1]]

#out = Mod.decoder(torch.mm(muV, muW.transpose(0,1)))

muV = muV.cpu().data.numpy()
muW = muW.cpu().data.numpy().T


#est = np.matmul(muV, muW)

from  sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# eclr = KMeans(n_clusters = 2).fit(nout)
# eclc = KMeans(n_clusters = 2).fit(nout.T)

# ari(clr, eclr.labels_)
# ari(eclc.labels_, clc)


# ## 
# colR = []
# for idx in range(len(clr)):
#     if clr[idx] == 0:
#         colR.append('red')
#     else:
#         colR.append('blue')

# ##
# colC = []
# for idx in range(len(clc)):
#     if clc[idx] == 0:
#         colC.append('yellow')
#     else:
#         colC.append('green')
        
# f, ax = plt.subplots(1)
# ax.scatter(muV[:,0], muV[:,1], color = colR)
# ax.scatter(muW[0,:], muW[1,:], color = colC)

plt.subplot(211)
plt.scatter(muV[:,0], muV[:,1], color = 'yellow', label = 'Fulldec_user') # user
plt.legend(loc='lower right')
plt.subplot(212)
plt.scatter(muW[0,:], muW[1,:], color = 'green', label = 'Fulldec_product') # product
plt.legend(loc='lower right')

f, ax = plt.subplots(1)
ax.scatter(muV[:,0], muV[:,1], color = 'blue') # user
ax.scatter(muW[0,:], muW[1,:], color = 'red', alpha = 0.5) # product
ax.set_title('Fulldec_user_product')

data = [mnout[minA==1], mnout[minA==2], mnout[minA==3], mnout[minA==4], mnout[minA==5]]
fig7, ax7 = plt.subplots()
ax7.set_title('Fulldec_Prediction')
ax7.boxplot(data)
        
emnout = np.round(mnout)
eonout = np.round(onout)
from sklearn.metrics import accuracy_score
print('Fulldec_miss_accuracy: {}'.format(accuracy_score(emnout, minA))) # missing
print('Fulldec_obs_accuracy: {}'.format(accuracy_score(eonout, oinN))) # observed

##
#print(" The estimated eta square is: {}".format(torch.exp(log_vareps)))
## RMSE on the test dataset
rmse = np.sqrt(np.sum( (mnout - minA)**2  )/len(mnout))
print( 'Our model RMSE: {}'.format(rmse))
# print(' ARI in row: {}'.format(ari(clr, eclr.labels_)))
# print(' ARI in col: {}'.format(ari(eclc.labels_, clc)))

######## Hierarchical Poisson Recommendation (for comparison) ##########
# import pandas as pd, numpy as np
# from hpfrec import HPF
# from sklearn.metrics import accuracy_score

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

## Initializing the model object
# recommender = HPF()


# ## For stochastic variational inference, need to select batch size (number of users)
# recommender = HPF(users_per_batch = 20)

# ## Full function call
# recommender = HPF(
#  	k=100, a=0.3, a_prime=0.3, b_prime=1.0,
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

# for i in range(10):
#     deepLTRS()