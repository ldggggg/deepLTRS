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
    
device = "cuda"
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

############# Auxiliary Functions ##########################################

def ipos(M, P, idx):
    return np.arange(idx*P, idx*P + P)

def jpos(M, P, idy):
    return np.arange(idy, idy + (M*P), P)

# loading Amazon notes    
#Y = np.genfromtxt('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/amazon/AmazonOrdinal.csv', 
#                  skip_header=1, delimiter=",")
Y = np.genfromtxt('AmazonOrdinal.csv', skip_header=1, delimiter=",")
dat = Y[:,1:]
M = dat.shape[0] # 1644
P = dat.shape[1] # 1733
clr = np.random.choice(range(2), M) 
clc = np.random.choice(range(2), P)

############## loading and manipulatind docs and vocabulary  ###############   
nonzero_pos = np.where(dat != 0)
nonzero_row = nonzero_pos[0].tolist()
nonzero_col = nonzero_pos[1].tolist()

dat_ = dat.copy()
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
for index in random.sample(range(0,len(nonzero_row)), len(nonzero_row)):    
    if np.random.uniform() < 1/10:    # validation
        ipv = np.vstack((ipv, nonzero_row[index]))
        jpv = np.vstack((jpv, nonzero_col[index]))
        dat_[nonzero_row[index], nonzero_col[index]] = 0
    elif np.random.uniform() > 3/4:                            # test
        ipt = np.vstack((ipt, nonzero_row[index]))
        jpt = np.vstack((jpt, nonzero_col[index]))
        dat_[nonzero_row[index], nonzero_col[index]] = 0        
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

############## loading and manipulatind docs and vocabulary  ###############
import csv

# with open('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/amazon/vocabulary.csv', newline='') as csvfile:
#     data = list(csv.reader(csvfile))
with open('vocabulary.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
#print(data)

from gensim.corpora import Dictionary
dct = Dictionary(data)
dctn = dct.token2id
V = len(dctn) #37181

############################################################################    
## Now I need to create two dtms (individual specific and object specific..)
    
# complete dtm
# dtm = np.genfromtxt('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/amazon/dtm.csv', 
#                    skip_header=1, delimiter=",") 
dtm = np.genfromtxt('dtm.csv', skip_header=1, delimiter=",") 
dtm = np.asarray(dtm, dtype = 'double') 

cdtm = np.zeros(shape = (len(nonzero_row), V))
for index in range(dtm.shape[0]):
    cdtm[int(dtm[:,0][index])][int(dtm[:,1][index])] = dtm[:,2][index]

num_doc_m = []
for num in range(M):
    num_doc_m.append(len((np.where(dat_[num] != 0))[0]))
print(num_doc_m)

# ** individuals
idtm = np.zeros(shape = (M, V))
i = 0
for idx in range(M):
    pos = np.arange(i, i + num_doc_m[idx])
    idtm[idx, :] = cdtm[pos, :].sum(0)
    i = i + num_doc_m[idx]

num_doc_p = []
for num in range(P):
    dat_trans= dat_.T
    num_doc_p.append(len((np.where(dat_trans[num] != 0))[0]))
print(num_doc_p)
    
# ** object
odtm = np.zeros(shape = (P, V))
j = 0
for idx in range(P):
    pos = np.arange(j, j + num_doc_p[idx])
    odtm[idx, :] = cdtm[pos, :].sum(0)
    j = j + num_doc_p[idx]

## Scaling the dtm(s)
#for idx in range(len(cdtm)):
#    cdtm[idx,:] /= np.sum(cdtm[idx,:])
#
#for idx in range(len(idtm)):
#    idtm[idx,:] /= np.sum(idtm[idx,:])
#
#for idx in range(len(odtm)):    
#    odtm[idx,:] /= np.sum(odtm[idx,:])

    
## We need to cbind idtm-dat and odtm-t(dat)
X = np.concatenate((idtm, dat_), axis = 1)
Y = np.concatenate((odtm, dat_.transpose()), axis = 1)

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
Ndata = MDataset(dat, transform = True)    # The ordinal data matrix (M x P) to reconstruct
Tdata = MDataset(cdtm, transform = True)   # The complete dtm ( one row --> (i,j) ) to reconstruct 
Rdata = MDataset(X, transform = True)      # The individual/input matrix
Cdata = MDataset(Y, transform = True)      # The object/input matrix


## Loading all datasets

Ndload = DataLoader(Ndata,
                    batch_size = Ndata.__len__(),
                    shuffle = False
                    )
Tdload = DataLoader(Tdata,
                    batch_size = Tdata.__len__(),
                    shuffle = False
                    )
Rdload = DataLoader(Rdata,
                    batch_size = Rdata.__len__(), 
                    shuffle = False
                    )  
Cdload = DataLoader(Cdata,
                    batch_size = Cdata.__len__(),
                    shuffle = False
                    )

# Global parameters
init_dim_R = (P + V)
init_dim_C = (M + V)
mid_dim = 50
mid_dim_out = 80

another_dim = 150
nb_of_topics = 50

int_dim = 50
epochs = 1500
mc_iter = 1

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
        self.en2 = nn.Linear(mid_dim, mid_dim)
        self.en2_drop   = nn.Dropout(0.2)
        #self.int = nn.Linear(mid_dim, 1)               # one coeff. for each row
        self.mu = nn.Linear(mid_dim, int_dim)
        self.mu_bn  = nn.BatchNorm1d(int_dim)                   # bn for mean        
        self.logv = nn.Linear(mid_dim, int_dim)
        self.logv_bn  = nn.BatchNorm1d(int_dim)                   # bn for mean
        
    
    def encode(self, x):
        h1 = F.softplus(self.en1(x))
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
          self.p_drop     = nn.Dropout(0.2)
          # decoding layers: notes
          self.D1 = nn.Linear(P, mid_dim_out)
          self.D2 = nn.Linear(mid_dim_out, P)
          # decoding layers: text
          self.newD1 = nn.Linear(2*int_dim, another_dim)
          self.newD2 = nn.Linear(another_dim, nb_of_topics)          
          self.D3 = nn.Linear(nb_of_topics, V, bias = True)             # beta         
          self.D3_bn = nn.BatchNorm1d(V, affine = False)   
          self.log_vareps = nn.Parameter(torch.randn(1))
         
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
    
    def decode_notes(self, zV, zW):
        val = torch.mm(zV, zW.transpose(0,1)) 
        out = self.D2(F.relu(self.D1(val)))
        #out = [o.data for o in out]
        return out
    
    def decode_text(self, zV, zW):
        s1 = zV.shape[0]
        s2 = zW.shape[0]
        s3 = zV.shape[1]
        theta = torch.zeros(size = (s1*s2, s3))
        if device == 'cuda':
            theta = theta.to(device)
        for idx in range(s1):
            start = idx*P
            theta[start:(start + P),:] = zV[idx,:] + zW
        theta = .5*theta 
        p = F.softmax(theta, -1)                                                
        p = self.p_drop(p)
        out_p = F.softmax(self.D3_bn(self.D3(p)), -1)
        #out_p = [o.data for o in out_p]
        return out_p
        
    def forward(self, x, y):
        muV, logv_V = self.E1(x)
        muW, logv_W = self.E2(y)
        zV = self.reparametrize(muV, logv_V)
        zW = self.reparametrize(muW, logv_W)
        out_notes = self.decode_notes(zV, zW)
        #out_notes = [o.data for o in out_notes]
        out_text = self.decode_text(zV, zW)
        #out_text = [o.data for o in out_text]
        #computing the Euclidean distance(s)
        #cp = torch.mm(V, W.transpose(0,1)) 
        #dV = torch.norm(V, dim = 1).reshape(-1,1)
        #dV = dV.expand_as(cp)
        #dW = torch.norm(W.transpose(0,1), dim = 0)
        #dW = dW.expand_as(cp)
        #out = self.decoder(dV - 2*cp + dW)
        return out_notes, out_text, muV, logv_V, muW, logv_W, self.log_vareps      
                
# loss function
def lossf(targetN, targetT, outN, outT, muV, muW, logv_V, logv_W, log_vareps):
    
        SV = torch.exp(logv_V)
        SW = torch.exp(logv_W)
        
        # ** main loss component (notes)
        vareps = torch.exp(log_vareps)
        if device == "cuda":
            vareps = vareps.to("cuda")
        
        MCN = (0.5/vareps)*(targetN-outN)*(targetN-outN)
        MCN = torch.sum(MCN + 0.5*log_vareps.expand_as(targetN))

        #print(' current vareps: {}'.format(vareps))
        #MC = loss(out, target)
        #print(MC)
        
        # ** main loss component (text)
        MCT  = - torch.sum(targetT * torch.log(outT+1e-40))
        #MCT = 0
        
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
            
        var_division2    = SW / v2
        diff2            = muW - m2
        diff_term2       = diff2 * diff2 / v2
        logvar_division2 = log_v2 - logv_W
                
        KLw = 0.5 * ( torch.sum(var_division2 + diff_term2 + logvar_division2 - scale_factor2))
        
        return  MCN + MCT + 2*KLv + 2*KLw   
    
def vlossf(targetN, outN):
        MCN = (targetN-outN)*(targetN-outN)
        MCN = torch.sum(MCN)
        return torch.sqrt(MCN/len(targetN))

itA = iter(Rdload)
itB = iter(Cdload)
inA = itA.next()
inB = itB.next()

itN = iter(Ndload)
itT = iter(Tdload)
inN = itN.next()
inT = itT.next()

# the models
Mod = Decoder()
if device=="cuda":
    Mod = Mod.cuda()
optimizer = optim.Adam(Mod.parameters(), lr=2e-3, betas=(0.99, 0.999))
#optimizer = optim.Adam(list(Mod.parameters())+list(Mod.E1.parameters())+list(Mod.E2.parameters()), lr=2e-3, betas=(0.99, 0.999))


def train(epoch):    
    Mod.train()
    # Mod.E1.train()
    # Mod.E2.train()
    optimizer.zero_grad()  
      
    out_notes, out_text, muV, logv_V, muW, logv_W, log_vareps = Mod.forward(inA, inB)
    # computing the loss on train dataset
    loss = lossf(inN[store[0], store[1]], inT, out_notes[store[0], store[1]], out_text, muV, muW, logv_V, logv_W, log_vareps)
    #loss = test_loss(inA, muV, muW)
    loss.backward()
    optimizer.step()
    Mod.eval()
    # validation RMSE loss
    vloss = vlossf(inN[val_pos[0], val_pos[1]], out_notes[val_pos[0], val_pos[1]])
    if epoch % 100 == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)/(len(inA)*len(inA.transpose(0,1)))))
        print('Validation RMSE {:.6f}'.format(vloss))
    del out_notes, out_text

for epoch in range(epochs):
    train(epoch)    
    

# test (not exactly...)
Mod.eval()

outN, outT, muV, logv_V, muW, logv_W, log_vareps = Mod.forward(inA, inB)

outN = outN.reshape(M, P)
noutN = outN.cpu().data.numpy()

del outN, outT

# observed
onout = noutN[store[0], store[1]]
oinN = dat[store[0], store[1]]

# missing
mnout = noutN[test_pos[0], test_pos[1]]
minN = dat[test_pos[0], test_pos[1]]


#out = Mod.decoder(torch.mm(muV, muW.transpose(0,1)))

muV = muV.cpu().data.numpy()
muW = muW.cpu().data.numpy().T


#est = np.matmul(muV, muW)

from  sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt

# eclr = KMeans(n_clusters = 2).fit(noutN)
# eclc = KMeans(n_clusters = 2).fit(noutN.T)

# print(" ARI (rows):{} ".format(ari(clr, eclr.labels_)))
# print(" ARI (cols):{} ".format(ari(eclc.labels_, clc)))

# ###
# print(" The estimated eta square is: {}".format(torch.exp(log_vareps)))


### 
#colR = []
#for idx in range(len(clr)):
#    if clr[idx] == 0:
#        colR.append('red')
#    else:
#        colR.append('blue')
#
###
#colC = []
#for idx in range(len(clc)):
#    if clc[idx] == 0:
#        colC.append('yellow')
#    else:
#        colC.append('green')
#
#from sklearn.decomposition import PCA       
#pca = PCA(n_components = 2) 
#o1 = pca.fit(muV).fit_transform(muV)
#o2 = pca.fit(muW.transpose()).fit_transform(muW.transpose())
#        
#f, ax = plt.subplots(1)
#ax.scatter(o1[:,0], o1[:,1], color = colR)
#ax.scatter(o2[:,0], o2[:,1], color = colC)
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
emnout = np.round(mnout)
eonout = np.round(onout)
from sklearn.metrics import accuracy_score
print('Fulldec_miss_accuracy: {}'.format(accuracy_score(emnout, minN))) # missing
print('Fulldec_obs_accuracy: {}'.format(accuracy_score(eonout, oinN))) # observed

### RMSE on the test dataset
rmse = np.sqrt(np.sum( (mnout - minN)**2  )/len(mnout))
print( 'Our model (test data) RMSE: {}'.format(rmse))
    
 

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
