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
#from torch.autograd import Variable
import time
import pickle
import torch.cuda
from scipy import sparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0')

#device = "cpu"
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

############# Auxiliary Functions #############################################

def ipos(M, P, idx):
    return np.arange(idx*P, idx*P + P)

def jpos(M, P, idy):
    return np.arange(idy, idy + (M*P), P)

###############################################################################
# loading Amazon notes    
#Y = np.genfromtxt('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/amazon/AmazonOrdinal.csv', 
#                  skip_header=1, delimiter=",")
Y = np.genfromtxt('AmazonOrdinal.csv', skip_header=1, delimiter=",")
dat = Y[:,1:]
M = dat.shape[0] # 1634
P = dat.shape[1] # 1733
clr = np.random.choice(range(2), M) 
clc = np.random.choice(range(2), P)

############## loading and manipulatind docs and vocabulary  ###############   
nonzero_pos = np.where(dat != 0)
nonzero_row = nonzero_pos[0].tolist() # len=32836
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
    if np.random.uniform() < 1/4:    # validation
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
store1 = np.where(dat_ != 0)
store2 = np.where(dat_.T != 0)
# validation and test positions
val_pos = (ipv, jpv)
test_pos = (ipt, jpt)
print(len(store1[0]), len(val_pos[0]), len(test_pos[0])) # number for train, val and test

pos1 = np.vstack((store1[0], store1[1]))
pos1 = pos1.transpose()  # no zero position for user
pos2 = np.vstack((store2[0], store2[1]))
pos2 = pos2.transpose()  # no zero position for product

############## loading and manipulatind docs and vocabulary  ###############
import csv

#with open('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/amazon/vocabulary.csv', newline='') as csvfile:
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
#dtm = np.genfromtxt('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/amazon/dtm.csv', 
#                    skip_header=1, delimiter=",") 
dtm = np.genfromtxt('dtm.csv', skip_header=1, delimiter=",") 
dtm = np.asarray(dtm, dtype='float32') 

cdtm = np.zeros(shape = (len(nonzero_row), V), dtype='float32')
for index in range(dtm.shape[0]):
    cdtm[int(dtm[:,0][index])][int(dtm[:,1][index])] = dtm[:,2][index]
    
def get_length_user(end):
    len_doc = 0
    for idx in range(end):
        len_doc = len_doc + len(np.where(pos1[:, 0] == idx)[0])
    return len_doc

# def get_length_product(end):
#     len_doc = 0
#     for idx in range(end):
#         len_doc = len_doc + len(np.where(pos2[:, 0] == idx)[0])
#     return len_doc

def cdtm_user(start, end):
    cdtm_u = np.zeros(((end-start)*P, V), dtype='float32')
    i = get_length_user(start)  ### i = len(all precedent user records)
    #print(i)
    for idx in range(start, end):
        idx_u = pos1[np.where(pos1[:, 0] == idx)].tolist()  # products by i-th user
        #print(idx_u)
        for item in idx_u:
            #print(item)
            cdtm_u[(item[1]+(idx-start)*P), :] = cdtm[i, :]
            i = i + 1
        #print(i)
    return cdtm_u

def cdtm_product(start, end):
    cdtm_p = np.zeros(((end-start)*M, V), dtype='float32')
    #i = get_length_product(start)  ### i = len(all precedent product records)
    for idx in range(start, end):
        idx_p = pos2[np.where(pos2[:, 0] == idx)].tolist()  # users buy j-th product
        #print(idx_p)
        for item in idx_p:
            #print(item)
            cdtm_p[(item[1]+(idx-start)*M), :] = cdtm_user(item[1], item[1]+1)[idx, :]
            #i = i + 1
        #print(i)
    return cdtm_p

# num_doc_m = []
# for num in range(M):
#     num_doc_m.append(len((np.where(dat_[num] != 0))[0]))
#print(num_doc_m)

# ** individuals
idtm = np.zeros(shape = (M, V))
i = 0
for idx in range(M):
    pos = np.arange(i, i + len((np.where(pos1[:,0] == idx))[0]))
    idtm[idx, :] = cdtm[pos, :].sum(0)
    i = i + len((np.where(pos1[:,0] == idx))[0])

# num_doc_p = []
# for num in range(P):
#     dat_trans= dat_.T
#     num_doc_p.append(len((np.where(dat_trans[num] != 0))[0]))
#print(num_doc_p)
    
# ** object
odtm = np.zeros(shape = (P, V))
j = 0 
for idx in range(P):
    pos = np.arange(j, j + len((np.where(pos2[:,0] == idx))[0]))
    odtm[idx, :] = cdtm[pos, :].sum(0)
    j = j + len((np.where(pos2[:,0] == idx))[0])
    
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
#Tdata = MDataset(cdtm, transform = True)   # The complete dtm ( one row --> (i,j) ) to reconstruct 
Rdata = MDataset(X, transform = True)      # The individual/input matrix
Cdata = MDataset(Y, transform = True)      # The object/input matrix


## Loading all datasets
batch_size = 10

Ndload = DataLoader(Ndata,
                    batch_size = batch_size,
                    shuffle = False
                    )
# Tdload = DataLoader(Tdata,
#                     batch_size = len(cdtm),
#                     shuffle = False
#                     )
Rdload = DataLoader(Rdata,
                    batch_size = batch_size, 
                    shuffle = False
                    )  
Cdload = DataLoader(Cdata,
                    batch_size = batch_size,
                    shuffle = False
                    )


# Global parameters
init_dim_R = (P + V)
init_dim_C = (M + V)
mid_dim = 50
mid_dim_out = 80

another_dim = 80
nb_of_topics = 50

int_dim = 50
# prior_variance = 0.5
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
        self.en2 = nn.Linear(mid_dim, mid_dim)
        self.en2_drop = nn.Dropout(0.2)        
        #self.int = nn.Linear(mid_dim, 1)               # one coeff. for each row
        self.mu = nn.Linear(mid_dim, int_dim)
        self.mu_bn  = nn.BatchNorm1d(int_dim) 
        self.logv = nn.Linear(mid_dim, int_dim)
        self.logv_bn  = nn.BatchNorm1d(int_dim) 
    
    def encode(self, x):
        h1 = F.softplus(self.en1(x))
        h2 = self.en2_drop(F.softplus(self.en2(h1)))
        mu = self.mu_bn(self.mu(h2))
        logv =self.logv_bn(self.logv(h2))
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
          self.p_drop = nn.Dropout(0.2)
          # decoding layers: notes
          self.D1 = nn.Linear(init_dim_R, mid_dim_out)
          self.D2 = nn.Linear(mid_dim_out, init_dim_R)
          # decoding layers: text
          self.newD1 = nn.Linear(2*int_dim, another_dim)
          self.newD2 = nn.Linear(another_dim, nb_of_topics)
          self.D3 = nn.Linear(nb_of_topics, V, bias = True)   # beta             
          self.D3_bn = nn.BatchNorm1d(V, affine = False)            
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
#### to solve   out = self.D2(F.relu(self.D1(val))) # dec 
        #out = [o.data for o in out]
        return out
    
    def decode_text(self, zV, zW):
        s1 = zV.shape[0]
        #print('s1',s1)
        s2 = zW.shape[0]
        #print('s2',s2)
        #s3 = zV.shape[1]
        theta = torch.zeros(size = (s1*s2, nb_of_topics))
        if device == 'cuda':
            theta = theta.to(device)
        for idx in range(s1):
            start = idx*s2
            in_dat = torch.cat((zV[idx,:].expand_as(zW), zW), 1)
            #print(in_dat.shape)
            #in_dat = torch.cat((zV[idx,:].expand_as(zW), zW), 0)           
            theta[start:(start + s2),:] = self.newD2((F.relu(self.newD1(in_dat))))
            #theta[start:(start + s2),:] = F.relu(self.newD3(in_dat))
        p = F.softmax(theta, -1)                                                
        p = self.p_drop(p)
        out_p = F.softmax(self.D3_bn(self.D3(p)), -1)
        #out_p = [o.data for o in out_p]
        return out_p
    
    def forward(self, x, in_row = True):
        if in_row == True:
            muV, logv_V = self.E1(x)
            # muW, logv_W = self.E2(torch.transpose(x,0,1))
            V = self.reparametrize(muV, logv_V)
            W = self.reparametrize(store_muW, store_logv_W)
            out_notes = self.decode_notes(V, W)
            out_text = self.decode_text(V, W)
            return out_notes, out_text, muV, logv_V, self.log_vareps 

        else:
            muW, logv_W = self.E2(x)
            V = self.reparametrize(store_muV, store_logv_V)
            W = self.reparametrize(muW, logv_W)
            out_notes = self.decode_notes(W, V)
            out_text = self.decode_text(W, V)
            return out_notes, out_text, muW, logv_W, self.log_vareps
        
             
# loss function
def lossf(targetN, targetT, outN, outT, mu, logv, log_vareps, in_row = True):
    
    S = torch.exp(logv)   
    vareps = torch.exp(log_vareps)
    if device == "cuda":
        vareps = vareps.to("cuda")  
        
    # ** main loss component (notes)
    MCN = (0.5/vareps)*(targetN-outN)*(targetN-outN)
    #print(MCN.data.numpy().shape)
    MCN = torch.sum(MCN + 0.5*log_vareps.expand_as(targetN))
    #print(targetN.shape, outN.shape)

    # ** main loss component (text)
    MCT  = - torch.sum(targetT * torch.log(outT+1e-40))
    #print(targetT.shape, outT.shape)

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
        
    m = m.expand_as(mu)
    v = v.expand_as(S)     
    log_v = log_v.expand_as(logv)  
    scale_factor=scale_factor.expand_as(mu)   
    
    var_division    = S / v
    diff            = mu - m
    diff_term       = diff * diff / v
    logvar_division = log_v - logv
            
    KL = 0.5 * (torch.sum(var_division + diff_term + logvar_division - scale_factor))

    if in_row == True:
        correct = batch_size/M
        return  correct*(MCN + MCT + 2*KL) 
    else:
        correct = batch_size/P  
        return  correct*(MCN + MCT + 2*KL) 

    
def vlossf(targetN, outN):
    MCN = (targetN-outN)*(targetN-outN)
    MCN = torch.sum(MCN)
    return torch.sqrt(MCN/len(targetN))

itA = iter(Rdload)
itB = iter(Cdload)
inA = itA.next()
inB = itB.next()

itN = iter(Ndload)
#itT = iter(Tdload)
inN = itN.next()
#inT = itT.next()

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
        #print(batch_idx, obs.cpu().data.numpy().shape)
        interval = torch.LongTensor(range(V,P+V))
        if device=="cuda":
            interval = interval.to(device)
        obs_note = torch.index_select(obs, 1, interval) # la partie de note
        # print(obs_note.numpy().shape) # (15,60)
        # interval2 = torch.LongTensor(range(0,V))
        # if device=="cuda":
        #     interval2 = interval2.to(device)
        # obs_text = torch.index_select(obs, 1, interval2) # la partie de text
        #print(obs_text.numpy().shape) # (15, 1034)       
        out_notes, out_text, muV, logv_V, log_vareps = Mod.forward(obs, in_row = True)
        print(out_notes.data.numpy().shape, out_text.data.numpy().shape) # (15, 60), (900, 1034)
        
        if (batch_idx+1)*batch_size < M:
            #idx_text = range(batch_idx*batch_size, (batch_idx+1)*batch_size)
            b = batch_size
        else:    # si pas divisible
            #idx_text = range(batch_idx*batch_size, M)
            b = M - batch_idx*batch_size  # la reste
        #print(idx_text)
        #out_text1 = torch.index_select(out_text, 0, torch.LongTensor(idx_text))
        # print(out_text1.data.numpy().shape) # (15, 1034)        
        #s = out_text.data.numpy().shape[0]
        
        obs_text1 = torch.zeros(out_text.shape, dtype=torch.float32)
        #for idx in range(b):
        cdtm_u = cdtm_user(batch_idx*b, (batch_idx+1)*b)
        # print('cdtm_u', cdtm_u.shape)
        # coo = sparse.coo_matrix(cdtm_u)
        # values = coo.data
        # indices = np.vstack((coo.row, coo.col))
        # indices = indices/1

        # i = torch.LongTensor(indices)
        # v = torch.FloatTensor(values)
        # shape = coo.shape

        # obs_text1 = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
        obs_text1 = torch.tensor(cdtm_u, dtype=torch.float32)
        if device=="cuda":
            obs_text1 = obs_text1.to(device)   
        #print(obs_text1.numpy().shape)
        #obs_text1 = obs_text.expand_as(out_text)
        loss = lossf(obs_note[obs_note!=0], obs_text1, out_notes[obs_note!=0], out_text, 
                     muV, logv_V, log_vareps, in_row = True)
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
    
    if epoch % 5 == 0:
        print('Train Epoch: {} \tRLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)/(len(inA)*len(inA.transpose(0,1)))))
    
    ######################
    ##  MB on columns   ##
    ######################    
    Mod.train()
    optimizer.zero_grad() 
    for batch_idx, obs in enumerate(Cdload):
        #print(batch_idx, obs.numpy().shape)
        # idx = batch_idx
        # print(idx)
        interval = torch.LongTensor(range(V,M+V))
        if device=="cuda":
            interval = interval.to(device)
        obs_note = torch.index_select(obs, 1, interval) # la partie de note
        #print(obs_note.numpy().shape) # (15,75)
        # interval2 = torch.LongTensor(range(0,V))
        # if device=="cuda":
        #     interval2 = interval2.to(device)
        # obs_text = torch.index_select(obs, 1, interval2) # la partie de text
        #print(obs_text.numpy().shape) # (15, 1034) 
        out_notes, out_text, muW, logv_W, log_vareps = Mod.forward(obs, in_row = False)
        # print(out_notes.data.numpy().shape, out_text.data.numpy().shape) # (15, 75), (1125, 1034)
        if (batch_idx+1)*batch_size < P:
            #idx_text = range(batch_idx*batch_size, (batch_idx+1)*batch_size)
            b = batch_size
        else:
            #idx_text = range(batch_idx*batch_size, P)
            b = P - batch_idx*batch_size
        #print(idx_text)
        #out_text1 = torch.index_select(out_text, 0, torch.LongTensor(idx_text))
        #s = out_text.data.numpy().shape[0]
        #print(out_text1.data.numpy().shape) # (15, 1034)
        
        obs_text1 = torch.zeros(out_text.shape, dtype=torch.float32)
        #for idx in range(b):
            # obs_text1[idx*M:(idx+1)*M, :] = obs_text[idx,:]
        cdtm_p = cdtm_product(batch_idx*b, (batch_idx+1)*b)
        # print('cdtm_p', cdtm_p.shape)
        # coo = sparse.coo_matrix(cdtm_p)
        # values = coo.data
        # indices = np.vstack((coo.row, coo.col))
        # indices = indices/1

        # i = torch.LongTensor(indices)
        # v = torch.FloatTensor(values)
        # shape = coo.shape
        
        # obs_text1 = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
        obs_text1 = torch.tensor(cdtm_p, dtype=torch.float32)
        if device=="cuda":
            obs_text1 = obs_text1.to(device) 

        loss = lossf(obs_note[obs_note!=0], obs_text1, out_notes[obs_note!=0], out_text, 
                     muW, logv_W, log_vareps, in_row = False)
        loss.backward()
        optimizer.step()
        
    ##############################################################
    ## Updating variational parameters: store_muW, store_logv_W ##                   
    ##############################################################
    Mod.eval()
    inY = torch.tensor(Y, dtype = torch.float32)
    if device=="cuda":
        inY = inY.to(device)
    store_muW_, store_logv_W_ = Mod.E2(inY)
    store_muW = store_muW_.detach()
    store_logv_W = store_logv_W_.detach()

    if epoch % 5 == 0:
        print('Train Epoch: {} \tCLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)/(len(inA)*len(inA.transpose(0,1)))))

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
    #print(out.shape)
    #out = out.cpu().data.numpy()
    indat = torch.Tensor(dat)
    if device=="cuda":
         indat = indat.to(device)
    perdita = vlossf(indat[val_pos], out[val_pos])
    store_vloss[epoch] = perdita
    if epoch % 5 == 0:
        print('Validation RMSE {:.6f}'.format(perdita))   
    
end = time.process_time()
print(end-start)    

## test
Mod.eval()
store_muV, store_logv_V, store_muW, store_logv_W = train(epoch, store_muV, store_logv_V, store_muW, store_logv_W) 
out1 = torch.mm(store_muV, store_muW.transpose(0,1))
out1 = out1.cpu().data.numpy()

mnout = out1[test_pos[0], test_pos[1]]
minA = dat[test_pos[0], test_pos[1]]

rmse = np.sqrt(np.sum( (mnout - minA)**2  )/len(mnout))
print( 'Our model RMSE: {}'.format(rmse))

from sklearn.metrics import accuracy_score
emnout = np.round(mnout)
print(" missing accuracy score:{} ".format(accuracy_score(emnout, minA)))

# muV = store_muV
# muW = store_muW.transpose(0,1)

# from  sklearn.metrics import adjusted_rand_score as ari
# from sklearn.cluster import KMeans
# #import matplotlib.pyplot as plt

# eclr = KMeans(n_clusters = 2).fit(out1)
# eclc = KMeans(n_clusters = 2).fit(out1.T)

# # print(ari(clr, eclr.labels_))
# # print(ari(eclc.labels_, clc))
 
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
        
# # plt.plot(store_vloss, color = 'red') 
        
# # f, ax = plt.subplots(1)
# # ax.scatter(muV[:,0], muV[:,1], color = colR)
# # ax.scatter(muW[0,:], muW[1,:], color = colC)

# #emnout = np.round(mnout)
# #eonout = np.round(onout)
# #from sklearn.metrics import accuracy_score
# #accuracy_score(emnout, minA)
# #accuracy_score(eonout, oinA)

# # ##
# # print(" The estimated eta square is: {}".format(torch.exp(log_vareps)))
# # ## RMSE on the test dataset
# # rmse = np.sqrt(np.sum( (mnout - minA)**2  )/len(mnout))
# # print( 'Our model RMSE: {}'.format(rmse))
# print(' ARI in row: {}'.format(ari(clr, eclr.labels_)))
# print(' ARI in col: {}'.format(ari(eclc.labels_, clc)))
