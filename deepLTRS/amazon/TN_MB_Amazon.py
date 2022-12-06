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
import torch.cuda
import pickle

begin = time.time() 

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
# loading Amazon notes and texts   
#dat = np.load("C:/Users/Dingge/Doctoral_projets/Pytorch/sim_data_notes_musical.npy")
dat = np.load("sim_data_notes_musical.npy")
#dat = np.load("sim_data_notes_amazon.npy")
M,P = dat.shape # (1643, 1733)

cdtm = np.load("C:/Users/Dingge/Doctoral_projets/Pytorch/cdtm_musical.npy")
cdtm_ = cdtm.copy()
cdtm_col = np.load("cdtm_col_musical.npy")
# cdtm = np.load("C:/Users/Dingge/Doctoral_projets/Pytorch/new_cdtm_amazon.npy") # (32811, 5743)
# cdtm_col = np.load("C:/Users/Dingge/Doctoral_projets/Pytorch/new_cdtm_col_amazon.npy")
# cdtm = np.load("new_cdtm_amazon_bigram.npy")
# cdtm_col = np.load("new_cdtm_col_amazon_bigram.npy")
# cdtm = np.load("new_cdtm_amazon.npy") # (32811, 5743) stop words
# cdtm_col = np.load("new_cdtm_col_amazon.npy")  # (32811, 5743) stop words
V = cdtm.shape[1] # 5743
small = np.sum(cdtm_, axis=0)
stock = np.where(np.where(small<3)[0])
reste = np.delete(cdtm_, stock, axis = 1)

############## loading and manipulatind docs and vocabulary  ###############   
nonzero_pos = np.where(dat != 0)
nonzero_row = nonzero_pos[0].tolist()
nonzero_col = nonzero_pos[1].tolist()

nonzero_pos_P = np.where(dat.T != 0)
nonzero_row_P = nonzero_pos_P[0].tolist()
nonzero_col_P = nonzero_pos_P[1].tolist() 

dat_ = dat.copy()
## inserting missing values
val = 0.0
import random
# we are storing the positions of validation and test entries!
ipv = jpv = ipt = jpt = np.zeros(shape = (1,1))
# random.seed(0)
# np.random.seed(0)
ix = [(nonzero_row[idx], nonzero_col[idx]) for idx in range(len(nonzero_row))] # All nozero positions
for row, col in random.sample(ix, int(round(0.2*len(ix)))): # 20 % for val and test
    if np.random.uniform() < 1/2:    # half for validation = 10 %
        ipv = np.vstack((ipv, row))
        jpv = np.vstack((jpv, col))
        dat_[row, col] = 0
    else:   # half for test = 10 %
        ipt = np.vstack((ipt, row))
        jpt = np.vstack((jpt, col))
        dat_[row, col] = 0    
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

# # For CDL
# train_cdl = np.zeros((len(store1[0]),3))
# for i in range(len(store1[0])):
#     train_cdl[i][0] = store1[0][i]
#     train_cdl[i][1] = store1[1][i]
#     train_cdl[i][2] = dat[store1[0][i]][store1[1][i]]
# np.save('C:/Users/Dingge/Doctoral_projets/Pytorch/train_CDL2', dat_)

# test_cdl = np.zeros((len(test_pos[0]),3))
# for i in range(len(test_pos[0])):
#     test_cdl[i][0] = test_pos[0][i]
#     test_cdl[i][1] = test_pos[1][i]
#     test_cdl[i][2]= dat[test_pos[0][i]][test_pos[1][i]]
# test_cdl2 = np.zeros((M,P))
# for i in range(len(test_pos[0])):
#     row = test_pos[0][i]
#     col = test_pos[1][i]
#     test_cdl2[row][col]= dat[row][col]
# for j in range(len(val_pos[0])):
#     row = val_pos[0][j]
#     col = val_pos[1][j]
#     test_cdl2[row][col]= dat[row][col]   
# np.save('C:/Users/Dingge/Doctoral_projets/Pytorch/test_CDL2', test_cdl2)    
    
############################################################################    
## Now I need to create two dtms (individual specific and object specific..)   
def get_length_user(end):
    len_doc = 0
    for idx in range(end):
        len_doc = len_doc + len(np.where(pos1[:, 0] == idx)[0])
    return len_doc

def get_length_product(end):
    len_doc = 0
    for idx in range(end):
        len_doc = len_doc + len(np.where(pos2[:, 0] == idx)[0])
    return len_doc

############# Wu ##############
Wu = np.zeros((len(store1[0]), ), dtype = 'int32')  # (3492, )
for row in range(len(store1[0])):
    Wu[row,] = pos1[row][0] * P + pos1[row][1]   
    
Wu_col = np.zeros((len(store1[0]), ), dtype = 'int32')  # (3492, )
for row in range(len(store1[0])):
    Wu_col[row,] = pos2[row][0] * M + pos2[row][1]  
    
############# I ##############    
I = np.zeros((M, 2), dtype = 'int32')
for idx in range(M):
    I[idx][0] = idx
    I[idx][1] = get_length_user(idx)
    
I_col = np.zeros((P, 2), dtype = 'int32')
for idx in range(P):
    I_col[idx][0] = idx
    I_col[idx][1] = get_length_product(idx)

# ** individuals
idtm = np.zeros(shape = (M, V))
i = 0
for idx in range(M):
    pos = np.arange(i, i + len((np.where(pos1[:,0] == idx))[0]))
    idtm[idx, :] = cdtm[pos, :].sum(0)
    i = i + len((np.where(pos1[:,0] == idx))[0])
    
# ** object
odtm = np.zeros(shape = (P, V))
j = 0 
for idx in range(P):
    pos = np.arange(j, j + len((np.where(pos2[:,0] == idx))[0]))
    odtm[idx, :] = cdtm_col[pos, :].sum(0)
    j = j + len((np.where(pos2[:,0] == idx))[0])
    
# l = list()    
# for i in range(P):
#     l.append(odtm[i])
# np.save('C:/Users/Dingge/Doctoral_projets/Pytorch/item_info', np.array(l))  # For CDL
    
## We need to cbind idtm-dat and odtm-t(dat)
X = np.concatenate((idtm, dat_), axis = 1) # (1643, 7476)
Y = np.concatenate((odtm, dat_.transpose()), axis = 1) # (1733, 7386)
print('OK')

############# mean of users and products ###############
mean_u = np.zeros((M,1))
for u in range(M):
    nz = len(np.where(dat[u,:]!=0)[0])
    mean_u[u,:] = dat[u,:].sum()/float(nz)
if device=="cuda":
    mean_u = mean_u.to(device)
    
mean_p = np.zeros((P,1))
for p in range(P):
    nz = len(np.where(dat.T[p,:]!=0)[0])
    mean_p[p,:] = dat.T[p,:].sum()/float(nz)
if device=="cuda":
    mean_p = mean_p.to(device)    
    
# mean_u = dat.sum(axis=1)/P
# mean_u = np.asarray(mean_u).reshape(-1,1)
# if device=="cuda":
#     mean_u = mean_u.to(device)
# mean_p = dat.T.sum(axis=1)/M
# mean_p = np.asarray(mean_p).reshape(-1,1)
# if device=="cuda":
#     mean_p = mean_p.to(device)    
    
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
Rdata = MDataset(X, transform = True)      # The individual/input matrix
Cdata = MDataset(Y, transform = True)      # The object/input matrix

## Loading all datasets
batch_size_R = 10
batch_size_C = 6

Rdload = DataLoader(Rdata,
                    batch_size = batch_size_R, 
                    shuffle = False
                    )  
Cdload = DataLoader(Cdata,
                    batch_size = batch_size_C,
                    shuffle = False
                    )

# Global parameters
init_dim_R = (P + V)
init_dim_C = (M + V)
mid_dim = 50
mid_dim_out = 80 

another_dim = 80
nb_of_topics = 150

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
          # decoding layers: text
          self.newD1 = nn.Linear(2*int_dim, another_dim)
          self.newD2 = nn.Linear(another_dim, nb_of_topics)
          self.D3 = nn.Linear(nb_of_topics, V, bias = True)   # beta             
          self.D3_bn = nn.BatchNorm1d(V, affine = False)            
          self.log_vareps = nn.Parameter(torch.randn(1))
          self.b_u = nn.Parameter(torch.Tensor(mean_u))
          self.b_p = nn.Parameter(torch.Tensor(mean_p))
          
    def reparametrize(self, m, log_v):
        std = torch.exp(0.5*log_v)
        eps = torch.randn_like(std)
        if device == 'cuda':
            eps = eps.to(device)        
        return m + eps*std

    def decode_notes(self, zV, zW, bu, bp):
        val = torch.mm(zV, zW.transpose(0,1))
        #out = torch.mm(F.relu(zV), F.relu(zW.transpose(0,1)))
        bias = torch.mm(bu, bp.transpose(0,1))
        out = val + bias
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
    
          
    def forward(self, x, idx, in_row = True):
        if in_row == True:
            muV, logv_V = self.E1(x)
            # muW, logv_W = self.E2(torch.transpose(x,0,1))
            V = self.reparametrize(muV, logv_V)
            W = self.reparametrize(store_muW, store_logv_W)
            if (idx+1)*batch_size_R < M:
                b = batch_size_R 
                b_u = self.b_u[idx*b:(idx+1)*b,:]
            else:    # si pas divisible
                b = M - idx*batch_size_R   # le reste
                b_u = self.b_u[idx*batch_size_R:M,:]
            b_p = store_b_p 
            out_notes = self.decode_notes(V, W, b_u, b_p)
            out_text = self.decode_text(V, W)
            return out_notes, out_text, muV, logv_V, self.log_vareps, self.b_u 

        else:
            muW, logv_W = self.E2(x)
            V = self.reparametrize(store_muV, store_logv_V)
            W = self.reparametrize(muW, logv_W)
            if (idx+1)*batch_size_C < P:
                b = batch_size_C 
                b_p = self.b_p[idx*b:(idx+1)*b,:]
            else:    # si pas divisible
                b = P - idx*batch_size_C   # le reste
                b_p = self.b_p[idx*batch_size_C:P,:]
            b_u = store_b_u 
            out_notes = self.decode_notes(W, V, b_p, b_u)
            out_text = self.decode_text(W, V)
            return out_notes, out_text, muW, logv_W, self.log_vareps, self.b_p
        
             
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
        correct = batch_size_R/M
        return  correct*(MCN + MCT + KL) 
    else:
        correct = batch_size_C/P  
        return  correct*(MCN + MCT + KL) 

    
def vlossf(targetN, outN):
    MCN = (targetN-outN)*(targetN-outN)
    MCN = torch.sum(MCN)
    return torch.sqrt(MCN/len(targetN))

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
store_b_u = torch.Tensor(M, 1).fill_(1.0)  # initialization
store_b_p = torch.Tensor(P, 1).fill_(1.0)

############# Wb #############
def getWb(I, Wu, cdtm, batch_size, batch_idx):
    Wb = np.zeros((batch_size * P, V))
    i = batch_idx * batch_size
    idx = I[i, 1]
    while Wu[idx] < (i + batch_size) * P:
        pos = Wu[idx] - i * P
        Wb[pos, :] = cdtm[idx, :]
        idx = idx + 1
        if  idx == len(Wu):
            break
        #print(idx)
    return Wb

def getWb_col(I_col, Wu_col, cdtm_col, batch_size, batch_idx):
    Wb = np.zeros((batch_size * M, V))
    j = batch_idx * batch_size
    idx = I_col[j, 1]
    while Wu_col[idx] < (j + batch_size) * M:
        pos = Wu_col[idx] - j * M
        Wb[pos, :] = cdtm_col[idx, :]
        idx = idx + 1
        if  idx == len(Wu_col):
            break            
        #print(idx, j, pos)
    return Wb

# the model
Mod = Decoder()
if device=="cuda": 
    Mod = Mod.cuda()
optimizer = optim.Adam(Mod.parameters(), lr=2e-3, betas=(0.99, 0.999))
#optimizer = optim.Adam(Mod.parameters(), lr=2e-4, betas=(0.99, 0.999), weight_decay=0.05) # weight decay
#optimizer = optim.Adagrad(Mod.parameters(), lr=2e-3, lr_decay=0.995)
#optimizer = optim.Adam(list(Mod.parameters())+list(Mod.E1.parameters())+list(Mod.E2.parameters()), lr=2e-3, betas=(0.99, 0.999))

def train(epoch, store_muV, store_logv_V, store_muW, store_logv_W, store_b_u, store_b_p):    
    Mod.train()
    ######################
    ## MB on rows first ##
    ######################
    batch_size = batch_size_R
    for batch_idx, obs in enumerate(Rdload):
        #print('batch_idx_R: ', batch_idx)
        optimizer.zero_grad() 
        interval = torch.LongTensor(range(V,P+V))
        if device=="cuda":
            interval = interval.to(device)
        obs_note = torch.index_select(obs, 1, interval) # la partie de note     
        out_notes, out_text, muV, logv_V, log_vareps, b_u = Mod.forward(obs, batch_idx, in_row = True)
        
        if (batch_idx+1)*batch_size < M:
            b = batch_size
        else:    # si pas divisible
            b = M - batch_idx*batch_size  # la reste  
            
        cdtm_u = getWb(I, Wu, cdtm, b, batch_idx)
        obs_text1 = torch.tensor(cdtm_u, dtype=torch.float32)
        if device=="cuda":
            obs_text1 = obs_text1.to(device)  
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
    store_b_u_ = b_u
    store_b_u = store_b_u_.detach() 
    
    if epoch % 1 == 0:
        print('Train Epoch: {} \tRLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)/(len(inA)*len(inA.transpose(0,1)))))
    
    ######################
    ##  MB on columns   ##
    ######################    
    Mod.train()
    batch_size = batch_size_C
    for batch_idx, obs in enumerate(Cdload):
        #print('batch_idx_C: ', batch_idx)
        optimizer.zero_grad() 
        interval = torch.LongTensor(range(V,M+V))
        if device=="cuda":
            interval = interval.to(device)
        obs_note = torch.index_select(obs, 1, interval) # la partie de note
        out_notes, out_text, muW, logv_W, log_vareps, b_p = Mod.forward(obs, batch_idx, in_row = False)
        
        if (batch_idx+1)*batch_size < P:
            b = batch_size
        else:
            b = P - batch_idx*batch_size
   
        cdtm_p = getWb_col(I_col, Wu_col, cdtm_col, b, batch_idx)
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
    store_b_p_ = b_p
    store_b_p = store_b_p_.detach() 

    if epoch % 1 == 0:
        print('Train Epoch: {} \tCLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)/(len(inB)*len(inB.transpose(0,1)))))

    return store_muV, store_logv_V, store_muW, store_logv_W, store_b_u, store_b_p
    
store_vloss = np.zeros(epochs)
all_muV = torch.zeros(size = [epochs, M, int_dim])
all_logv_V = torch.zeros(size = [epochs, M, int_dim])
all_muW = torch.zeros(size = [epochs, P, int_dim])
all_logv_W = torch.zeros(size = [epochs, P, int_dim])
all_b_u = torch.zeros(size = [epochs, M, 1])
all_b_p = torch.zeros(size = [epochs, P, 1])

start = time.time()
for epoch in range(epochs):
    store_muV, store_logv_V, store_muW, store_logv_W, store_b_u, store_b_p = train(epoch, store_muV, store_logv_V, store_muW, store_logv_W, store_b_u, store_b_p)    
    ###############
    # valid check #
    ###############
    out = torch.mm(store_muV, store_muW.transpose(0,1)) + torch.mm(store_b_u, store_b_p.transpose(0,1))
    #print(out.shape)
    #out = out.cpu().data.numpy()
    indat = torch.Tensor(dat)
    if device=="cuda":
         indat = indat.to(device)
    perdita = vlossf(indat[val_pos], out[val_pos]) 
    ########## store all the loss and output #########
    store_vloss[epoch] = perdita
    all_muV[epoch] = store_muV
    all_logv_V[epoch] = store_logv_V
    all_muW[epoch] = store_muW
    all_logv_W[epoch] = store_logv_W
    all_b_u[epoch] = store_b_u
    all_b_p[epoch] = store_b_p
    
    if epoch % 1 == 0:
        print('Validation RMSE {:.6f}'.format(perdita))   
    
end = time.time()
print('Train time', end-start)   

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
best_b_u = all_b_u[minidx]
best_b_p = all_b_p[minidx] 

# np.save('new_best_muV_musical', best_muV)
# np.save('new_best_muW_musical', best_muW)
# np.save('new_best_logv_V_musical', best_logv_V)
# np.save('new_best_logv_W_musical', best_logv_W )
# np.save('new_best_b_u', best_b_u)
# np.save('new_best_b_p', best_b_p)

# np.save('best_muV_bigram_5D_bs30', best_muV)
# np.save('best_muW_bigram_5D_bs30', best_muW)
# np.save('best_logv_V_bigram_5D_bs30', best_logv_V)
# np.save('best_logv_W_bigram_5D_bs30', best_logv_W )
       
## test
out_f = torch.mm(best_muV, best_muW.transpose(0,1)) + torch.mm(best_b_u, best_b_p.transpose(0,1))
out_f = out_f.cpu().data.numpy()

# test fitted values
est_out = out_f[test_pos[0], test_pos[1]]
print(est_out)
real_out = dat[test_pos[0], test_pos[1]]

rmse = np.sqrt(np.sum((est_out - real_out)**2  )/len(est_out))
print( 'Our model RMSE: {}'.format(rmse))

from sklearn.metrics import accuracy_score
est_out = np.round(est_out)
print(" Accuracy: ", accuracy_score(est_out, real_out))

gap = 1 - np.min(est_out)   ## Normalisation: c'est nécessaire 
print(" Gap: ", gap) 
# est_out += gap
# print(est_out)
# print(" Gap accuracy: ", accuracy_score(est_out, real_out))

#rmse = np.sqrt(np.sum((est_out - real_out)**2  )/len(est_out))
#print( 'Final True RMSE: {}'.format(rmse))

#abs_out = abs(est_out)  ## Abs value
#print(" Abs accuracy: ", accuracy_score(abs_out, real_out))
#rmse = np.sqrt(np.sum((abs_out - real_out)**2  )/len(abs_out))
#print( 'Abs RMSE: {}'.format(rmse))

last = time.time()
print("All the Time: ", last-begin)
print('best_b_u: ', best_b_u)

#torch.save(Mod.state_dict(), 'new_amazon_musical.pt')

#############################################################################
###################################
## Specific muV, muW, log_V, log_W ##
###################################
# beg = time.time()

# muV1 = best_muV[1618]
# logv_V1 = best_logv_V[1618] 
# muV2 = best_muV[1373] 
# logv_V2 = best_logv_V[1373] 
# muV3 = best_muV[202]
# logv_V3 = best_logv_V[202] 
# muV4 = best_muV[280] 
# logv_V4 = best_logv_V[280] 

# muW1 = best_muW[1571] 
# logv_W1 = best_logv_W[1571]
# muW2 = best_muW[1096] 
# logv_W2 = best_logv_W[1096]

# R1 = torch.zeros(size=(1, int_dim))
# R2 = torch.zeros(size=(1, int_dim))
# R3 = torch.zeros(size=(1, int_dim))
# R4 = torch.zeros(size=(1, int_dim))

# C1 = torch.zeros(size=(1, int_dim))
# C2 = torch.zeros(size=(1, int_dim))

# # R1[:,:] = torch.normal.multivariate(muV1, logv_V1)
# # R2[:,:] = torch.normal.multivariate(mean = muV2, std = logv_V2, size = int_dim)
# # C1[:,:] = torch.normal.multivariate(mean = muW1, std = logv_W1, size = int_dim)

# for i in range(int_dim):
#       R1[:,i] = torch.normal(mean = muV1[i], std = torch.exp(0.5*logv_V1[i]))
#       R2[:,i] = torch.normal(mean = muV2[i], std = torch.exp(0.5*logv_V2[i]))
#       R3[:,i] = torch.normal(mean = muV3[i], std = torch.exp(0.5*logv_V3[i]))
#       R4[:,i] = torch.normal(mean = muV4[i], std = torch.exp(0.5*logv_V4[i]))
#       C1[:,i] = torch.normal(mean = muW1[i], std = torch.exp(0.5*logv_W1[i]))
#       C2[:,i] = torch.normal(mean = muW2[i], std = torch.exp(0.5*logv_W2[i]))
   
# out, proba = Mod.decode_specific(R1, C1)
# list_proba = proba.cpu().data.numpy().T.tolist()
# out2, proba2 = Mod.decode_specific(R2, C1)
# list_proba2 = proba2.cpu().data.numpy().T.tolist()
# out3, proba3 = Mod.decode_specific(R3, C2)
# list_proba3 = proba3.cpu().data.numpy().T.tolist()
# out4, proba4 = Mod.decode_specific(R4, C2)
# list_proba4 = proba4.cpu().data.numpy().T.tolist()

# proba_dict = {}
# proba_dict2 = {}
# proba_dict3 = {}
# proba_dict4 = {}
# for index, item in enumerate(list_proba):
#    proba_dict[index] = item
# for index, item in enumerate(list_proba2):
#    proba_dict2[index] = item    
# for index, item in enumerate(list_proba3):
#     proba_dict3[index] = item 
# for index, item in enumerate(list_proba4):
#     proba_dict4[index] = item     
    
# sort_words = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
# print(sort_words[:10])
# sort_words2 = sorted(proba_dict2.items(), key=lambda x: x[1], reverse=True)
# print(sort_words2[:10])
# sort_words3 = sorted(proba_dict3.items(), key=lambda x: x[1], reverse=True)
# print(sort_words3[:10])
# sort_words4 = sorted(proba_dict4.items(), key=lambda x: x[1], reverse=True)
# print(sort_words4[:10])

# # Loading dictionary
# #dct = pickle.load(open('C:/Users/Dingge/Doctoral_projets/Pytorch/dizionario_amazon.pkl','rb')) 
# dct = pickle.load(open('dizionario_amazon_bigram.pkl','rb')) 
# top = list() 
# for item in sort_words[:10]:
#     top.append(dct[item[0]])
#     top.append(item[1])
# print('Top words R1:', top)
# print('Note R1: ', out)

# top2 = list() 
# for item in sort_words2[:10]:
#     top2.append(dct[item[0]])
#     top2.append(item[1])
# print('Top words R2:', top2)
# print('Note R2: ', out2)

# top3 = list() 
# for item in sort_words3[:10]:
#     top3.append(dct[item[0]])
#     top3.append(item[1])
# print('Top words R3:', top3)
# print('Note R3: ', out3)

# top4 = list() 
# for item in sort_words4[:10]:
#     top4.append(dct[item[0]])
#     top4.append(item[1])
# print('Top words R4:', top4)
# print('Note R4: ', out4)
    
# las = time.time()
# print("Top words Time: ", las-beg)


