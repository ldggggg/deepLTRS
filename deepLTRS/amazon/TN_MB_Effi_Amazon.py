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
# loading Amazon notes    
# Y = np.genfromtxt('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/amazon/AmazonOrdinal.csv', 
#                   skip_header=1, delimiter=",")
Y = np.genfromtxt('AmazonOrdinal.csv', skip_header=1, delimiter=",")
dat = Y[:,1:]
M = dat.shape[0] # 1644
P = dat.shape[1] # 1733
clr = np.random.choice(range(2), M) 
clc = np.random.choice(range(2), P)

############## loading and manipulatind docs and vocabulary  ###############   
nonzero_pos = np.where(dat != 0)
nonzero_row = nonzero_pos[0].tolist() # len=32836
nonzero_col = nonzero_pos[1].tolist()

nonzero_pos_P = np.where(dat.T != 0)
nonzero_row_P = nonzero_pos_P[0].tolist() # len=
nonzero_col_P = nonzero_pos_P[1].tolist() 

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
#                     skip_header=1, delimiter=",") 
dtm = np.genfromtxt('dtm.csv', skip_header=1, delimiter=",") 
dtm = np.asarray(dtm, dtype='float32') 

connection = np.genfromtxt('out.csv', skip_header=1, delimiter=",") 

cdtm = np.zeros(shape = (len(nonzero_row), V), dtype='float32')
for index in range(dtm.shape[0]):
    cdtm[int(dtm[:,0][index])][int(dtm[:,1][index])] = dtm[:,2][index]
        
cdtm_col = np.zeros(shape = (len(nonzero_col), V), dtype='float32')
for index in range(len(nonzero_col)):
    correspond = int(connection[index][2])
    cdtm_col[index,:] = cdtm[correspond,:]   
    
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
    
## We need to cbind idtm-dat and odtm-t(dat)
X = np.concatenate((idtm, dat_), axis = 1) # (1644, 38914)
Y = np.concatenate((odtm, dat_.transpose()), axis = 1) # (1733, 38825)
print('OK')

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
batch_size_R = 5
batch_size_C = 5

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
nb_of_topics = 50

int_dim = 50
# prior_variance = 0.5
epochs = 50
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
        correct = batch_size_R/M
        return  correct*(MCN + MCT + 2*KL) 
    else:
        correct = batch_size_C/P  
        return  correct*(MCN + MCT + 2*KL) 

    
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
#optimizer = optim.Adam(list(Mod.parameters())+list(Mod.E1.parameters())+list(Mod.E2.parameters()), lr=2e-3, betas=(0.99, 0.999))

def train(epoch, store_muV, store_logv_V, store_muW, store_logv_W):    
    Mod.train()
    ######################
    ## MB on rows first ##
    ######################
    batch_size = batch_size_R
    for batch_idx, obs in enumerate(Rdload):
        # print('batch_idx_R: ', batch_idx)
        # time5 = time.time()
        optimizer.zero_grad() 
        interval = torch.LongTensor(range(V,P+V))
        if device=="cuda":
            interval = interval.to(device)
        obs_note = torch.index_select(obs, 1, interval) # la partie de note     
        out_notes, out_text, muV, logv_V, log_vareps = Mod.forward(obs, in_row = True)
        
        if (batch_idx+1)*batch_size < M:
            b = batch_size
        else:    # si pas divisible
            b = M - batch_idx*batch_size  # la reste
        
        # time1 = time.time()
        cdtm_u = getWb(I, Wu, cdtm, b, batch_idx)
        # time2 = time.time()
        # print('getWb time: ', time2-time1)
        obs_text1 = torch.tensor(cdtm_u, dtype=torch.float32)
        if device=="cuda":
            obs_text1 = obs_text1.to(device)  
        # time3 = time.time()
        loss = lossf(obs_note[obs_note!=0], obs_text1, out_notes[obs_note!=0], out_text, 
                      muV, logv_V, log_vareps, in_row = True)
        # time4 = time.time()
        # print('lossf time: ', time4-time3)
        loss.backward()
        optimizer.step()
        # time6 = time.time()
        # print('One batch_idx time: ', time6-time5)
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
        print('Train Epoch: {} \tRLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)/(len(inA)*len(inA.transpose(0,1)))))
    
    ######################
    ##  MB on columns   ##
    ######################    
    Mod.train()
    batch_size = batch_size_C
    for batch_idx, obs in enumerate(Cdload):
        # print('batch_idx_C: ', batch_idx)
        # time5 = time.time()
        optimizer.zero_grad() 
        interval = torch.LongTensor(range(V,M+V))
        if device=="cuda":
            interval = interval.to(device)
        obs_note = torch.index_select(obs, 1, interval) # la partie de note
        out_notes, out_text, muW, logv_W, log_vareps = Mod.forward(obs, in_row = False)
        
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
        # time6 = time.time()
        # print('One batch_idx time: ', time6-time5)
        
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

    if epoch % 1 == 0:
        print('Train Epoch: {} \tCLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)/(len(inB)*len(inB.transpose(0,1)))))

    return store_muV, store_logv_V, store_muW, store_logv_W
    
store_vloss = np.zeros(epochs)

start = time.time()
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
    if epoch % 1 == 0:
        print('Validation RMSE {:.6f}'.format(perdita))   
    
end = time.time()
print('Train time', end-start)    

## test
out_f = torch.mm(store_muV, store_muW.transpose(0,1))
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
est_out += gap
# print(est_out)
print(" Gap accuracy: ", accuracy_score(est_out, real_out))

rmse = np.sqrt(np.sum((est_out - real_out)**2  )/len(est_out))
print( 'Final True RMSE: {}'.format(rmse))

last = time.time()
print("All the Time: ", last-begin)
