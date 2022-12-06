#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:09:46 2019
@author: marco
"""
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader  
from torch.autograd import Variable
import math
import time
import pickle
import torch.cuda
import os

begin = time.time() 
# device = "cpu"

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0')

############# Auxiliary Functions #############################################

def ipos(M, P, idx):
    return np.arange(idx*P, idx*P + P)

def jpos(M, P, idy):
    return np.arange(idy, idy + (M*P), P)

###############################################################################
# loading notes
dat = np.load("sim_data_notes_100_600.npy")
M,P = dat.shape

# loading labels
clr = np.load('clr_100_600.npy')
clc = np.load('clc_100_600.npy')

############## loading and manipulatind docs and vocabulary  ###############
dat_ = dat.copy()

############## loading and manipulatind docs and vocabulary  ###############
dct = pickle.load(open('dizionario.pkl','rb'))
dctn = dct.token2id
V = len(dctn)

with open ('sim_data_docs_100_600', 'rb') as fp:
    docs = pickle.load(fp)

# num version of docs
ndocs = []
for doc in range(len(docs)):
    tmp = []
    for word in docs[doc]:
        tmp.append(dctn[word])
    ndocs.append(tmp)
    
################ row major #################
# complete dtm for row
cdtm = []
for idx in range(len(ndocs)):
    cdtm.append( np.bincount(ndocs[idx], minlength = V ) )   
cdtm = np.asarray(cdtm, dtype = 'float32')

############### col major ##################
# complete dtm for col
cdtm_col = np.zeros((cdtm.shape))
for row in range(P):
    cdtm_col[row*M:(row+1)*M, :] = cdtm[[idx for idx in range(row, M*P, P)], :]

# inserting missing values
val = 0.0
import random
# we are storing the positions of validation and test entries!
ipv = jpv = ipt = jpt = np.zeros(shape = (1,1))
# random.seed(0)
# np.random.seed(0)
delete = list()
delete_col = list()
ix = [(row, col) for row in range(dat_.shape[0]) for col in range(dat_.shape[1])]
for row, col in random.sample(ix, int(round(0.99*len(ix)))):
    dat_[row, col] = 0
    delete.append(row*P+col)
    delete_col.append(col*M+row)   
    if np.random.uniform() < 1/1000:    # validation
        ipv = np.vstack((ipv, row))
        jpv = np.vstack((jpv, col))
        dat_[row, col] = 0
        # delete.append(row*P+col)
        # delete_col.append(col*M+row)
    elif np.random.uniform() > 999/1000:    # test
        ipt = np.vstack((ipt, row))
        jpt = np.vstack((jpt, col))
        dat_[row, col] = 0 
        # delete.append(row*P+col)
        # delete_col.append(col*M+row)
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

# delete no exist rows

print('del:', len(delete))
cdtm = np.delete(cdtm, delete, axis = 0)
print(cdtm.shape)
print('del:', len(delete_col))
cdtm_col = np.delete(cdtm_col, delete_col, axis = 0)
print(cdtm_col.shape)

pos1 = np.vstack((store1[0], store1[1]))
pos1 = pos1.transpose()  # no zero position for user
pos2 = np.vstack((store2[0], store2[1]))
pos2 = pos2.transpose()  # no zero position for product

############################# new concatnation ###############################
# nonzero = list()
# for i in range(M):
#     for j in range(P):
#         if dat_[i,j] != 0:
#             nonzero.append(dat_[i,j])
# print(len(nonzero))
# rating = np.asarray(nonzero).reshape(-1,1)

# nonzero_col = list()
# for i in range(P):
#     for j in range(M):
#         if dat_.T[i,j] != 0:
#             nonzero_col.append(dat_.T[i,j])
# print(len(nonzero_col))
# rating_col = np.asarray(nonzero_col).reshape(-1,1)

############# Wu ##############
Wu = np.zeros((len(store1[0]), ), dtype = 'int32')  # (3492, )
for row in range(len(store1[0])):
    Wu[row,] = pos1[row][0] * P + pos1[row][1]   
    
Wu_col = np.zeros((len(store1[0]), ), dtype = 'int32')  # (3492, )
for row in range(len(store1[0])):
    Wu_col[row,] = pos2[row][0] * M + pos2[row][1]  

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
    odtm[idx, :] = cdtm_col[pos, :].sum(0)                               # !!!!!!!!!!!!!!!!!!!!! Bug fixed !!!!!!!! 
    j = j + len((np.where(pos2[:,0] == idx))[0])

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
mean_all = np.full((M, P), dat.sum()/float(M*P*3))
if device=="cuda":
    mean_all = mean_all.to(device)       
    
####### Identity vector ########
bu_i = torch.Tensor(M, 1).fill_(1.0)
bp_j = torch.Tensor(P, 1).fill_(1.0)     
    
###### mean fixed #######
b_u = torch.Tensor(mean_u)
b_p = torch.Tensor(mean_p)
b_all = torch.Tensor(mean_all) 

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

## We need to cbind idtm-dat and odtm-t(dat)
X = np.concatenate((idtm, dat_), axis = 1)
Y = np.concatenate((odtm, dat_.T), axis = 1)
print(X.shape, Y.shape)

## target data, row and column data
Rdata = MDataset(X, transform = True)      # The individual/input matrix
Cdata = MDataset(Y, transform = True)      # The object/input matrix


## Loading all datasets
batch_size_R = 5
batch_size_C = 4


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
epochs = 20
mc_iter = 1

# variable status
# status = 'row'
# Encoding class 
class Encoder(nn.Module):
    def __init__(self, type = 'R', idx = 0):
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
        h1 = F.softplus(self.en1(x))
        h2 = self.en2_drop(F.softplus(self.en2(h1)))
        mu = self.mu_bn(self.mu(h2))
        logv =self.logv_bn(self.logv(h2))
        return mu, logv
    
    def pos_emb(self, type = 'R'):
        if type == 'R':
            dim = M
        elif type == 'C':
            dim = P
        pe = torch.zeros(V, dim)
        position = torch.arange(0, V).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        
        return pe
        
    def forward(self, x, idx, type = 'R'):
        if type == 'R':
            pe = self.pos_emb('R').T
            if idx != None:
                if (idx+1)*batch_size_R < M:
                    b = batch_size_R 
                    pe = pe[idx*b:(idx+1)*b, :]
                else:    # si pas divisible
                    b = M - idx*batch_size_R   # le reste
                    pe = pe[idx*batch_size_R:M, :]
            x[:, :V] = x[:, :V] + Variable(pe, requires_grad=False)
        
        elif type == 'C':
            pe = self.pos_emb('C').T
            if idx != None:
                if (idx+1)*batch_size_C < P:
                    b = batch_size_C 
                    pe = pe[idx*b:(idx+1)*b, :]
                else:    # si pas divisible
                    b = P - idx*batch_size_C   # le reste
                    pe = pe[idx*batch_size_C:P, :]
            x[:, :V] = x[:, :V] + Variable(pe, requires_grad=False)
          
        return self.encode(x)
        
# Decoding NN, containing two Encoders
class Decoder(nn.Module):
    def __init__(self):
          super(Decoder, self).__init__()           
          self.E1 = Encoder('R', idx = 0)
          self.E2 = Encoder('C', idx = 0)
          self.p_drop = nn.Dropout(0.2)
          # decoding - text only
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

    def decode_notes(self, zV, zW, bu, bp, bu_, bp_):
        val = torch.mm(zV, zW.transpose(0,1))
        #print(zV.size(), zW.size(), val.size())
        #val = torch.mm(F.relu(zV), F.relu(zW.transpose(0,1)))
        #out = F.relu(val) # nodec 
        bias = torch.mm(bu, bp.transpose(0,1)) + torch.mm(bu_, bp_.transpose(0,1))
        #print(bu.size(), bp.size(), bias.size())
        out = val + bias
        return out
    
    def decode_text(self, zV, zW):
        s1 = zV.shape[0]
        s2 = zW.shape[0]
        theta = torch.zeros(size = (s1*s2, nb_of_topics))
        if device == 'cuda':
            theta = theta.to(device)
        for idx in range(s1):
            start = idx*s2
            in_dat = torch.cat((zV[idx,:].expand_as(zW), zW), 1)       
            theta[start:(start + s2),:] = self.newD2((F.relu(self.newD1(in_dat))))
        p = F.softmax(theta, -1)                                                
        p = self.p_drop(p)
        out_p = F.softmax(self.D3_bn(self.D3(p)), -1)
        return out_p
    
    # def decode_specific(self, R, C):
    #     val = torch.mm(R, C.transpose(0,1))
    #     out = val
        
    #     in_dat = torch.cat((R, C), 1)
    #     theta = self.newD2((F.relu(self.newD1(in_dat))))
    #     p = F.softmax(theta, -1)                                                
    #     p = self.p_drop(p)
    #     out_p = F.softmax(self.D3_bn(self.D3(p)), -1)
    #     return out, out_p
    
    # def forward(self, x, idx, in_row = True): # optimize two means
    #     if in_row == True:
    #         muV, logv_V = self.E1(x)
    #         V = self.reparametrize(muV, logv_V)
    #         W = self.reparametrize(store_muW, store_logv_W)
    #         if (idx+1)*batch_size_R < M:
    #             b = batch_size_R 
    #             b_u = self.b_u[idx*b:(idx+1)*b,:]
    #             b_u_ = store_b_u[idx*b:(idx+1)*b,:]
    #         else:    # si pas divisible
    #             b = M - idx*batch_size_R   # le reste
    #             b_u = self.b_u[idx*batch_size_R:M,:]
    #             b_u_ = store_b_u[idx*batch_size_R:M,:]
    #         #print(idx)
    #         #print(bu.size())
    #         #b_p = store_b_p 
    #         out_notes = self.decode_notes(V, W, b_u, store_b_p, b_u_, self.b_p)
    #         out_text = self.decode_text(V, W)
    #         return out_notes, out_text, muV, logv_V, self.log_vareps, self.b_u 

    #     else:
    #         muW, logv_W = self.E2(x)
    #         V = self.reparametrize(store_muV, store_logv_V)
    #         W = self.reparametrize(muW, logv_W)
    #         if (idx+1)*batch_size_C < P:
    #             b = batch_size_C 
    #             b_p = self.b_p[idx*b:(idx+1)*b,:]
    #             b_p_ = store_b_p[idx*b:(idx+1)*b,:]
    #         else:    # si pas divisible
    #             b = P - idx*batch_size_C   # le reste
    #             b_p = self.b_p[idx*batch_size_C:P,:]
    #             b_p_ = store_b_p[idx*batch_size_C:P,:]
    #         #b_u = store_b_u 
    #         out_notes = self.decode_notes(W, V, b_p, store_b_u, b_p_, self.b_u)
    #         #out_notes = self.decode_notes(W, V)
    #         out_text = self.decode_text(W, V)
    #         return out_notes, out_text, muW, logv_W, self.log_vareps, self.b_p
        
    def forward(self, x, idx, in_row = True):   # not optimize two means
        if in_row == True:
            muV, logv_V = self.E1(x, idx, 'R')
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
            #print(idx)
            #print(bu.size())
            #b_p = store_b_p 
            out_notes = self.decode_notes(V, W, b_u1, bp_j, b_u_, b_p)
            out_text = self.decode_text(V, W)
            return out_notes, out_text, muV, logv_V, self.log_vareps

        else:
            muW, logv_W = self.E2(x, idx, 'C')
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
            #b_u = store_b_u 
            out_notes = self.decode_notes(W, V, b_p1, bu_i, b_p_, b_u)
            #out_notes = self.decode_notes(W, V)
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
    MCN = torch.sum(MCN + 0.5*log_vareps.expand_as(targetN))
    # ** main loss component (text)
    MCT  = - torch.sum(targetT * torch.log(outT+1e-40))

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
# store_b_u = torch.Tensor(mean_u)  # mean of users
# store_b_p = torch.Tensor(mean_p)  # mean of products

# the model
Mod = Decoder()
if device=="cuda":
    Mod = Mod.cuda()
#optimizer = optim.Adam(Mod.parameters(), lr=2e-4, betas=(0.99, 0.999), weight_decay=0.05)
optimizer = optim.Adam(Mod.parameters(), lr=2e-3, betas=(0.99, 0.999))
# optimizer = optim.SGD(Mod.parameters(), lr=2e-3)
#optimizer = optim.Adagrad(Mod.parameters(), lr=2e-3, lr_decay=0.995)

############# Wb #############
def getWb(I, Wu, cdtm, batch_size, batch_idx):
    Wb = np.zeros((batch_size * P, V))
    i = batch_idx * batch_size
    idx = I[i, 1]
    while Wu[idx] < (i + batch_size) * P : #and idx < len(Wu):
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
    while Wu_col[idx] < (j + batch_size) * M : #and idx < len(Wu_col):
        pos = Wu_col[idx] - j * M
        Wb[pos, :] = cdtm_col[idx, :]
        idx = idx + 1    
        if idx == len(Wu_col):
            break
        #print(idx, j, pos)
    return Wb

store_Rloss = torch.zeros(epochs)
store_Closs = torch.zeros(epochs)

def train(epoch, store_muV, store_logv_V, store_muW, store_logv_W):    
    Mod.train()
    
    ################
    ## MB on rows ##
    ################
    batch_size = batch_size_R
    for batch_idx, obs in enumerate(Rdload):
        optimizer.zero_grad() 
        interval = torch.LongTensor(range(V,P+V))
        if device=="cuda":
            interval = interval.to(device)
        obs_note = torch.index_select(obs, 1, interval) # la partie de note       
        out_notes, out_text, muV, logv_V, log_vareps = Mod.forward(obs, batch_idx, in_row = True)
        
        if (batch_idx+1)*batch_size < M:
            b = batch_size
        else:    # si pas divisible
            b = M-batch_idx*batch_size  # le reste
          
        cdtm_u = getWb(I, Wu, cdtm, b, batch_idx)
        obs_text1 = torch.tensor(cdtm_u, dtype = torch.float32)
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
    store_muV_, store_logv_V_ = Mod.E1(inX, idx = None)
    store_muV = store_muV_.detach()
    store_logv_V = store_logv_V_.detach()
    
    if epoch % 1 == 0:
        store_Rloss[epoch] = torch.Tensor.item(loss)
        #/(len(inA)*len(inA.transpose(0,1)))
        print('Train Epoch: {} \tRLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)))
    
    ######################
    ##  MB on columns   ##
    ######################    
    Mod.train()
    batch_size = batch_size_C
    for batch_idx, obs in enumerate(Cdload):
        optimizer.zero_grad() 
        interval = torch.LongTensor(range(V,M+V))
        if device=="cuda":
            interval = interval.to(device)
        obs_note = torch.index_select(obs, 1, interval) # la partie de note
        out_notes, out_text, muW, logv_W, log_vareps = Mod.forward(obs, batch_idx, in_row = False)

        if (batch_idx+1)*batch_size < P:
            b = batch_size
        else:
            b = P-batch_idx*batch_size
 
        cdtm_p = getWb_col(I_col, Wu_col, cdtm_col, b, batch_idx)
        obs_text1 = torch.tensor(cdtm_p, dtype = torch.float32)         
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
    store_muW_, store_logv_W_ = Mod.E2(inY, type = 'C', idx = None)
    store_muW = store_muW_.detach()
    store_logv_W = store_logv_W_.detach()

    if epoch % 1 == 0:
        store_Closs[epoch] = torch.Tensor.item(loss)
        #/(len(inB)*len(inB.transpose(0,1)))
        print('Train Epoch: {} \tCLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss)))

    return store_muV, store_logv_V, store_muW, store_logv_W
store_vloss = np.zeros(epochs)
all_muV = torch.zeros(size = [epochs, M, int_dim])
all_logv_V = torch.zeros(size = [epochs, M, int_dim])
all_muW = torch.zeros(size = [epochs, P, int_dim])
all_logv_W = torch.zeros(size = [epochs, P, int_dim])
# all_b_u = torch.zeros(size = [epochs, M, 1])
# all_b_p = torch.zeros(size = [epochs, P, 1])

####### Identity vector ########
bu_i = torch.Tensor(M, 1).fill_(1.0)
bp_j = torch.Tensor(P, 1).fill_(1.0)   

start = time.time()
for epoch in range(epochs):
    store_muV, store_logv_V, store_muW, store_logv_W = train(epoch, store_muV, store_logv_V, store_muW, store_logv_W)    
    ###############
    # valid check #
    ###############
    out = torch.mm(store_muV, store_muW.transpose(0,1)) + torch.mm(b_u, bp_j.transpose(0,1)) + torch.mm(bu_i, b_p.transpose(0,1))
    indat = torch.Tensor(dat)
    if device=="cuda":
          indat = indat.to(device)
    perdita = vlossf(indat[val_pos], out[val_pos])
    store_vloss[epoch] = perdita
    all_muV[epoch] = store_muV
    all_logv_V[epoch] = store_logv_V
    all_muW[epoch] = store_muW
    all_logv_W[epoch] = store_logv_W
    # all_b_u[epoch] = store_b_u
    # all_b_p[epoch] = store_b_p
    if epoch % 1 == 0:
        print('Validation RMSE {:.6f}'.format(perdita)) 
        
end = time.time()
print("Running Time: ", end-start)  

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
# best_b_u = all_b_u[minidx]
# best_b_p = all_b_p[minidx]

## test
out_f = torch.mm(best_muV, best_muW.transpose(0,1)) + torch.mm(b_u, bp_j.transpose(0,1)) + torch.mm(bu_i, b_p.transpose(0,1))
out_f = out_f.cpu().data.numpy()
store_Rloss = store_Rloss.cpu().data.numpy()
store_Closs = store_Closs.cpu().data.numpy()

# test fitted values
est_out = out_f[test_pos[0], test_pos[1]]
#print(est_out)
real_out = dat[test_pos[0], test_pos[1]]

# from sklearn.metrics import accuracy_score
# est_out = np.round(est_out)
# print(" Our Accuracy: ", accuracy_score(est_out, real_out))

rmse = np.sqrt(np.sum((est_out - real_out)**2  )/len(est_out))
print( 'Our model RMSE: {}'.format(rmse))
#torch.save(Mod, 'essai.pt')
#torch.save(Mod.state_dict(), 'essai2.pt')
# np.save('simu_deepLTRS', est_out)

#Later to restore:
# Mod.load_state_dict(torch.load('C:/Users/Dingge/Doctoral_projets/Pytorch/Toserver/Toserver/essai2.pt'))
# Mod.eval()

#torch.load('C:/Users/Dingge/Doctoral_projets/Pytorch/Toserver/Toserver/essai.pt')

# muV = best_muV
# muW = best_muW

# np.save('best_muV_simu_100K', best_muV)
# np.save('best_muW_simu_100K', best_muW)

# from  sklearn.metrics import adjusted_rand_score as ari
# from sklearn.cluster import KMeans

# eclr = KMeans(n_clusters = 2).fit(out_f)
# eclc = KMeans(n_clusters = 2).fit(out_f.T)

# # print(ari(clr, eclr.labels_))
# # print(ari(eclc.labels_, clc))
 
# colR = []
# for idx in range(len(clr)):
#     if clr[idx] == 0:
#         colR.append('red')
#     else:
#         colR.append('blue')

# colC = []
# for idx in range(len(clc)):
#     if clc[idx] == 0:
#         colC.append('yellow')
#     else:
#         colC.append('green')

# import matplotlib.pyplot as plt        
# # plt.plot(store_vloss, color = 'red') 

# # from sklearn.decomposition import PCA       
# # pca = PCA(n_components = 2) 
# # o1 = pca.fit(muV).fit_transform(muV)
# # o2 = pca.fit(muW).fit_transform(muW)

# import numpy as np
# ## load data
# muV = np.load('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/best_muV_simu.npy')
# muW = np.load('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/best_muW_simu.npy') 

# from sklearn.manifold import TSNE
# V_embedded = TSNE(n_components=2,init='pca', perplexity=30,early_exaggeration=100).fit_transform(muV)
# W_embedded = TSNE(n_components=2,init='pca', perplexity=30,early_exaggeration=100 ).fit_transform(muW)
        
# f, ax = plt.subplots(1,figsize=(15,10))
# ax.scatter(V_embedded[:,0], V_embedded[:,1], color = 'yellow', label='User')
# ax.scatter(W_embedded[:,0], W_embedded[:,1], color = 'blue', label='Product')
# ax.set_xlabel('Visualization of users and products for simulated data',fontsize=20)
# plt.legend(loc='upper left',fontsize=20)
# plt.show()
# fig.savefig("C:/Users/Dingge/DeepL/ICML2020/visu_simu.pdf", bbox_inches='tight')

# print(' ARI in row: {}'.format(ari(clr, eclr.labels_)))
# print(' ARI in col: {}'.format(ari(eclc.labels_, clc)))

from sklearn.metrics import accuracy_score
est_out = np.round(est_out)
# print(est_out)
print(" missing accuracy score:{} ".format(accuracy_score(est_out, real_out)))

gap = 1 - np.min(est_out)   ## Normalisation: c'est nécessaire 
print(" Gap: ", gap) 
# est_out += gap
# print(est_out)
# print(" Gap Accuracy: ", accuracy_score(est_out, real_out))

rmse = np.sqrt(np.sum((est_out - real_out)**2  )/len(est_out))
print( 'Final True RMSE: {}'.format(rmse))

last = time.time()
print("All the Time: ", last-begin)

# print(est_out)
# print(best_b_u)

#############################################################################
###################################
## Specific muV, muW, log_V, log_W ##
###################################
# muV1 = best_muV[16]
# logv_V1 = best_logv_V[16] 
# muV2 = best_muV[13] 
# logv_V2 = best_logv_V[13] 
# muW1 = best_muW[15] 
# logv_W1 = best_logv_W[15]

# R1 = torch.zeros(size=(1, int_dim))
# R2 = torch.zeros(size=(1, int_dim))
# C1 = torch.zeros(size=(1, int_dim))

# # R1[:,:] = torch.normal.multivariate(muV1, logv_V1)
# # R2[:,:] = torch.normal.multivariate(mean = muV2, std = logv_V2, size = int_dim)
# # C1[:,:] = torch.normal.multivariate(mean = muW1, std = logv_W1, size = int_dim)

# for i in range(int_dim):
#      R1[:,i] = torch.normal(mean = muV1[i], std = logv_V1[i])
#      R2[:,i] = torch.normal(mean = muV2[i], std = logv_V2[i])
#      C1[:,i] = torch.normal(mean = muW1[i], std = logv_W1[i])
   
# note, proba = Mod.decode_specific(R1, C1)
# print('Note: ', note)
# list_proba = proba.cpu().data.numpy().T.tolist()
# proba_dict = {}
# for index, item in enumerate(list_proba):
#     proba_dict[index] = item
    
# a = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
# print(a[:10])
# Loading dictionary
# dct = pickle.load(open('C:/Users/Dingge/Doctoral_projets/Pytorch/dizionario_amazon.pkl','rb')) 
# top = list() 
# for item in a[:10]:
#     top.append(dct[item[0]])
#     top.append(item[1])
# print(top)