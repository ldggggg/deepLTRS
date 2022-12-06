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

############# Auxiliary Functions #############################################

def ipos(M, P, idx):
    return np.arange(idx*P, idx*P + P)

def jpos(M, P, idy):
    return np.arange(idy, idy + (M*P), P)

###############################################################################

#dat = np.load("sim_data_notes.npy")
#dat = dat[:25,:]

# loading notes
dat = np.load("sim_data_notes.npy")
M,P = dat.shape

# loading labels
clr = np.load('clr.npy')
clc = np.load('clc.npy')

############## loading and manipulatind docs and vocabulary  ###############

dct = pickle.load(open('dizionario.pkl','rb'))
dctn = dct.token2id
V = len(dctn)

with open ('sim_data_docs', 'rb') as fp:
    docs = pickle.load(fp)

# num version of docs
ndocs = []
for doc in range(len(docs)):
    tmp = []
    for word in docs[doc]:
        tmp.append(dctn[word])
    ndocs.append(tmp)

############################################################################    
## Now I need to create two dtms (individual specific and object specific..)

# complete dtm
cdtm = []
for idx in range(len(ndocs)):
    #print(idx)
    cdtm.append( np.bincount(ndocs[idx], minlength = V ) )   
cdtm = np.asarray(cdtm, dtype = 'double')

# ** individuals
idtm = np.zeros(shape = (M, V))
for idx in range(M):
    #print(idx)
    pos = ipos(M,P,idx)
    val = cdtm[pos, :].sum(0)
    idtm[idx,:] = val

# ** object
odtm = np.zeros(shape = (P, V))
for idy in range(P):
    # print(idy)
    pos = jpos(M,P,idy)
    val = cdtm[pos, :].sum(0)
    odtm[idy,:] = val

## Scaling the dtm(s)
#for idx in range(len(cdtm)):
#    cdtm[idx,:] /= np.sum(cdtm[idx,:])
#
#for idx in range(len(idtm)):
#    idtm[idx,:] /= np.sum(idtm[idx,:])
#
#for idx in range(len(odtm)):    
#    odtm[idx,:] /= np.sum(odtm[idx,:])

## inserting missing values in the ordinal data matrix
dat_ = dat.copy()
val = 0.0
import random
random.seed(0)
np.random.seed(0)
# we are storing the positions of validation and test entries!
ipv = jpv = ipt = jpt = np.zeros(shape = (1,1))
ix = [(row, col) for row in range(dat_.shape[0]) for col in range(dat_.shape[1])]
for row, col in random.sample(ix, int(round(.2*len(ix)))):
    if np.random.uniform() > 0.5:    # validation
        ipv = np.vstack((ipv, row))
        jpv = np.vstack((jpv, col))
        dat_[row, col] = 0
    else:                            # test
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
    
## We need to cbind idtm-dat and odtm-t(dat)
X = np.concatenate((idtm, dat_), axis = 1)
Y = np.concatenate((odtm, dat_.transpose()), axis = 1)
#X = dat.copy()


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

int_dim = 50
epochs = 1800
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
          self.D3 = nn.Linear(int_dim, V, bias = True)              # beta             
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
        return out_p
        
    def forward(self, x, y):
        muV, logv_V = self.E1(x)
        muW, logv_W = self.E2(y)
        zV = self.reparametrize(muV, logv_V)
        zW = self.reparametrize(muW, logv_W)
        out_notes = self.decode_notes(zV, zW)
        out_text = self.decode_text(zV, zW)
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

for epoch in range(epochs):
    train(epoch)    
    

# test (not exactly...)

Mod.eval()

outN, outT, muV, logv_V, muW, logv_W, log_vareps = Mod.forward(inA, inB)

outN = outN.reshape(M, P)
noutN = outN.cpu().data.numpy()

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

eclr = KMeans(n_clusters = 2).fit(noutN)
eclc = KMeans(n_clusters = 2).fit(noutN.T)

print(" ARI (rows):{} ".format(ari(clr, eclr.labels_)))
print(" ARI (cols):{} ".format(ari(eclc.labels_, clc)))

###
print(" The estimated eta square is: {}".format(torch.exp(log_vareps)))


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
#emnout = np.round(mnout)
#eonout = np.round(onout)
#from sklearn.metrics import accuracy_score
#accuracy_score(emnout, minA)
#accuracy_score(eonout, oinA)
#
### RMSE on the test dataset
rmse = np.sqrt(np.sum( (mnout - minN)**2  )/len(mnout))
print( 'Our model (test data) RMSE: {}'.format(rmse))

######## Hierarchical Poisson Recommendation (for comparison) ##########
#import pandas as pd, numpy as np
#from hpfrec import HPF
#
#(i,j) = X.nonzero()
#counts_df = pd.DataFrame({
#        'UserId' : i,
#        'ItemId' : j,
#        'Count'  : X[(i,j)].astype('int32')
#        }
#        )
#
#val_df = pd.DataFrame({
#        'UserId' : ipv,
#        'ItemId' : jpv,
#        'Count'  : dat[val_pos]
#        })
#
### Initializing the model object
#recommender = HPF()
#
#
### For stochastic variational inference, need to select batch size (number of users)
#recommender = HPF(users_per_batch = 20)
#
### Full function call
#recommender = HPF(
#	k=30, a=0.3, a_prime=0.3, b_prime=1.0,
#	c=0.3, c_prime=0.3, d_prime=1.0, ncores=-1,
#	stop_crit='train-llk', check_every=10, stop_thr=1e-3,
#	users_per_batch=None, items_per_batch=None, step_size=lambda x: 1/np.sqrt(x+2),
#	maxiter=100, reindex=True, verbose=True,
#	random_seed = None, allow_inconsistent_math=False, full_llk=False,
#	alloc_full_phi=False, keep_data=True, save_folder=None,
#	produce_dicts=True, keep_all_objs=True, sum_exp_trick=False
#)
#
### Fitting the model while monitoring a validation set
#recommender = HPF(stop_crit='val-llk')
#recommender.fit(counts_df, val_set = val_df )
#
### Fitting the model to the data
##recommender.fit(counts_df)
#
### Making predictions on the train dataset
#obsout = recommender.predict(user = i, item = j)
#obsout = np.round(obsout)
#accuracy_score(obsout, X[(i,j)])
#
### Making predictions on the test dataset
#mout = recommender.predict(user = test_pos[0], item = test_pos[1])
#accuracy_score(np.round(mout), dat[test_pos])
#
### computing the RMSE on the test dataset
#rmse_hpf = np.sqrt(np.sum( (mout - dat[test_pos])**2  )/len(mout))
#print( 'Our model RMSE: {}'.format(rmse))
#print( 'HPF RMSE: {}'.format(rmse_hpf))
#
################
#
#def print_top_words_z(beta, vocab, n_top_words = 10):
#    # averaging  accross all topix
#     nbeta = beta.mean(axis = 0)
#     nbeta = beta - nbeta
#     for i in range(len(nbeta)):
#                 print('\n--------------Topic{}:-----------------'.format(i+1))
#                 line = " ".join([vocab[j] 
#                            for j in beta[i].argsort()[:-n_top_words - 1:-1]])
#                 print('     {}'.format(line))
#                 print('--------------End of Topic{}---------------'.format(i+1))
#
#
#beta = Mod.D3.weight.data.numpy().T
#print_top_words_z(beta, dct)
