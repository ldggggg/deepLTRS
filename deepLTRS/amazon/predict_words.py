# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:44:06 2020

@author: Dingge
"""

import numpy as np
import pickle

muV = np.load('C:/Users/Dingge/Downloads/script/musical/Fix_best_muV_musical9004.npy')
logv_V = np.load('C:/Users/Dingge/Downloads/script/musical/Fix_best_logv_V_musical9004.npy')
muW = np.load('C:/Users/Dingge/Downloads/script/musical/Fix_best_muW_musical9004.npy')
logv_W = np.load('C:/Users/Dingge/Downloads/script/musical/Fix_best_logv_W_musical9004.npy')

R1 = muV[14,:].reshape(1,-1) # (1, 50)
#R2 = muV[1358,:].reshape(1,-1)
#R = np.concatenate((R1, R2), axis = 0) # (2, 50)
C1 = muW[706,:].reshape(1,-1) # (1, 50) rating=2
C2 = muW[309,:].reshape(1,-1) # (1, 50) rating=5
C = np.concatenate((C1, C2), axis = 0) # (2, 50)

beta = Mod.decode_text(torch.tensor(R1), torch.tensor(C))
beta = beta.cpu().data.numpy()

dct = pickle.load(open('C:/Users/Dingge/Doctoral_projets/Pytorch/dizionario_musical.pkl','rb'))
words = np.delete(np.array(dct), stock, axis = 0)

def GetEntropy(M,K):
  out = np.zeros(M.shape[1])
  for idx in range(M.shape[1]):
    y = M[:,idx]/np.sum(M[:,idx])
    out[idx] = 1+np.sum(y * np.log(y))/np.log(K)
  return out
 
def terms(beta, words, n = 20, s = 2, if_print = False):
  K = beta.shape[0]
  #V = beta.shape[1]
  ent = GetEntropy(beta, K)
  spec = np.zeros(shape = (beta.shape[0], beta.shape[1]))
  for k in range(K):
    spec[k,:] = beta[k,:]*(ent**s)
    Wk = np.array(words)[np.argsort(-spec[k,:])][:n]
    if (if_print == True):
        l = list()
        for i in range(n):
            l.append(dct[Wk[i]])
        print("* Topic" , k , ';' , l)
  return spec

terms(beta, words, n = 20, s = 2, if_print = True)

x = pickle.load(open('C:/Users/Dingge/Doctoral_projets/Tensorflow/SBM-meet-GNN-master/osbm_code/data/ind.cora.x','rb'), encoding='latin1')