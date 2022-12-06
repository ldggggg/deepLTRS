# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:05:18 2020

@author: Dingge
"""

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
import time
import pickle
import torch.cuda
import os
import scipy.io as scio

# device = "cpu"

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# device = torch.device('cuda:0')

###############################################################################
# loading notes
dat = np.load("C:/Users/Dingge/Doctoral_projets/Pytorch/sim_data_notes_amazon.npy")
M,P = dat.shape

Otest = np.zeros((M,P))
Otraining = dat.copy()

############## loading and manipulatind docs and vocabulary  ###############
dat_ = dat.copy()

# saving positions of observed values
nonzero_pos = np.where(dat != 0)
nonzero_row = nonzero_pos[0].tolist()
nonzero_col = nonzero_pos[1].tolist()

nonzero_pos_P = np.where(dat.T != 0)
nonzero_row_P = nonzero_pos_P[0].tolist() # len=
nonzero_col_P = nonzero_pos_P[1].tolist() 

dat_ = dat.copy()
## inserting missing values
import random

ix = [(nonzero_row[idx], nonzero_col[idx]) for idx in range(len(nonzero_row))] # All nozero positions
for row, col in random.sample(ix, int(round(0.2*len(ix)))): # 20 % for val and test   
    Otest[row, col] = dat_[row, col]
    Otraining[row, col] = 0

# saving positions of observed values
store1 = np.where(dat_ != 0)
store2 = np.where(Otest!= 0)
store3 = np.where(Otraining!=0)
print(len(store1[0]), len(store2[0]), len(store3[0])) # number for train, val and test

dic = {}
dic = {'M':dat.T, 'Otest':Otest.T, 'Otraining':Otraining.T}
#scio.savemat('training_test_dataset.mat',dic)

#data = scio.loadmat('training_test_dataset.mat')

import h5py
hf = h5py.File('training_test_dataset.mat', 'w')
for k, v in dic.items():
    hf.create_dataset(k, data=np.array(v))
hf.close()    

h5f = h5py.File('training_test_dataset.mat','r')  
#h5f.close()  
