 # -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:13:13 2020

@author: Dingge
"""
import numpy as np
import mat73
import scipy.io as scio

data_dict = mat73.loadmat('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/gc-mc-master/gcmc/data/douban/training_test_dataset.mat')
#data_dict = scio.loadmat('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/gc-mc-master/gcmc/data/douban/training_test_dataset.mat')
a = np.where(data_dict['M']!=0)
print(len(a[0]))
b = np.where(data_dict['Otest']!=0)
print(len(b[0]))
c = np.where(data_dict['Otraining']!=0)
print(len(c[0]))

# from PIL import Image

# image1 = Image.open(r'C:/Users/Dingge/DeepL/NeurIPS20/deepLTRS.png')
# im1 = image1.convert('RGB')
# im1.save(r'C:/Users/Dingge/DeepL/ICML2020/deepLTRS.pdf')

import matplotlib.pyplot as plt        
# plt.plot(store_vloss, color = 'red') 

import numpy as np
## load data
muV = np.load('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/best_muV_simu_75K.npy')
muW = np.load('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/best_muW_simu_75K.npy') 

from sklearn.manifold import TSNE
V_embedded = TSNE(n_components=2,init='pca', perplexity=30,early_exaggeration=100).fit_transform(muV)
W_embedded = TSNE(n_components=2,init='pca', perplexity=30,early_exaggeration=100).fit_transform(muW)
# V_embedded = TSNE(n_components=2,init='pca').fit_transform(muV)
# W_embedded = TSNE(n_components=2,init='pca').fit_transform(muW)
        
f, ax = plt.subplots(1,figsize=(15,10))
ax.scatter(V_embedded[:,0], V_embedded[:,1], color = 'yellow', label='User')
ax.scatter(W_embedded[:,0], W_embedded[:,1], color = 'blue', label='Product')
#ax.set_xlabel('Visualization of users and products for simulated data',fontsize=20)
ax.set_xlabel('K = 75',fontsize=20)
plt.legend(loc='upper left',fontsize=20)
plt.show()
#f.savefig("C:/Users/Dingge/DeepL/ICML2020/visu_simu.pdf", bbox_inches='tight')

from sklearn.decomposition import PCA       
pca = PCA(n_components = 2) 
o1 = pca.fit(muV).fit_transform(muV)
o2 = pca.fit(muW).fit_transform(muW)

f, ax = plt.subplots(1,figsize=(15,10))
ax.scatter(o1[:,0], o1[:,1], color = 'yellow', label='User')
ax.scatter(o2[:,0], o2[:,1], color = 'blue', label='Product')
ax.set_xlabel('Visualization of users and products for simulated data',fontsize=20)
plt.legend(loc='upper left',fontsize=20)
plt.show()