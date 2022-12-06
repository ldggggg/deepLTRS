#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:59:47 2019

@author: marco
"""

#import nltk
from nltk.corpus import stopwords 
from gensim.corpora import Dictionary
#from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
import numpy as np
from random import sample, shuffle
import pickle

# loading texts and removing punctuation
tokenizer = RegexpTokenizer(r'\w+')
A = tokenizer.tokenize(open('msgA.txt').read())
B = tokenizer.tokenize(open('msgB.txt').read())
C = tokenizer.tokenize(open('msgC.txt').read())
D = tokenizer.tokenize(open('msgD.txt').read())

# Turning everything to lowercase
A = [idx.lower() for idx in A]
B = [idx.lower() for idx in B]
C = [idx.lower() for idx in C]
D = [idx.lower() for idx in D]

# removing stop words
stop_words = set(stopwords.words('english')) 
A = [w for w in A if not w in stop_words]
B = [w for w in B if not w in stop_words]
C = [w for w in C if not w in stop_words]
D = [w for w in D if not w in stop_words]


# creating a dictionary from the above texts
corpus = [A,B,C,D]
# dct = Dictionary(corpus)
# dct.save("dizionario.pkl")
# dct[72]
# dct.token2id["baby"]
dct = pickle.load(open('dizionario.pkl','rb'))
dctn = dct.token2id

M = 75
P = 60
Q = 2
L = 2

clr = np.random.choice(range(Q), M) 
clc = np.random.choice(range(L), P)

# Nd = 10000
Theta = 0.06*np.ones(shape = (4,4))
np.fill_diagonal(Theta, .82)
Theta = Theta.reshape((Q,Q,4))

#
gammas = [1.5, 2.5, 3.5, np.inf]

mu = np.matrix([[1.0, 4.0],[4.0, 1.0]])
sd = np.repeat(1.0, 4)
sd = sd.reshape((2,2))

def gnote(gammas, val):
    out = 0
    while val > gammas[out]:
        out += 1
    return out+1    
        
# Row major ordering
docs = []
T = []
notes = np.zeros(shape = (M,P))
latent_notes = np.zeros(shape = (M,P))
for i in range(M):
    for j in range(P):
        cli = clr[i]
        clj = clc[j]
        # doc sampling
        T.append(np.argmax(Theta[cli, clj, :]))
        Nw = np.round(np.random.normal(100,5))
        # Number of words picked in each text
        store = np.random.multinomial(Nw, Theta[cli, clj, :])
        pos = 0
        msg = []
        for idy in store:
            # print(idy)
            sampled_pos = sample(range(len(corpus[pos])), idy)
            for idz in sampled_pos:
                # print(idz)
                msg.append(corpus[pos][idz])
            pos += 1
        docs.append(msg)
        # notes sampling
        val = np.random.normal(mu[cli, clj], sd[cli, clj])
        latent_notes[i,j] = val
        notes[i,j] = gnote(gammas, val)
        
## Saving data
#with open('sim_data_docs', 'wb') as fp:
#    pickle.dump(docs, fp)

## test (it works!)
# with open ('sim_data_docs', 'rb') as fp:
#    itemlist = pickle.load(fp)

# np.save('sim_data_notes', notes)
# np.save('sim_data_notes_latent', latent_notes)
np.save('clc', clc)
np.save('clr', clr)

from sklearn.cluster import KMeans
from  sklearn.metrics import adjusted_rand_score as ari

# direct k-means on Y
kclr = KMeans(n_clusters = 2).fit(notes)
kclc = KMeans(n_clusters = 2).fit(notes.transpose())

ari(clr, kclr.labels_)
ari(kclc.labels_, clc)

# direct k-means on Z
kclr = KMeans(n_clusters = 2).fit(latent_notes)
kclc = KMeans(n_clusters = 2).fit(latent_notes.transpose())

ari(clr, kclr.labels_)
ari(kclc.labels_, clc)

######### Universal format to be read by the Neural Net
seq_cols = np.arange(0,P)
seq_rows = np.arange(0,M)

with open('universal_simu.txt', 'w') as f:
    f.write("Individual,Object,Note,Text\n")
    for row in seq_rows:
        for col in seq_cols:
            pos = row*P + col
            f.write("%s," % row)
            f.write("%s," % col)
            f.write("%s," % notes[row, col])
            for word in docs[pos]:
                f.write(" %s" % word)
            f.write("\n")

######### Writing a text file for the HFT executable (McAuley, Lesckovec, 2013)
# seq_rows = np.arange(0,M)
# shuffle(seq_rows)

# seq_cols = np.arange(0,P)
# shuffle(seq_cols)

# with open('first_attempt.txt', 'w') as f:
#     for row in seq_rows:
#         for col in seq_cols:
#             pos = row*P + col
#             f.write("%s " % row)
#             f.write("%s " % col)
#             f.write("%s " % notes[row, col])
#             f.write("%s " %  int(np.round(1 + 1000000*np.random.rand())))
#             f.write("%s" % len(docs[pos]))
#             for word in docs[pos]:
#                 f.write(" %s" % word)
#             f.write("\n")





