#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:59:47 2019

@author: marco
"""

#import nltk
import pickle
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
B = tokenizer.tokenize(open('msgB.txt', encoding="utf8").read())
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
# corpus2 = [A,B,C,A]
# corpus3 = [A,B,B,A]
corpus4 = [D,B,C,D]
corpus5 = [A,C,C,A]

# dct = Dictionary(corpus)
# dct2 = Dictionary(corpus2)
# dct3 = Dictionary(corpus3)
# dct4 = Dictionary(corpus4)
# dct5 = Dictionary(corpus5)
# dct4.save("dizionario_BCD.pkl")
# dct[72]
# dct.token2id["baby"]
dct = pickle.load(open('dizionario.pkl','rb'))
dctn = dct.token2id

M = 100
P = 600
Q = 2
L = 2

clr = np.random.choice(range(Q), M)
clc = np.random.choice(range(L), P)
# clr = np.load('clr_750_600.npy')
# clc = np.load('clc_750_600.npy')

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
            sampled_pos = sample(range(len(corpus4[pos])), idy)
            for idz in sampled_pos:
                # print(idz)
                msg.append(corpus4[pos][idz])
            pos += 1
        docs.append(msg)
        # notes sampling
        val = np.random.normal(mu[cli, clj], sd[cli, clj])
        latent_notes[i,j] = val
        notes[i,j] = gnote(gammas, val)
        
## Saving data
with open('sim_data_docs_100_600', 'wb') as fp:
    pickle.dump(docs, fp)

## test (it works!)
# with open ('sim_data_docs', 'rb') as fp:
#    itemlist = pickle.load(fp)

np.save('sim_data_notes_100_600', notes)
# np.save('sim_data_notes_latent', latent_notes)
np.save('clc_100_600', clc)
np.save('clr_100_600', clr)

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

######### Writing a text file for the HFT executable (McAuley, Lesckovec, 2013)
# seq_rows = np.arange(0,M)
# shuffle(seq_rows)
#
# seq_cols = np.arange(0,P)
# shuffle(seq_cols)
#
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


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Create linear regression object
regr = LinearRegression()
regr1 = LinearRegression()
regr2 = LinearRegression()
time = [464.52, 637.96, 732.83, 848.79, 1111.57]
words = [325, 558, 721, 920, 1034]
time1 = [239.08, 296.21, 398.93, 693.81, 1111.57]
users = [100, 150, 250, 500, 750]
time2 = [190.47, 274.47, 415.81, 729.67, 1111.57]
items = [50, 100, 200, 400, 600]

regr.fit(np.array(words).reshape(-1, 1), np.array(time).reshape(-1, 1))
regr1.fit(np.array(users).reshape(-1, 1), np.array(time1).reshape(-1, 1))
regr2.fit(np.array(items).reshape(-1, 1), np.array(time2).reshape(-1, 1))

# Make predictions using the testing set
y_pred = regr.predict(np.array(words).reshape(-1, 1))
y_pred1 = regr1.predict(np.array(users).reshape(-1, 1))
y_pred2 = regr2.predict(np.array(items).reshape(-1, 1))

# The coefficients
print('Coefficients: \n', regr.coef_)
print('Coefficients1: \n', regr1.coef_)
print('Coefficients0: \n', regr2.coef_)

plt.figure()

# words
plt.subplot(221)
plt.scatter(words, time,  color='black')
plt.loglog(words, y_pred, color='blue', linewidth=3, label='Coefficient = %.4f' %regr.coef_)
plt.xlabel('number of words (log)',fontsize=11)
plt.ylabel('training time (log)',fontsize=11)
plt.legend(loc='upper left')
plt.title('(a)')

# users
plt.subplot(222)
plt.scatter(users, time1,  color='black')
plt.loglog(users, y_pred1, color='blue', linewidth=3, label='Coefficient = %.4f' %regr1.coef_)
plt.xlabel('number of users (log)',fontsize=11)
plt.ylabel('training time (log)',fontsize=11)
plt.legend(loc='upper left')
plt.title('(b)')

# items
plt.subplot(223)
plt.scatter(items, time2,  color='black')
plt.loglog(items, y_pred2, color='blue', linewidth=3, label='Coefficient = %.4f' %regr2.coef_)
plt.xlabel('number of items (log)',fontsize=11)
plt.ylabel('training time (log)',fontsize=11)
plt.legend(loc='upper left')
plt.title('(c)')

plt.show()

# Plot outputs
plt.scatter(items, time2,  color='black')
plt.loglog(items, y_pred, color='blue', linewidth=3, label='Coefficient = 1.6539')
plt.xlabel('number of items (log)',fontsize=11)
plt.ylabel('training time (log)',fontsize=11)
# plt.xticks(())
# plt.yticks(())
# plt.xscale('log')
# plt.yscale('log')
plt.legend(loc='upper left')
plt.show()

f, ax = plt.subplots(1,figsize=(15,10))
time = [464.52, 637.96, 732.83, 848.79, 1111.57]
words = [325, 558, 721, 920, 1034]
ax.set_xscale('log')
ax.set_yscale('log')
sns.regplot(x=words, y=time)

ax.set_xscale('log')
ax.set_yscale('log')
ax.scatter(words, time, color='r', marker = 's')
plt.show()

reg = LinearRegression().fit(words, time)
reg.score(words, time)


