# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:51:01 2020

@author: Dingge
"""

import numpy as np
import pandas as pd
import csv

# complete dtm
# dtm = np.genfromtxt('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/amazon/dtm.csv', 
#                     skip_header=1, delimiter=",") 

# idconnect = np.genfromtxt('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/amazon/idConnected.csv', 
#                     skip_header=1, delimiter=",") 

# df1 = pd.DataFrame(data = dtm)
# df2 = pd.DataFrame(data = idconnect)
# df2.columns = ['id_i', 'id_j', 'row']
# df3 = df2.sort_values(by='id_j' , ascending=True)

# df3.to_csv('out.csv', index=False)
# connection = np.genfromtxt('out.csv', skip_header=1, delimiter=",") 

#connection = df3.to_numpy()
#df = pd.concat([df1, df2])


# with open('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/amazon/AmazonDa.txt', "w") as my_output_file:
#     with open('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/amazon/AmazonData.csv', "r") as my_input_file:
#         [my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file, delimiter=',', quotechar='"',
#                            quoting=csv.QUOTE_MINIMAL)]
#     my_output_file.close()

import re
from nltk.corpus import stopwords 

# cols = [0,1,2,3,4]  # column index numbers to be extracted
# extracted = []
# remove_chars = '[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
# stop_words = set(stopwords.words('english')) 

# with open('C:/Users/Dingge/Doctoral_projets/Pytorch/ToServer/ToServer/amazon/AmazonData.csv', newline='') as csvfin:
#     csvReader = csv.reader(csvfin, delimiter=',', quotechar='"',
#                             quoting=csv.QUOTE_MINIMAL)
#     with open('output2.csv', 'a', newline='') as csvfout:
#         csvWriter = csv.writer(csvfout, delimiter=',', quotechar='"',
#                                 quoting=csv.QUOTE_MINIMAL)
#         # for row in csvReader:               
#         #     extracted.append(row[4].lower())
#         #     print(extracted)
#         #     remove_chars = '[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
#         #     stop_words = set(stopwords.words('english')) 
#         #     extracted = [re.sub(remove_chars, '', text) for text in extracted]
#         #     # extracted = [re.sub('\d+', 'number', change) for change in extracted]
#         #     csvWriter.writerow(extracted)
#         #     extracted = []
            
#         for row in csvReader:
#             for col_num in cols:
#                 extracted.append(row[col_num].lower())      
#                 extracted = [re.sub(remove_chars, '', text) for text in extracted]
#                 extracted = [' '.join(text.split()) for text in extracted]
#                 # extracted = [w for w in extracted if not w in stop_words]
#                 # extracted = [re.sub('\d+', 'number', change) for change in extracted]                                 
#             csvWriter.writerow(extracted)
#             extracted = []

extra = []
number = []            
with open('output2.csv', newline='') as csvfin:
    csvReader = csv.reader(csvfin, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in csvReader:
        extra = row[4].split()
        # print(extra, len(extra))
        number.append(len(extra))

num = []                    
df = pd.read_csv('AmazonData.csv') 
df['Text'] = df['Text'].str.replace(r'[^\w\s]+', '')
df['Text'] = df['Text'].str.replace(r' +', ' ')
df['Text'] = df['Text'].str.lower()
sp = df['Text'].str.split().tolist()
for item in range(36443):
    print(sp[item])
    num.append(len(sp[item])) ##### 需要删除 NAN 的text ！！！！
      