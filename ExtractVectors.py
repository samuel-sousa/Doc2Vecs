#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:33:57 2019

@author: samuel
"""
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import KeyedVectors
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#load model
model = Doc2Vec.load('doc2vec250.model')

#read semcor
semcor = pd.read_csv('ExamplesSemcor.csv',sep=',',header=0)
answer = []
sensesPlot = {}

#insert the sentences
testData = ['art change ringing be peculiar english', 'be easy know be ancient art rope']
''',
            'lose be hard master thing',
            'Great pick where nature end',
            'Vision be see be invisible other',
            'be losing game']'''

#tokenize sentences
for sentence in testData:
    tokens = word_tokenize(sentence.lower())
    for token in tokens:
        try:
            vec = model.wv[token]
            answer.append(np.append(vec,[token]))
        except KeyError:
            pass
    #print(tokens)
    #vec = model.infer_vector(tokens)
    #print(vec)
    #answer.append(np.append(vec,[testData.index(sentence)]))

#read model
docvecs = model.docvecs

#get lemmas
lemmas = {'art%1:04:00::':1274,'art%1:06:00::':1275,'art%1:09:00::':1276,'art%1:10:00::':1277}#'love%1:09:00::':15103,'love%1:12:00::':15104,'love%1:18:00::':15105,'love%2:37:00::':15106,'love%2:37:01::':15107,'love%2:37:02::':15108}

#get index of the lemmas
'''semcorSenses = semcor['SENSE'].values.tolist()
print(semcorSenses)
for senseSemcor in semcorSenses:
    try:
        sensesPlot[senseSemcor] = lemmas.index(senseSemcor)
    except AttributeError:
        pass
'''

for sense in lemmas:
    vec =  docvecs[lemmas[sense]]
    print(vec)
    answer.append(np.append(vec,[sense]))
print(answer)


df = pd.DataFrame(answer)
print(df)

X = df.iloc[:,0:250]
labels = df.iloc[:,250].values.tolist()

pca = PCA(n_components=2)
pca.fit(X)
reduced = pca.transform(X)

for index,vec in enumerate(reduced):
    # print ('%s %s'%(words_label[index],vec))
    if index <100:
        x,y=vec[0],vec[1]
        plt.scatter(x,y)
        plt.annotate(labels[index],xy=(x,y))
plt.show()


    

    
