#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:15:10 2019

@author: samuel
"""
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
import time

#Select the context of a word in semcor
def selectContext(value):
    dataFrame = pd.read_csv('SWORDSSemcor.csv',sep=',',header=0)
    result = dataFrame.loc[dataFrame['ID'] == value]
    result = np.ravel(result.iloc[:,:9].values).tolist()
    result = [str(item) for item in result]
    nullWords = ['NA','nan','na']
    return [item for item in result if item not in nullWords]

#Validate a key
def validateKeys(keys):
    validkeys = []
    for key in keys:
        lemma = key.split('%')[0]
        lemmas = wn.lemmas(lemma)
        trueKeys = [lemma.key() for lemma in lemmas]
        if key in trueKeys:
            validkeys.append(key)
        else:
            pass
    return validkeys

#Select semcor labels [].unique
def getKeys(df):
    keys = np.unique(df['SENSE'].values)
    keys = keys.tolist()
    validKeys = validateKeys(keys)
    print(validKeys)
    return validKeys

#Select all the surrounding words of this sense in semcor
def selectSurroundingWords(keysList,df):
    data = []
    for senseKey in keysList:
        results = df.loc[df['SENSE'] == senseKey]
        wordDefinitionAndUsage = []
        #Append synset and examples from WordNet
        synset =  wn.lemma_from_key(senseKey).synset()
        examples = synset.examples()
        wordDefinitionAndUsage.append(synset.definition())
        wordDefinitionAndUsage.append(','.join(examples))
        #Look for exampples on Semcor
        for index, row in results.iterrows():
            swords = selectContext(row['ID'])
            context = []
            [context.append(word) for word in swords[:5]]
            context.append(row['LEMMA'])
            [context.append(word) for word in swords[5:]]
            wordDefinitionAndUsage.append(' '.join(context))
        #print(wordDefinitionAndUsage)
        text = ' '.join(wordDefinitionAndUsage)
        data.append([text,senseKey])
    answerDF = pd.DataFrame(data,columns=['TEXT','SENSE'])
    answerDF.to_csv('ExamplesSemcor.csv',sep=',',header=True,index=False)
    return answerDF
#Run doc2Vec
def initDoc2Vec(df):
    #Extract doc2Vec for all senses in the list
    data = df['TEXT'].values.tolist()
    #print(data)
    #here the words are tokenized
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
    print(tagged_data)
    #here the doc2vec parameters are set up
    max_epochs = 100
    vec_size = 250
    alpha = 0.025
    #here the model is initialized
    model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
    model.save("doc2vec250.model")
    print("Model Saved")

def main():
    startTime = time.time()
    semcorArch = 'Semcor-processed.csv'
    semcorDF = pd.read_csv(semcorArch,sep=',',header=0)
    #semcorDF = semcorDF.iloc[:10,:]
    senseKeysList = getKeys(semcorDF)
    currentTime = time.time()
    print('Lenght of keys list: {}. Time: {}.'.format(len(senseKeysList),(currentTime-startTime)))
    answerData = selectSurroundingWords(senseKeysList,semcorDF)
    currentTime = time.time()
    print('Size od Data Set: {}. Time: {}.'.format(answerData.iloc[:,0].count(),(currentTime-startTime)))
    initDoc2Vec(answerData)
    currentTime = time.time()
    print('Model Trained. Time: {}.'.format((currentTime-startTime)))
    
if __name__=='__main__':
    main()
    