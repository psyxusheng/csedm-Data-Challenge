# -*- coding: utf-8 -*-

import re,math,json,random
import numpy as np
from gensim.models import LsiModel,TfidfModel
from gensim.corpora import Dictionary
import pandas as pd

class Parameters():
    mapping_keys = ['SubjectID','ProblemID']
    reg4code  = re.compile('[a-zA-Z0-9]+|[\+\-\*/><=\!\|\~\!\.\^\&]+')

def loadvocab(vocabFile):
    tmp = open('vocab.txt','r',encoding='utf8').read().split('\n')
    tmp = [i.strip() for i in tmp if i.strip()!='']
    return {w:i for i,w in enumerate(tmp)}
    

def code2vocab(code,vocab = None):
    if isinstance(code,str):
        if vocab:
            words = Parameters.reg4code.findall(code)
            return list(filter(lambda x:x in vocab ,words))
        else:
            return Parameters.reg4code.findall(code)
    if math.isnan(code):
        return ['']

def loadFiles(mainTableFile,codeStateFile):
    # do two things : make vocab mapping for variables such like problem id
    # do pretrain for subjects and problems
    mainData = pd.read_csv(mainTableFile)
    codeData = pd.read_csv(codeStateFile)
    #using learner - attemps to embedding learners
    #using correctCode - task to embedding  each correct code and 
    #   then using averarge of right codes to represent each problem
    return mainData , codeData

def get_mappings(data,mapping_keys=['SubjectID','ProblemID']):
    mappings = dict()
    for name in mapping_keys:
        mapping = {'unk':1,'pad':0}
        values = list(set(data.loc[:,name].values))
        for c in values:
            mapping[c] = mapping.__len__()
        mappings[name] = mapping
    return mappings

def tofloat(ndarray):
    return list(map(float,ndarray))
        

def lsa(corpus,size = 8):
    dic = Dictionary(corpus)
    dic.filter_extremes(no_below = 5 , no_above = 0.8,)
    dic.filter_n_most_frequent(remove_n=10)
    dic.compactify()
    index_corpus = [dic.doc2bow(sent) for sent in corpus]
    tfidf = TfidfModel(index_corpus,dictionary=dic)
    normed_corpus = [tfidf[sent] for sent in index_corpus]
    lsi = LsiModel(normed_corpus , num_topics = size)
    return [[x[1] for x in lsi[sent]] for sent in normed_corpus]
    


def main(mainTableFile,codeStateFile,vocabFile = None,embedSize = 8):
    maindata,codes = loadFiles(mainTableFile,codeStateFile)
    #first using word2vec building basic word vectors
    codes = list(codes.loc[:,'Code'].values)
    if vocabFile : 
        vocab = loadvocab(vocabFile)
    else:
        vocab = None
    corpus = [code2vocab(code,vocab) for code in codes]
    vectors = lsa(corpus,embedSize)
    prob_vecs = {}
    for i in range(maindata.shape[0]):
        codeID,pn,crct = maindata.loc[i,['CodeStateID','ProblemID','Correct']].values
        if not crct:
            continue
        if pn not in prob_vecs:
            prob_vecs[pn] = {}
        if codeID not in prob_vecs[pn]:
            if vectors[codeID-1]==[]:
                continue
            else:
                prob_vecs[pn][codeID]=vectors[codeID-1]
    prob2id = {'pad':0,
              'unk':1}
    probVecs =[[0.0]*embedSize,
               [random.random() for i in range(embedSize)]]
    for key,vecs in prob_vecs.items():
        print(key,prob_vecs[key].__len__())
        meanVec = np.array(list(vecs.values())).reshape([-1,embedSize]).mean(0)
        prob2id[key] = prob2id.__len__()
        probVecs.append(meanVec)
    json.dump(prob2id,open('problems.jsn','w',
                   encoding='utf8'),ensure_ascii=False)
    np.save('problemVecs',np.array(probVecs))
    
    subjs = set(maindata.loc[:,'SubjectID'].values)
    subj2id = {'unk':0}
    for key in subjs:
        subj2id[key]= subj2id.__len__()
    json.dump(subj2id,open('subjects.jsn','w',
                   encoding='utf8'),ensure_ascii=False)    
          
        
    return probVecs



if __name__=='__main__':
    mainTableFile = '../MainTable.csv'
    codeStateFile = '../CodeState.csv'
    embeddingSize = 8
    x= main(mainTableFile,codeStateFile,None,embeddingSize)