# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from config import Config




def loadData(cfg):
    data = pd.read_csv(cfg.dataFile)
    rawData = data.copy()
    data = data.loc[:,cfg.keeps]
    dtypes = list(data.dtypes)
    names  = list(data.columns)
    for i,dt in enumerate(dtypes):
        if dt == bool:
            data[names[i]] = data[names[i]].map({True:1,False:0})
    data['SubjectID'] = data['SubjectID'].map(cfg.subj2id)
    data['ProblemID'] = data['ProblemID'].map(cfg.prob2id)
    data_combined = {}
    data_combined_raw = {}
    combineIndex = names.index(cfg.combineBy)
    for i in range(data.shape[0]):
        log = list(data.iloc[i,:].values)
        log_raw = list(rawData.iloc[i,:].values)
        key = log[combineIndex]
        if key not in data_combined:
            data_combined[key]=[]
            data_combined_raw[key]=[]
        data_combined[key].append(log)
        data_combined_raw[key].append(log_raw)
    
    data_dealt = []
    data_raw   = []
    for key in data_combined:
        data_dealt.append(np.array(data_combined[key]))
        data_raw.append(data_combined_raw[key])
    
    
    
    return data_raw,data_dealt
        
class DataFeeder():
    def __init__(self,config):
        self.dataRaw,self.data = loadData(config)
        self.size = self.data.__len__()
        self.cfg  = config
        self.nump   = config.numP
    def init_feat(self,shape,dtype=np.int32):
        return np.zeros(shape,dtype=dtype)
    def next_batch(self,
                   batch_size,cum = True):
        ridxs = np.random.choice(self.size ,
                                 size = batch_size, 
                                 replace = False)
        #gather and shuffle the data
        data_gathered = []
        #data_check    = []
        for rid in ridxs:
            tmp = np.copy(self.data[rid])
            #data_check.append(self.data[rid])
            np.random.shuffle(tmp)
            data_gathered.append(tmp)
        
        lengths     = self.init_feat([batch_size])
        indFeatures = {name:self.init_feat([batch_size]) for name in self.cfg.indFeatures}
        for i,data in enumerate(data_gathered):
            for name,ind in self.cfg.indFeatIndices.items():
                indFeatures[name][i] = data[0,ind]
            lengths[i] = data.shape[0]
        
        max_seq_len = lengths.max()
        shape2 = [batch_size , max_seq_len+1]
        shape3 = [batch_size , max_seq_len+1,self.nump]
        
        #build inputs and targets
        #and masks
        masks    = self.init_feat(shape3,np.float32)
        inpFeatures = {name:self.init_feat(shape2) for name in self.cfg.inpFeatures}
        tgtFeatures = {name:self.init_feat(shape3) for name in self.cfg.tgtFeatures}
        for i,data in enumerate(data_gathered):
            for name,ind in self.cfg.inpFeatIndices.items():
                inpFeatures[name][i,1:lengths[i]+1] = data[:,ind]
            for name,ind in self.cfg.tgtFeatIndices.items():
                tgtFeatures[name][i,np.arange(1,lengths[i]+1),data[:,self.cfg.maskIndex]] = data[:,ind]
            masks[i,np.arange(1,lengths[i]+1),data[:,self.cfg.maskIndex]] = 1.0
        
        #slice to make inputs and targets
        for name in inpFeatures:
            inpFeatures[name] = inpFeatures[name][:,:-1]
        for name in tgtFeatures:
            tgtFeatures[name] = tgtFeatures[name][:,1:,:]
        masks = masks[:,1:,:]
        backup_masks = np.copy(masks)
        if cum:
            for name in tgtFeatures:
                np.cumsum(tgtFeatures[name] , axis=1 , out = tgtFeatures[name])
            np.cumsum(masks,axis = 1, out = masks)
        #merge
        inpFeatures.update(indFeatures)
        inpFeatures.update({'masks':masks,'lengths':lengths})
        return inpFeatures,tgtFeatures,backup_masks.astype(np.bool)
        
    def next_one(self):
        for d in self.data:
            max_seq_len = d.shape[0]
            lengths  = np.array([d.shape[0]],dtype  = np.int32)
            indFeatures = {name:self.init_feat([1]) for name in self.cfg.indFeatures}
            inpFeatrues = {name:self.init_feat([1,max_seq_len+1]) for name in self.cfg.inpFeatures}
            seqIndices  = d[:,self.cfg.maskIndex]
            masks       = self.init_feat([1,max_seq_len,self.nump])
            masks[0,np.arange(lengths[0]),d[:,self.cfg.maskIndex]] = 1.0
            for name,ind in self.cfg.indFeatIndices.items():
                indFeatures[name][0] = d[0,ind]
            for name,ind in self.cfg.inpFeatIndices.items():
                inpFeatrues[name][0,1:1+lengths[0]] = d[:,ind]
            #seqIndices = d[:,self.cfg.maskIndex]
            #slice 
            for name,ind in inpFeatrues.items():
                inpFeatrues[name] = inpFeatrues[name][:,:-1]
            tgtFeatures = {name:self.init_feat([1,max_seq_len]) for name in self.cfg.tgtFeatures}
            for name,ind in self.cfg.tgtFeatIndices.items():
                tgtFeatures[name][0,:lengths[0]] = d[:,ind]
            inpFeatrues.update(indFeatures)
            inpFeatrues.update({'lengths':lengths,'masks':masks})
            yield inpFeatrues,tgtFeatures,seqIndices
            
class TestData():
    def __init__(self,cfg):
        self.data = pd.read_csv(cfg.dataFile)
        self.cfg  = cfg
    def export(self):
        values = list(set(self.data.loc[:,self.cfg.combineBy].values))
        for value in values:
            data = self.data.loc[self.data.loc[:,self.cfg.combineBy]==value,:]
            yield data,self.convert2DKTInput(data)
            
    def init_feat(self,shape,dtype=np.int32):
        return np.zeros(shape,dtype=dtype)
    
    def convert2DKTInput(self,data):
        data = data.copy()
        data = data.loc[:,self.cfg.keeps]
        labels = self.cfg.keeps
        dtypes = list(data.dtypes)
        
        for i,dt in enumerate(dtypes):
            if dt == bool:
                data[labels[i]] = data[labels[i]].map({True:1,False:0})
        
        data['SubjectID'] = data['SubjectID'].map(self.cfg.subj2id)
        data['ProblemID'] = data['ProblemID'].map(self.cfg.prob2id)
        lengths  = np.array([data.shape[0]],dtype=np.int32)
        max_seq_len = lengths.max()
        
        indFeats = {name:self.init_feat([1]) for name in self.cfg.indFeatures}
        inpFeats = {name:self.init_feat([1,max_seq_len]) for name in self.cfg.inpFeatures}
        targets  = {name:self.init_feat([1,max_seq_len]) for name in self.cfg.tgtFeatures}
        seqIndices = data.iloc[:,self.cfg.maskIndex].values
        
        for name,ind in self.cfg.indFeatIndices.items():
            indFeats[name][0] = data.iloc[0,ind]
        for name,ind in self.cfg.inpFeatIndices.items():
            inpFeats[name][0,1:lengths[0]]   = data.iloc[:-1,ind].values
        for name,ind in self.cfg.tgtFeatIndices.items():
            targets[name][0,:] = data.iloc[:,ind].values
        masks = self.init_feat([1,max_seq_len , self.cfg.numP])
        inpFeats['lengths'] = lengths
        inpFeats['masks']   = masks
        inpFeats.update(indFeats)
        return inpFeats,targets,seqIndices
        
            
            


if __name__=='__main__':
    cfg = Config()
    cfg.load()
    for d in TestData(cfg).export():
        break