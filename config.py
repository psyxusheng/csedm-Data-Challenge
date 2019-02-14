# -*- coding: utf-8 -*-

import json
import numpy as np


class Config():
    def __init__(self,**args):
        self.dataFile       = '../CV/Fold0/Training.csv'
        self.problem2idFile = 'problems.jsn'
        self.subject2idFile = 'subjects.jsn'
        self.subjectVecFile = None
        self.problemVecFile = 'problemVecs.npy'
        self.keeps          = ["SubjectID","ProblemID",
                          "FirstCorrect","EverCorrect",
                          "UsedHint"]
        self.combineBy      = 'SubjectID'
        self.maskBy         = 'ProblemID'
        self.embSize        = 16
        self.indFeatures    = ['SubjectID']
        self.inpFeatures    = ['ProblemID','FirstCorrect','EverCorrect','UsedHint']
        self.tgtFeatures    = ['FirstCorrect']
        for key,value in args.items():
            setattr(self,key,value)
    
    
    def load(self):
        
        prob2id = json.load(open(self.problem2idFile,'r',
                               encoding  ='utf8')) 
        subj2id = json.load(open(self.subject2idFile,'r',
                                 encoding = 'utf8'))
        
        self.id2prob = {id_:prob for prob,id_ in prob2id.items()}
        self.id2subj = {id_:subj for subj,id_ in subj2id.items()}
        
        self.numP     = prob2id.__len__()
        self.numS     = subj2id.__len__()
        if self.problem2idFile:
            probVecs = np.load(self.problemVecFile)
        else:
            probVecs = [self.numP , self.embSize]
        if self.subjectVecFile:
            subjVecs =np.load(self.subjectVecFile)
        else:
            subjVecs = [self.numS, self.embSize]
        self.prob2id  = prob2id
        self.probVecs = probVecs
        self.subj2id  = subj2id
        self.subjVecs = subjVecs
        
        self.indFeatIndices = {name:self.keeps.index(name) for name in self.indFeatures}
        self.inpFeatIndices = {name:self.keeps.index(name) for name in self.inpFeatures}
        self.tgtFeatIndices = {name:self.keeps.index(name) for name in self.tgtFeatures}
        self.maskIndex      = self.keeps.index(self.maskBy)
