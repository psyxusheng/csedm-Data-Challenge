# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import csv
from sklearn.metrics import confusion_matrix,f1_score , precision_score , recall_score , cohen_kappa_score


tasks = '''helloWorld
doubleX
raiseToPower
convertToDegrees
leftoverCandy
intToFloat
findRoot
howManyEggCartons
kthDigit
nearestBusStop
hasTwoDigits
overNineThousand
canDrinkAlcohol
isPunctuation
oneToN
backwardsCombine
isEvenPositiveInt
firstAndLast
singlePigLatin'''.split('\n')



filename = 'cv_predict.csv'


data = pd.read_csv(filename)
data['FirstCorrect'] = data['FirstCorrect'].map({True:1,False:0})
folds_indices = []
for i in range(10):# 10 folds
    fold_rows = data.loc[:,'fold'] == i
    y_pred    = data.loc[fold_rows,'prediction'].values
    y_true    = data.loc[fold_rows,'FirstCorrect'].values
    y_pred = y_pred.astype(np.bool)
    y_true = y_true.astype(np.bool)
    tp = np.mean(y_true & y_pred)
    tn = np.mean(~y_true & ~y_pred)
    fp = np.mean(~y_true & y_pred)
    fn = np.mean(y_true & ~y_pred)
    tp = float(tp)
    tn = float(tn)
    fp = float(fp)
    fn = float(fn)
    accuracy = tp + tn
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    kap = cohen_kappa_score(y_true,y_pred)
    folds_indices.append([y_true.mean(),y_pred.mean(),tn,fp,
                          fn,tp,precision,recall,f1,kap])

folds_indices = list(np.array(folds_indices,dtype=np.float).mean(0))

fobj = open('evaluation_overall.csv','w',newline='')
writer = csv.writer(fobj)
writer.writerow(['pCorrect','pPredicted',
                 'tp','tn','fp','fn','precision','recall','f1','kappa'])
writer.writerow(folds_indices) 
fobj.close()  
  
fobj = open('evaluation_by_problem.csv','w',newline='')
writer = csv.writer(fobj)
writer.writerow(['ProblemID','pCorrect','pPredicted',
                 'tp','tn','fp','fn','precision','recall','f1','kappa'])
tasks_indices = []
for tk in tasks:
    tk_rows = data.loc[:,'ProblemID'] == tk
    y_pred    = data.loc[tk_rows,'prediction'].values
    y_true    = data.loc[tk_rows,'FirstCorrect'].values
    y_pred = y_pred.astype(np.bool)
    y_true = y_true.astype(np.bool)
    tp = np.mean(y_true & y_pred)
    tn = np.mean(~y_true & ~y_pred)
    fp = np.mean(~y_true & y_pred)
    fn = np.mean(y_true & ~y_pred)
    accuracy = tp + tn
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    kap = cohen_kappa_score(y_true,y_pred)
    writer.writerow([tk,float(y_true.mean()),
                     float(y_pred.mean()),tp,tn,fp,tn,
                             precision,recall,f1,kap])
fobj.close()
        




