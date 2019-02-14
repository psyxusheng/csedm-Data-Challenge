# -*- coding: utf-8 -*-

import numpy as np
import csv
import tensorflow as tf
from config import Config
from DataFeeder import DataFeeder,TestData
from model import DKT
from sklearn.metrics import f1_score,precision_score,recall_score

indices = [precision_score,recall_score,f1_score]

def make_prediction(folderName,index,max_iters = 200,target_key = 'FirstCorrect'):
    tf.reset_default_graph()
    cfg = Config(dataFile = '%s/Training.csv'%folderName)
    cfg.load()
    DF_train = DataFeeder(cfg)
    # problem vectors  cfg.probVecs
    features = [['ProblemID','inp',[cfg.numP,8],False],
                ['FirstCorrect','inp',[2,8],True],
                ['EverCorrect','inp',[2,8],True],
                ['UsedHint','inp',[2,8],True]]
    targets  = [['FirstCorrect',2 , 1. , [1., 1.2]]]
    
    model4train = DKT(features = features,
                      targets   = targets,
                      keep_prob = 0.1,
                      num_items = cfg.numP, 
                      rnn_units = [32,32],
                      training  = True,
                      lr_decay  = [1e-3,0.9,50])
    
    model4test = DKT(features  = features,
                     targets   = targets,
                     keep_prob = 1.,
                     num_items = cfg.numP, 
                     rnn_units = [32,32],
                     training  = False,
                     lr_decay  = [5*1e-2,0.9,100])
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    print('training on %s'%folderName)
    for i in range(1,max_iters+1):
        inputs,targets,bu_masks = DF_train.next_batch(batch_size = DF_train.size,
                                                      cum = True)
        feed_data = model4train.zip_data(inputs,model4train.input_holders)
        feed_data_t = model4train.zip_data(targets,model4train.target_holders)
        feed_data.update(feed_data_t)
        
        _,predicts,costs = session.run([model4train.trainop,
                                        model4train.predicts,
                                        model4train.costs] ,
                                       feed_dict=feed_data)
        if i%max_iters == 0:
            for name,values in predicts.items():
#                y_pred = values[bu_masks]
#                y_true = targets[name][bu_masks]
#                indices = [func(y_true,y_pred) for func in evalue_indices]
                print('final cost',round(costs[target_key],3))
    cfg_test = Config(dataFile = '%s/Test.csv'%folderName)
    cfg_test.load()
    TD       = TestData(cfg_test)
    result = []
    predictions = []
    groundtruth = []
    for data,(inputs,targets,seqIndices) in TD.export():
        feed_data = model4test.zip_data(inputs,model4test.input_holders)
        predicts,probablities  = session.run([model4test.predicts,
                                 model4test.probablities],feed_dict = feed_data)
            
        probs_on_correct = probablities[target_key][0,np.arange(inputs['lengths'][0]),seqIndices,1]
        y_pred           = predicts[target_key][0,np.arange(inputs['lengths'][0]),seqIndices]
        y_true           = targets[target_key][0,:]
        predictions.append(y_pred)
        groundtruth.append(y_true)
        for i in range(data.shape[0]):
            raw_data = list(data.iloc[i,:].values)
            raw_data +=[float(probs_on_correct[i]) , int(y_pred[i]) , index]
            result.append(raw_data)
    y_true = np.concatenate(groundtruth,axis=0)
    y_pred = np.concatenate(predictions,axis=0)
    index = [round(func(y_true,y_pred),3) for func in indices]
    print(' '*4,'testing',index)
    return result,list(data.columns)
        

def main(datafolder):
    total_predicts = []
    for i in range(10): 
        predicts,labels = make_prediction(folderName = datafolder+'/fold%d'%i,
                                   index = i,
                                   max_iters  = 400)
        total_predicts.extend(predicts)
    
    fobj = open('cv_predict.csv','w',newline='')
    writer = csv.writer(fobj)
    writer.writerow(labels+['pCorrectProblem','prediction','fold'])
    for line in total_predicts:
        writer.writerow(line)
    fobj.close()
    return True


if __name__=='__main__':
    dataFolder = r'C:\Users\G7\Desktop\itemRL\DataChellenge\CV'
    main(dataFolder)
