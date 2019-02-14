# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from config import Config
from DataFeeder import DataFeeder
from model import DKT
from sklearn.metrics import f1_score, precision_score ,cohen_kappa_score, recall_score


def main(folderName,max_iters = 200):
    tf.reset_default_graph()
    cfg = Config()
    cfg.dataFile = '../CV/%s/Training.csv'%folderName
    cfg.load()
    DF_train = DataFeeder(cfg)
    
    cfg.dataFile = '../CV/%s/Test.csv'%folderName
    DF_test = DataFeeder(cfg)
    # problem vectors  cfg.probVecs
    features = [['ProblemID','inp',[40,8],True],
                ['FirstCorrect','inp',[2,8],True],
                ['EverCorrect','inp',[2,8],True],
                ['UsedHint','inp',[2,8],True]]
    targets  = [['FirstCorrect',2 , 1. , [.4, .6]]]
    model4train = DKT(features = features,
                      targets   = targets,
                      keep_prob = 0.1,
                      num_items = cfg.numP, 
                      rnn_units = [32,8],
                      training  = True,
                      lr_decay  = [5*1e-2,0.9,50])
    
    model4test = DKT(features  = features,
                     targets   = targets,
                     keep_prob = 1.,
                     num_items = cfg.numP, 
                     rnn_units = [32,8],
                     training  = False,
                     lr_decay  = [5*1e-2,0.9,100])
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    for i in range(max_iters):
        inputs,targets,bu_masks = DF_train.next_batch(batch_size = DF_train.size,
                                                     cum = True)
        feed_data = model4train.zip_data(inputs,model4train.input_holders)
        feed_data_t = model4train.zip_data(targets,model4train.target_holders)
        feed_data.update(feed_data_t)
        
        _,predicts,costs = session.run([model4train.trainop,
                                        model4train.predicts,
                                        model4train.costs] ,
                                       feed_dict=feed_data)
        #if i%(max_iters // 10) == 0:
#    masked_positions = bu_masks.astype(np.bool)
#    for name,data in targets.items():
#        ytrue = data[masked_positions]
#        ypred = predicts[name][masked_positions]
#        report = classification_report(ytrue , ypred)
#        print('Training result')
#        print(name, 'trained %d'%i , costs[name],'\n', report)
            
     
    
    
    y_true = {}
    y_pred = {}
    
    for inputs,targets,seqIndices in DF_test.next_one():
        feed_data = model4test.zip_data(inputs,model4test.input_holders)
        predicts  = session.run(model4test.predicts,feed_dict = feed_data)
        for name,d in targets.items():
            if name not in y_true:
                y_true[name]=[]
                y_pred[name] =[]
            y_true[name].append(d[0])
            preds = predicts[name][inputs['masks'].astype(np.bool)]
            y_pred[name].append(preds)
    result = {}
    for name,d in y_true.items():
        y_ture_tmp = np.concatenate(d)
        y_pred_tmp = np.concatenate(y_pred[name])
#        rp = classification_report(y_ture_tmp,y_pred_tmp)
        result[name] = f1_score(y_ture_tmp,y_pred_tmp,average = 'binary')
        print(name,result[name])
    problemVec = session.run(model4train.embeddings['ProblemID'])
    np.save('learnedVecs',problemVec)
    return result

if __name__=='__main__':
    results = {}
    for i in range(1):
        print(i)
        for i in range(10):
            x = main('Fold'+str(i),500)
            for name,d in x.items():
                if name not in results:
                    results[name]=[]
                results[name].append(d)
        for name,data in results.items():
            print('    ',name , sum(data) / len(data) )