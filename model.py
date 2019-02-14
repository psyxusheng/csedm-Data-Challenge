# -*- coding: utf-8 -*-
from numpy import ndarray
import tensorflow as tf

def get_shape(tensor):
    shape = []
    static_shape = tensor.get_shape().as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    for p,s in enumerate(static_shape):
        if isinstance(s , type(None)):
            shape.append(dynamic_shape[p])
        else:
            shape.append(s)
    return shape

class DKT():
    def __init__(self,
                 features ,
                 targets  ,
                 num_items  = 39,
                 rnn_units  = [32,8],
                 training   = True,
                 keep_prob  = 0.5,
                 batch_size = None,
                 time_steps = None,
                 lr_decay   = [1e-3,0.9,100]):
#        print('>>>>building model')
        def embedding_lookup(table , ids , scope ,skip_zero = False):
            with tf.variable_scope(scope+'lookup' ):
                if skip_zero:
                    table_ = tf.concat([tf.zeros_like(table[0:1,:]),
                                        table[1:,:]],axis=0)
                else:
                    table_ = table
                return tf.nn.embedding_lookup(table_,ids)
        
        with tf.variable_scope('embeddingTables',reuse = tf.AUTO_REUSE):
            embedding_inputs = []
            embedding_states = []
            name_inputs,name_states     = [],[]
            for name,type_,shape_or_vec,trainable in features:
                if isinstance(shape_or_vec , ndarray):
                    embTable = tf.get_variable(name = name+'embeddingTable',
                                               shape = shape_or_vec.shape,
                                               initializer = tf.constant_initializer(value= shape_or_vec,
                                                                                     dtype = tf.float32),
                                               trainable = trainable)
                elif isinstance(shape_or_vec , (list,tuple)):
                    embTable = tf.get_variable(name  = name + 'embeddingTable',
                                               shape = shape_or_vec,
                                               trainable=trainable,
                                               initializer= tf.random_uniform_initializer(-1e-2,1e-2))
                else:
                    raise Exception('wrong featrues information')
                if type_ == 'inp':
                    embedding_inputs.append(embTable)
                    name_inputs.append(name)
                elif type_ == 'ind':
                    name_states.append(name)
                    embedding_states.append(embTable)
        
        with tf.variable_scope('input_placeholders'):
            placeholders_inputs = []# for each time-step's input
            placeholders_states = []# for initial state's input
            for name,type_,shape_or_vec,trainable in features:
                if type_ == 'inp':
                    placeholder = tf.placeholder(name = name +'inputs',
                                                 shape = [batch_size,time_steps],
                                                 dtype = tf.int32)
                    placeholders_inputs.append(placeholder)
                elif type_ == 'ind':
                    placeholder = tf.placeholder(name = name + 'inputs',
                                                 shape = [batch_size],
                                                 dtype = tf.int32)
                    placeholders_states.append(placeholder)
                else:
                    raise Exception('wrong feature type')
        
            lengths = tf.placeholder(name = 'lengths',
                                    shape = [batch_size],
                                    dtype = tf.int32)
            masks   = tf.placeholder(name = 'masks',
                                     shape = [batch_size,time_steps,num_items],
                                     dtype = tf.float32)
            
        with tf.variable_scope('target_placeholders'):
            placeholder_targets = []
            predicts_units = []
            predicts_names = []
            loss_weights   = [] # for weighted sum of different costs
            loss_scalars   = [] # for balancing categories with in one target
            for name,num_cls,lw,ls in targets:
                ph = tf.placeholder(name = name +'_target',
                                    shape  = [batch_size,time_steps,num_items],
                                    dtype=tf.int32)
                placeholder_targets.append(ph)
                predicts_names.append(name)
                predicts_units.append(num_cls)
                loss_weights.append(lw)
                loss_scalars.append(ls)
                
        with tf.variable_scope('LSTM'):
            cells = []
            for i,units in enumerate(rnn_units):
                cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=units,
                                                    name = 'Layer%dCell'%(i+1),
                                                    activation=tf.nn.tanh,
                                                    reuse = tf.AUTO_REUSE,
                                                    state_is_tuple=True)
                if training:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=keep_prob)
                cells.append(cell)
            LSTMCell = tf.nn.rnn_cell.MultiRNNCell(cells,state_is_tuple=True)
        
        input_embededs = []
        for i,ph in enumerate(placeholders_inputs):
            input_embededs.append(embedding_lookup(embedding_inputs[i],ph,name_inputs[i],False))
        
        state_embededs = []
        for i,ph in enumerate(placeholders_states):
            state_embededs.append(embedding_lookup(embedding_states[i],ph,name_states[i],False))
        
        input_vec = tf.concat(input_embededs,axis=-1,name = 'concat4LstmInputs') 
        input_vec = tf.layers.dropout(input_vec,rate = keep_prob,
                                           training = training)
        input_features = tf.layers.conv1d(inputs  = input_vec,
                                          filters = rnn_units[0]//2,
                                          activation = tf.nn.relu,
                                          kernel_size =1,
                                          strides     = 1,
                                          name = 'inputFeatures',
                                          reuse= tf.AUTO_REUSE)
        input_features = tf.layers.dropout(input_features,rate = keep_prob , training=training)
        ##### building initial states for lstm
        if state_embededs != []:
            state_vec = tf.concat(state_embededs ,axis=-1 , name = 'concat4StateInput' )
            state_inputs = tf.layers.dense(state_vec,sum(rnn_units)*2,
                                           activation=tf.nn.relu,
                                           reuse= tf.AUTO_REUSE,
                                           name = 'feature4states')
            intial_state = []
            state_each_layer = tf.split(state_inputs , axis=-1,
                                        num_or_size_splits= [i*2 for i in rnn_units])
            for i,st in  enumerate(state_each_layer):
                c,h = tf.split(st,axis=-1,num_or_size_splits=2)
                intial_state.append(tf.nn.rnn_cell.LSTMStateTuple(c=c,h=h))
            initial_state = tuple(intial_state)
        else:
            bs,*_ = get_shape(placeholders_inputs[0])
            initial_state = LSTMCell.zero_state(bs,dtype=tf.float32)
        
        batch_size,time_steps = get_shape(placeholders_inputs[0])
        rnn_output,_ = tf.nn.dynamic_rnn(cell = LSTMCell,
                                         inputs = input_features,
                                         sequence_length=lengths,
                                         initial_state= initial_state,
                                         scope = 'LstmEncoding')
        
        rnn_output = tf.layers.dropout(rnn_output , 
                                            rate = keep_prob ,
                                            training = training)
        
        num_each_targets = [num_items*i for i in predicts_units]
        
        projects = tf.layers.dense(rnn_output , 
                                   units = sum(num_each_targets),
                                   reuse = tf.AUTO_REUSE,
                                   activation=None)
        logitsCollection = tf.split(projects , axis=-1, num_or_size_splits= num_each_targets)
        
        costs       = []
        predicts    = []
        probablities = []
        
        for i,logits_ in enumerate(logitsCollection):
            logits = tf.reshape(logits_,shape = [batch_size,time_steps,num_items,
                                              predicts_units[i]])
            preds = tf.argmax(logits,axis=-1,output_type=tf.int32)
            probs = tf.nn.softmax(logits , axis = -1)
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
                                                                  labels = placeholder_targets[i])
            scalar = tf.gather(loss_scalars[i] , placeholder_targets[i])
            masked_cost = tf.reduce_sum(tf.multiply(cost * scalar , masks)) / tf.reduce_sum(masks)
            costs.append(masked_cost)
            predicts.append(preds)
            probablities.append(probs)
        
        total_cost = tf.reduce_sum(tf.multiply(costs , loss_weights))

        with tf.variable_scope('training',reuse = tf.AUTO_REUSE):
            init_lr,decay_rate,decay_steps = lr_decay
            global_step = tf.get_variable(initializer=tf.constant_initializer(0.0,dtype=tf.float32),
                                          shape = [],
                                          dtype=tf.float32,
                                          trainable = False,
                                          name = 'global_step')
            learning_rate = tf.train.exponential_decay(learning_rate = init_lr,
                                                       global_step = global_step,
                                                       decay_steps = decay_steps,
                                                       decay_rate=decay_rate,
                                                       staircase=True,
                                                       name='learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,)
            grads, variables = zip(*optimizer.compute_gradients(total_cost))
            
            grads, global_norm = tf.clip_by_global_norm(grads,10)
            trainOp = optimizer.apply_gradients(zip(grads, variables),
                                                global_step=global_step)
        
        self.predicts = dict(list(zip(predicts_names,predicts)))
        self.probablities = dict(list(zip(predicts_names,probablities)))
        
        self.costs    = dict(list(zip(predicts_names,costs)))           
        self.embeddings = dict(list(zip(name_inputs,embedding_inputs))+\
                               list(zip(name_states,embedding_states)))
        self.input_holders =  dict(list(zip(name_inputs,placeholders_inputs))+ \
                                list(zip(name_states,placeholders_states))+\
                                [('lengths',lengths),('masks',masks)])
        self.target_holders = dict(list(zip(predicts_names,placeholder_targets)))
        self.trainop = trainOp
        self.model_detail = [learning_rate , global_step]
        
    def zip_data(self,data,to):
        feed_in = {}
        for name,holder in to.items():
            if name not in data:
                raise Exception('Misssing Data named %s'%name)
            feed_in[holder] = data[name]
        return feed_in
            
            
            
        
                    
                
if __name__ == '__main__':
    tf.reset_default_graph()
    model1 = DKT([['items','inp',[39,2],True],
                 ['firstCrt','inp',[3,2],True],
                 ['everCrt','inp',[3,2],True],
                 ['subj','ind',[88,2],True],
                 ['subj2','ind',[12,2],True]],
                num_items = 39,
                targets = [['fcrct',3, .5 , [.0 , 1. , 3.]],
                           ['ecrct',3, .5 , [.0 , 1. , 3.]]],
                batch_size=72,
                time_steps=36)
    model2 = DKT([['items','inp',[39,2],True],
                 ['firstCrt','inp',[3,2],True],
                 ['everCrt','inp',[3,2],True],
                 ['subj','ind',[88,2],True],
                 ['subj2','ind',[12,2],True]],
                num_items = 39,
                targets = [['fcrct',3, .5 , [.0 , 1. , 3.]],
                           ['ecrct',3, .5 , [.0 , 1. , 3.]]],
                batch_size=72,
                time_steps=36)                    
          