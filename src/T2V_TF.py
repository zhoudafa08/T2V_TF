import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import numpy as np
import pandas as pd
import sys,  datetime
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from Time2Vector import Time2Vector
from Transformer import *
#from keras import losses
from sklearn.metrics import precision_recall_fscore_support
from hyperopt import fmin, hp, Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from Rank_Loss import Rank_Loss
from evaluator import evaluate
from tensor_reshape import TensorReshape
from tensorflow.keras.layers import BatchNormalization
import datetime
from evaluator_trading_fee import evaluate_trading_fee
print('Tensorflow version: {}'.format(tf.__version__))

import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def data():
    X = np.load('../data/a50_feats_pca10_wnd5.npy') 
    Y = np.load('../data/a50_labels_pca10_wnd5.npy') 
    print(X.shape, Y.shape)
    train_num = int(0.7*X.shape[0])
    val_num = int(0.2*X.shape[0])
    x_train = X[:train_num, :, :, :]
    y_train = Y[:train_num, :, :, :]
    x_val = X[train_num:train_num+val_num, :, :, :]
    y_val = Y[train_num:train_num+val_num, :, :, :]
    x_test = X[train_num+val_num:, :, :, :]
    y_test = Y[train_num+val_num:, :, :, :]
    print('Training set shape', x_train.shape, y_train.shape)
    print('Validation set shape', x_val.shape, y_val.shape)
    print('Testing set shape' ,x_test.shape, y_test.shape)
    return x_train, y_train, x_val, y_val, x_test, y_test


def create_model(x_train, y_train, x_val, y_val, x_test, y_test, old_loss=10):
    def rank_loss(y_true, y_pred, alpha=1000):
        cpn_size = y_true.shape[1]
        base_close = y_true[:, :, 1, :]
        return_ratio = tf.divide(tf.subtract(y_pred, base_close), base_close)
        gt = y_true[:, :, 0]
        reg_loss = tf.reduce_mean(tf.square(gt-return_ratio))
        pred_row = tf.tile(return_ratio, multiples=[1, 1, cpn_size])
        pred_col = tf.tile(tf.transpose(return_ratio, perm=[0,2,1]), multiples=[1, cpn_size, 1])
        pre_pw_dif = tf.subtract(pred_row, pred_col)
        
        gt_row = tf.tile(gt, multiples=[1, 1, cpn_size])
        gt_col = tf.tile(tf.transpose(gt, perm=[0,2,1]), multiples=[1, cpn_size, 1])
        gt_pw_dif = tf.subtract(gt_col, gt_row)
        rank_loss = tf.reduce_mean(tf.nn.relu(tf.multiply(tf.cast(pre_pw_dif, tf.float32), tf.cast(gt_pw_dif, tf.float32))))
        return reg_loss + alpha * rank_loss
    
    '''Initialize time and transformer layers'''
    time_embedding = Time2Vector(5)
    attn_layer1 = TransformerEncoder({{choice([8,16,32,64])}},{{choice([8,16,32,64])}},{{choice([4,6,8])}},{{choice([8,16,32,64])}})
    attn_layer2 = TransformerEncoder({{choice([8,16,32,64])}},{{choice([8,16,32,64])}},{{choice([4,6,8])}},{{choice([8,16,32,64])}})
    attn_layer3 = TransformerEncoder({{choice([8,16,32,64])}},{{choice([8,16,32,64])}},{{choice([4,6,8])}},{{choice([8,16,32,64])}})
    attn_layer4 = TransformerEncoder({{choice([8,16,32,64])}},{{choice([8,16,32,64])}},{{choice([4,6,8])}},{{choice([8,16,32,64])}})
  
    '''Construct model'''
    #in_seq = Input(shape=(seq_len, 5))
    in_seq = Input(shape(45, 5, 28)) # the values of the parameters S, T and F that appear in Algorithm 3 of the reference "T2V_TF: ..." are 45, 5, and 28, respectively.
    x = time_embedding(in_seq)
    x = Concatenate(axis=-1)([in_seq, x])
    x = attn_layer1((x, x, x))
    x = attn_layer2((x, x, x))
    x = attn_layer3((x, x, x))
    x = tf.reduce_mean(x, -1)
    x = Dropout({{uniform(0,1)}})(x)
    x = Dense({{choice([32, 48, 64, 96])}}, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout({{uniform(0,1)}})(x)
    x = Dense({{choice([16, 32, 48, 64])}}, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout({{uniform(0,1)}})(x)
    x = Dense({{choice([8, 16, 24, 32])}}, activation='sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dropout({{uniform(0,1)}})(x)
    out = Dense(1)(x)
  
    model = Model(inputs=in_seq, outputs=out)
    model.compile(loss=Rank_Loss(1000), optimizer='adam')
    model.summary()
    history = model.fit(x_train, y_train,
                        batch_size=16,
                        epochs={{choice(range(4, 80, 10))}},
                        validation_data=(x_val, y_val))
    pred = model.predict(x_test)
    rank_loss = rank_loss(y_test, pred, 1000)
    filepath = 'Transformer+TimeEmbedding_model.hdf5'
    try:
        with open('metric.txt') as f:
            min_loss = float(f.read().strip().split('(')[1].split(',')[0])
    except FileNotFoundError:
        min_loss = rank_loss
    if rank_loss <= min_loss:
        model.save(filepath)
        with open('metric.txt', 'w') as f:
            f.write(str(rank_loss))
    sys.stdout.flush()
    return {'loss': rank_loss, 'model': model, 'status': STATUS_OK}

if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test=data()
    best_run, best_model=optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=5, trials=Trials())
    model = tf.keras.models.load_model('./Transformer+TimeEmbedding_model.hdf5',
                                       custom_objects={'Time2Vector': Time2Vector,
                                                       'SingleAttention': SingleAttention,
                                                       'MultiAttention': MultiAttention,
                                                       'Rank_Loss': Rank_Loss,
                                                       'TransformerEncoder': TransformerEncoder})
    
    #Calculate predication for training, validation and test data
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    starttime = datetime.datetime.now()
    test_pred = model.predict(x_test)
    endtime = datetime.datetime.now()
    print('Testing time:', (endtime-starttime))
    return_ratio = (test_pred[:, :, 0] - y_test[:,:,1,0]) / y_test[:, :, 1, 0]
    gt = y_test[:, :, 0, 0] #days * stock number
   
    return_ratio = return_ratio.transpose()
    gt = gt.transpose()
    np.savetxt('test_pred_results.csv', return_ratio,  delimiter=',', fmt='%.4f')
    np.savetxt('test_true_results.csv', gt, delimiter=',', fmt='%.4f')

    weights = pd.read_csv('../data/csi300_weight_2020.csv')[['code', 'weight']]  
    weights['code'] = weights['code'].str[:6]
    weights = weights.set_index('code')
    mask = np.zeros((gt.shape[0], 1))
    id = 0
    with open("../data/code_list", "r") as f:
        for line in f.readlines():
            line = line.strip('\n').split(',')
            code = ''.join(line)
            mask[id, 0] = weights.loc[code]
            id += 1

    performance = evaluate_trading_fee(return_ratio, gt, mask) #consider trading fee
    print(performance)
