import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import tensorflow as tf
#from keras.engine.topology import Layer
#from keras.engine import InputSpec
from tensorflow.python.keras.layers import Layer, InputSpec
#import keras

class Rank_Loss(tf.keras.losses.Loss):
    def __init__(self, alpha=1000, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        cpn_size = y_true.shape[1]
        base_close = y_true[:,:,1,:]
        return_ratio = tf.divide(tf.subtract(y_pred, base_close), base_close)
        gt = y_true[:,:,0, :]
        reg_loss = tf.losses.mean_squared_error(gt, return_ratio)
        
        pred_row = tf.tile(return_ratio, multiples=[1, 1, cpn_size])
        pred_col = tf.tile(tf.transpose(return_ratio, perm=[0,2,1]), multiples=[1, cpn_size, 1])
        pre_pw_dif = tf.subtract(pred_row, pred_col)
        
        gt_row = tf.tile(gt, multiples=[1, 1, cpn_size])
        gt_col = tf.tile(tf.transpose(gt, perm=[0,2,1]), multiples=[1, cpn_size, 1])
        gt_pw_dif = tf.subtract(gt_col, gt_row)
        
        rank_loss = tf.reduce_mean(tf.nn.relu(tf.multiply(pre_pw_dif, gt_pw_dif)))
        return reg_loss + self.alpha * rank_loss
    
