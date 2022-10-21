import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

class Time2Vector(Layer):
  def __init__(self, seq_len, **kwargs):
    super(Time2Vector, self).__init__()
    self.seq_len = seq_len

  def build(self, input_shape):
    '''Initialize weights and biases with shape (batch, seq_len)'''
    self.weights_linear = self.add_weight(name='weight_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.bias_linear = self.add_weight(name='bias_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.weights_periodic1 = self.add_weight(name='weight_periodic1',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic1 = self.add_weight(name='bias_periodic1',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.weights_periodic2 = self.add_weight(name='weight_periodic2',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic2 = self.add_weight(name='bias_periodic2',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
  
    #self.weights_periodic3 = self.add_weight(name='weight_periodic3',
    #                            shape=(int(self.seq_len),),
    #                            initializer='uniform',
    #                            trainable=True)

    #self.bias_periodic3 = self.add_weight(name='bias_periodic3',
    #                            shape=(int(self.seq_len),),
    #                            initializer='uniform',
    #                            trainable=True)
  
  def call(self, x):
    '''Calculate linear and periodic time features'''
    x = tf.math.reduce_mean(x[:,:,: :], axis=-1) 
    #x = tf.math.reduce_mean(x[:,:,:4], axis=-1) 
    time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
    time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)
    
    time_periodic1 = tf.math.sin(tf.multiply(x, self.weights_periodic1) + self.bias_periodic1)
    time_periodic1 = tf.expand_dims(time_periodic1, axis=-1) # Add dimension (batch, seq_len, 1)
    time_periodic2 = tf.math.sin(tf.multiply(x, self.weights_periodic2) + self.bias_periodic2)
    time_periodic2 = tf.expand_dims(time_periodic2, axis=-1) # Add dimension (batch, seq_len, 1)
    #time_periodic3 = tf.math.sin(tf.multiply(x, self.weights_periodic3) + self.bias_periodic3)
    #time_periodic3 = tf.expand_dims(time_periodic3, axis=-1) # Add dimension (batch, seq_len, 1)
    #return tf.concat([time_linear, time_periodic1, time_periodic2, time_periodic3], axis=-1) # shape = (batch, seq_len, 2)
    return tf.concat([time_linear, time_periodic1, time_periodic2], axis=-1) # shape = (batch, seq_len, 2)
   
  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'seq_len': self.seq_len})
    return config
