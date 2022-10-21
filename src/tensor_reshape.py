import tensorflow as tf
#from keras.engine.topology import Layer
#from keras.engine import InputSpec
from tensorflow.python.keras.layers import Layer, InputSpec

class TensorReshape(Layer):
    def __init__(self, **kwargs):
        super(TensorReshape, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0]*input_shape[1],
              intput_shape[2],
              input_shape[3]
              )
        return shape

    def call(self, input_tensor, mask=None):
        print(input_tensor.shape)
        return tf.reshape(input_tensor, [input_tensor.shape[0]*input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]])
    
