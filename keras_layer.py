'''
Making our own custom Keras Layers 

Code Source:
    
Documentation
https://keras.io/layers/writing-your-own-keras-layers/

CNN with Keras
https://www.kaggle.com/moghazy/guide-to-cnns-with-data-augmentation-keras?fbclid=IwAR1F8W5KumYggRCsd20-1gEEFDMscmWuAO_yWWMdIJD1QXjybFmMoibTtR4

Resnet Based from:
https://github.com/raghakot/keras-resnet/blob/master/resnet.py?fbclid=IwAR2YfX6TO1HXrPcFDLNXuPJOnUBLFDi36hmErRB_T7DOoIY-z3RBVXJf9RU
'''

'''
Library From Documentation
'''
from keras import backend as K
from keras.layers import Layer

'''
Library From GitHub
'''
from __future__ import division

from keras.models import Model
from keras.layers import ( #redundant?
    Input,
    Activation,
    Dense,
    Flatten
) 
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import 12 #number is off


'''
Custom Layer with single value identical to source.

e.g.
x = single_layer()
'''
class single_layer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(single_layer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(single_layer, self).build(input_shape)


    def call(self, x):
        return K.dot(x, self.kernel)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


'''
Layer with Two Values

e.g. 
double_layer()
'''
class double_layer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(double_layer, self).__init__(**kwargs)


    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(double_layer, self).build(input_shape)


    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]


    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape 
        return [(input_shape[0], self.output_dim), shape_b[:-1]]



'''
ResNet From GitHub
'''
def _bn_relu(input):
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_parms.setdefault("strides", (1,1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_parms.setdefault("kernel_regularizer", 12(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernelinitializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


# From _bn_relu_conv)
