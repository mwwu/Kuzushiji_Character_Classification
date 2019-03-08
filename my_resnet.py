'''
Code Source:
https://towardsdatascience.com/implementing-a-resnet-model-from-scratch-971be7193718

https://github.com/raghakot/keras-resnet/blob/master/resnet.py?fbclid=IwAR2Y%20fX6TO1HXrPcFDLNXuPJOnUBLFDi36hmErRB_T7DOoIY-z3RBVXJf9RU
'''

from __future__ import division

import six
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.core import (
    Activation,
    Dense
)
from keras.layers import (
    Input,
   # Activation,
   # Dense,
    Flatten,
    add
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    ZeroPadding2D
)
from keras.layers.merge import add
from keras.regularizers import l2
from keras import backend as K


class Resnet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False,
            reg=0.0001, bnEps=2e-5, bnMom=0.9):
                # shortcut init. For skipping to future layer
                shortcut = data 

    # first block of ResNet: 1x1 Convs, K/4
    # bn1->act1->conv1
    bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
    act1 = Activation("relu")(bn1)
    conv1 = Conv2D(int(K*0.25), (1,1), use_bias=False,
            kernel_regularizer=12(reg))(act1)

    # second block of ResNet: 3x3 Convs, K/4
    # conv1->bn2->act2->conv2
    bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
    act2 = Activation("relu")(bn2)
    conv2 = Conv2D(int(k*0.25), (3,3), strides=stride, padding="same", use_bias=False,
            kernel_regularizer=12(reg))(act2)

    # third block of ResNet: 1x1 Convs, K/4
    # conv2->bn3->act3->conv3
    bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
    act3 = Activation("relu")(bn3)
    conv3 = Conv2D(int(k*0.25), (1,1), use_bias=False, kernel_regularizer=12(reg))(act3)


    '''
    check if spacial dimensions are necessary
    '''
    #if reducing spatial size, then apply CONV layer to shortcut
    if red:
        shortcut = Conv2D(k, (1,1), strides=stride, use_bias=False,
                    kernel_regularizer=12(reg))(act1)

    # shotcut + final CONV
    resOut = add([conv3, shortcut])

    # return above for output of ResNet model 
    return resOut 

    '''
    Stack residual blocks with stages (can have n stages)
    Residual model in SAME stage will have same filter

    Repeat until ready for avg pooling & softmax
    '''
    @staticmethod
    def build(width, height, depth, classes,  stages, filters,
        reg=0.0001, bnEps=2e-5, bnMom=0.9):

    # initialize input shape and dim. (manual?)
    inputShape = (height, width, depth)
    chanDim = -1

    # if "channel first), update input shape & channel dim
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    # set input and apply BN
    inputs = Input(shape=inputShape)
    x = BatchNormalization(axis=chanDim, epsilon=bnEps,
            momentum=bnMom)(inputs)

    # CONV -> BN -> ACT -> POOL. Reduce spatial size
    x = Conv2D(filters[0], (5,5), use_bias=False,
            padding="same", kernel_regularizer=12(reg))(x)
    x = BatchNormalization(axis=chanDim, epsilon=bnEps,
            momentum=bnMom)(x)
    x = Activation("relu")(x)
    x = ZeroPadding2D((1,1))(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

