'''
Small Resnet to train 34 layers

code source:
https://github.com/raghakot/keras-resnet/blob/master/resnet.py?fbclid=IwAR2YfX6TO1HXrPcFDLNXuPJOnUBLFDi36hmErRB_T7DOoIY-z3RBVXJf9RU    
and
https://towardsdatascience.com/implementing-a-resnet-model-from-scratch-971be7193718
'''


# From my_resnet.py no code modified here

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


class SmallResNet:
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
                    kernel_regularizer=l2(reg))(act1)

            # second block of ResNet: 3x3 Convs, K/4
            # conv1->bn2->act2->conv2
            bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
            act2 = Activation("relu")(bn2)
            conv2 = Conv2D(int(K*0.25), (3,3), strides=stride, padding="same", use_bias=False,
                    kernel_regularizer=l2(reg))(act2)

            # third block of ResNet: 1x1 Convs, K/4
            # conv2->bn3->act3->conv3
            bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
            act3 = Activation("relu")(bn3)
            conv3 = Conv2D(K, (3,3), use_bias=False, kernel_regularizer=l2(reg))(act3)
            # 2nd argument of Conv2D: (1,1) -> (3,3) for standard instead of bottleneck


            '''
            check if spacial dimensions are necessary
            '''
            #if reducing spatial size, then apply CONV layer to shortcut
            # 2nd argument of Conv2D: (1,1) -> (3,3) for standard instead of bottleneck
            if red:
                shortcut = Conv2D(K, (3,3), strides=stride, use_bias=False,
                            kernel_regularizer=l2(reg))(act1)

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

            # if "channel first", update input shape & channel dim
            if K.image_data_format() == "channels_first":
                inputShape = (depth, height, width)
                chanDim = 1

            # set input and apply BN
            inputs = Input(shape=inputShape)
            x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                    momentum=bnMom)(inputs)

            # CONV -> BN -> ACT -> POOL. Reduce spatial size
            x = Conv2D(filters[0], (5,5), use_bias=False,
                    padding="same", kernel_regularizer=l2(reg))(x)
            x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                    momentum=bnMom)(x)
            x = Activation("relu")(x)
            #x = ZeroPadding2D((1,1))(x)
            x = MaxPooling2D((3,3), strides=(2,2))(x)
            
            #reduce input volume 
            for i in range(0, len(stages)):
                # init stride, apply residual
                stride = (1,1) if i == 0 else (2,2)
                x = ResNet.residual_module(x, filters[i+1], stride,
                        chanDim, red=True, bnEps=bnEps, bnMom=bnMom)
                
                # loop through layers in stage
                for j in range(0, stages[i] - 1):
                    # apply resnet module
                    x = ResNet.residual_module(x, filters[i+1],
                            (1,1), chanDim, bnEps=bnEps, bnMom=bnMom)

            x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                    momentum=bnMom)(x)
            x = Activation("relu")(x)
            x = AveragePooling2D((2,2))(x)

            # softmax
            x = Flatten()(x)
            x = Dense(classes, kernel_regularizer=l2(reg))(x)
            x = Activation("softmax")(x)

            # create the model
            model = Model(inputs, x, name="resnet")

            # return constructed network architecture
            return model

