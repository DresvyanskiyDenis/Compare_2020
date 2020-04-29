'''import numpy as np
from scipy.stats import pearsonr
from keras import backend as K
import tensorflow as tf
x1=np.array([23,5634,123,234,364,123,543])
y1=np.array([123,5,6745,2134,7432,457,11])
print(pearsonr(x1,y1))

x2=np.array([123,523,123,543,756,123,346])
y2=np.array([132,89,567,234,7,879,1])
print(pearsonr(x2,y2))

x3=np.array([234,1,2,4,12,543,3])
y3=np.array([123,4,54,234,5,767,98])
print(pearsonr(x3,y3))
x=np.array([x1,x2,x3])
y=np.array([y1,y2,y3])

x_tf=tf.Variable(x, dtype=tf.float32)
y_tf=tf.Variable(y, dtype=tf.float32)

def pearson_corr(y_true, y_pred):
    x=y_true
    y=y_pred
    mx=K.mean(x, axis=1, keepdims=True)
    my=K.mean(y, axis=1, keepdims=True)
    xm,ym=x-mx,y-my
    r_num=K.sum(tf.multiply(xm, ym), axis=1)
    sum_square_x=K.sum(K.square(xm), axis=1)
    sum_square_y = K.sum(K.square(ym), axis=1)
    sqrt_x = tf.sqrt(sum_square_x)
    sqrt_y = tf.sqrt(sum_square_y)
    r_den=tf.multiply(sqrt_x, sqrt_y)
    result=tf.divide(r_num, r_den)
    #tf.print('result:', result)
    result=K.mean(result)
    #tf.print('mean result:', result)
    return 1-result

pearson_corr(x_tf, y_tf)'''

'''import numpy as np

from Breathing_1.end_to_end.utils import smoothing

arr=np.array([1,2,3,4,5,6,7,8])
res=smoothing(arr, 5)'''
import tensorflow as tf
input_shape=(256000,1)


def conv_block(input_tensor,filters, block_number):
    filter1, filter2, filter3=filters
    x = tf.keras.layers.Conv1D(filters=filter1, kernel_size=1, strides=1, activation=None, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=filter2, kernel_size=5, strides=1, activation=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=filter3, kernel_size=1, strides=1, activation=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(name='last_conv_bn_block_'+str(block_number))(x)
    shortcut = tf.keras.layers.Conv1D(filters=filter3, kernel_size=1, strides=1, activation=None, use_bias=False)(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization(name='shortcut_bn_block_'+str(block_number))(shortcut)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

input=tf.keras.layers.Input(shape=input_shape)
x=tf.keras.layers.Conv1D(filters=64, kernel_size=8, strides=1, activation=None, padding='same')(input)
x=tf.keras.layers.BatchNormalization()(x)
x=tf.keras.layers.Activation(activation='relu')(x)
output_block1=tf.keras.layers.MaxPool1D(pool_size=10)(x)

x=conv_block(output_block1, [64,64,256])
output_block2=tf.keras.layers.MaxPool1D(pool_size=8)(x)

x=conv_block(output_block2, [128,128,512])
output_block3=tf.keras.layers.MaxPool1D(pool_size=4)(x)

x=conv_block(output_block2, [256,256,1024])
output_block4=tf.keras.layers.AvgPool1D(pool_size=2)(x)

x=tf.keras.layers.LSTM(256, return_sequences=True)(output_block4)
x=tf.keras.layers.Dropout(0.2)(x)
x=tf.keras.layers.LSTM(256, return_sequences=True)(x)
x=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='tanh'))(x)
x=tf.keras.layers.Flatten()(x)
model=tf.keras.Model(inputs=[input], outputs=[x])
print(model.summary())





'''def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1), strides=strides,
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_1)(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size, padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_2)(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_3)(x)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj'
    bn_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj/bn'
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer='orthogonal',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay),
                      trainable=trainable,
                      name=conv_name_4)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_4)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x'''
