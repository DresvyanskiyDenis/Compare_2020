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
import numpy as np
import pandas as pd

'''import numpy as np

from Breathing_1.end_to_end.utils import smoothing

arr=np.array([1,2,3,4,5,6,7,8])
res=smoothing(arr, 5)'''
'''import tensorflow as tf
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
print(model.summary())'''

predictions=np.array([
    [
        [1,2,3,4,5,6],
        [5,6,7,8,9,0],
        [2,3,6,5,4,3]
    ],
    [
        [1, 2, 3, 4, 5, 6],
        [5, 6, 7, 8, 9, 0],
        [2, 3, 6, 5, 4, 3]
    ]
])

timesteps=np.array([
    [
        [0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
        [0.06, 0.08, 0.10, 0.12, 0.14, 0.16],
        [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    ],
    [
        [0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
        [0.06, 0.08, 0.10, 0.12, 0.14, 0.16],
        [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    ]
])

class_dict={0:'train_00.wav',1:'train_01.wav'}
true_values=pd.DataFrame(columns=['filename', 'timeFrame', 'upper_belt'], data=np.array([
    ['train_00.wav','train_00.wav','train_00.wav','train_00.wav','train_00.wav','train_00.wav','train_00.wav','train_00.wav','train_00.wav','train_00.wav','train_01.wav','train_01.wav','train_01.wav','train_01.wav','train_01.wav','train_01.wav','train_01.wav','train_01.wav','train_01.wav','train_01.wav'],
    [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
]).T)



def new_concatenate_prediction(true_values, predicted_values, timesteps_labels, class_dict):
    predicted_values=predicted_values.reshape(timesteps_labels.shape)
    result_predicted_values=pd.DataFrame(columns=true_values.columns, dtype='float32')
    result_predicted_values['filename']=result_predicted_values['filename'].astype('str')
    for instance_idx in range(predicted_values.shape[0]):
        predicted_values_tmp=predicted_values[instance_idx].reshape((-1,1))
        timesteps_labels_tmp=timesteps_labels[instance_idx].reshape((-1,1))
        tmp=pd.DataFrame(columns=['timeFrame', 'upper_belt'], data=np.concatenate((timesteps_labels_tmp, predicted_values_tmp), axis=1))
        tmp=tmp.groupby(by=['timeFrame']).mean().reset_index()
        tmp['filename']=class_dict[instance_idx]
        result_predicted_values=result_predicted_values.append(tmp)
    result_predicted_values['timeFrame']=result_predicted_values['timeFrame'].astype('float32')
    result_predicted_values['upper_belt'] = result_predicted_values['upper_belt'].astype('float32')
    return result_predicted_values[true_values.columns]

def concatenate_prediction(true_values, predicted_values, timesteps_labels, class_dict):
    predicted_values=predicted_values.reshape(timesteps_labels.shape)
    tmp=np.zeros(shape=(true_values.shape[0],3))
    result_predicted_values=pd.DataFrame(data=tmp, columns=true_values.columns, dtype='float32')
    result_predicted_values['filename']=result_predicted_values['filename'].astype('str')
    index_temp=0
    for instance_idx in range(predicted_values.shape[0]):
        timesteps=np.unique(timesteps_labels[instance_idx])
        for timestep in timesteps:
            # assignment for filename and timestep
            result_predicted_values.iloc[index_temp,0]=class_dict[instance_idx]
            result_predicted_values.iloc[index_temp,1]=timestep
            # calculate mean of windows
            result_predicted_values.iloc[index_temp,2]=np.mean(predicted_values[instance_idx,timesteps_labels[instance_idx]==timestep])
            index_temp+=1
        #print('concatenation...instance:', instance_idx, '  done')

    return result_predicted_values

print(concatenate_prediction(true_values, predictions, timesteps, class_dict))
print(new_concatenate_prediction(true_values, predictions, timesteps, class_dict))

a=concatenate_prediction(true_values, predictions, timesteps, class_dict)
b=new_concatenate_prediction(true_values, predictions, timesteps, class_dict)
b['timeFrame']=b['timeFrame'].astype('float32')
b['upper_belt']=b['upper_belt'].astype('float32')

print(concatenate_prediction(true_values, predictions, timesteps, class_dict)==new_concatenate_prediction(true_values, predictions, timesteps, class_dict))












