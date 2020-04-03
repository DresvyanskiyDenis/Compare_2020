import os
from functions import create_model
import keras
from sklearn.metrics import recall_score
import tensorflow as tf
from keras import Sequential, metrics
from keras.engine.saving import load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, TimeDistributed, LSTM
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

def normalize_data(data, scaler=None):
    temp = np.reshape(data, (-1, 1))
    if scaler==None:
        scaler=StandardScaler()
        scaler=scaler.fit(temp)
    temp=scaler.transform(temp)
    data=np.reshape(temp, data.shape)
    return data, scaler

def load_data(path_to_labels, path_to_data, prefix, kind_labels):
    labels = pd.read_csv(path_to_labels + 'labels.csv', sep=',')
    labels.drop(labels.columns.difference(['filename_audio', kind_labels]), 1, inplace=True)
    labels=labels.loc[labels['filename_audio'].str.contains(prefix)]
    example = pd.read_csv(path_to_data + labels.iloc[0, 0].split('.')[0] + '.csv', sep=',', header=None)
    result_data = np.zeros(shape=(labels.shape[0], example.shape[0], example.shape[1], 1))
    for index in range(labels.shape[0]):
        filename_temp = labels.iloc[index, 0].split('.')[0] + '.csv'
        data = pd.read_csv(path_to_data + filename_temp, sep=',', header=None).values
        data = data[..., np.newaxis]
        result_data[index] = data

    labels.drop(['filename_audio'], 1, inplace=True)
    labels = to_categorical(labels)
    return result_data, labels



path_labels='C:\\Users\\Denis\\Desktop\\ComParE2020_Elderly\\ComParE2020_Elderly\\lab\\'
path_data='C:\\Users\\Denis\\Desktop\\ComParE2020_Elderly\\ComParE2020_Elderly\\Mel_spectrogram_2\\Mel_spectrogram_2\\'
prefix='devel'
kind_labels='A_cat_no'
validation_data, labels=load_data(path_to_labels=path_labels, path_to_data=path_data, prefix=prefix, kind_labels=kind_labels)

path_to_model='model.h5'
path_to_weights='model.h5'


model=create_model()
model.load_weights('weights.h5')
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[keras.metrics.Recall()])
# prediction and then evaluation UAC
#training_data, _= load_data(path_to_labels=path_labels, path_to_data=path_data, prefix='train', kind_labels=kind_labels)
#scaler=StandardScaler()
#scaler=scaler.fit(np.reshape(training_data,(-1,1)))
validation_data,_=normalize_data(validation_data)

predictions=model.predict(validation_data)
predictions=np.argmax(predictions, axis=1)
UAC=recall_score(y_true=np.argmax(labels, axis=1), y_pred=predictions, average='macro')
print(UAC)
