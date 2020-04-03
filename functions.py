import os

import keras
from keras import backend as K, regularizers, Model
from keras import Sequential, metrics
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, TimeDistributed, LSTM, AveragePooling2D, Dropout, \
    Bidirectional, GRU, Conv1D, MaxPool1D
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler

def normalize_data(data, scaler=None):
    temp = np.reshape(data, (-1, 1))
    if scaler==None:
        scaler=StandardScaler()
        scaler=scaler.fit(temp)
    temp=scaler.transform(temp)
    data=np.reshape(temp, data.shape)
    return data, scaler

def normalize_data_instance(data):
    for i in range(data.shape[0]):
        scaler=StandardScaler()
        temp= data[i].reshape((-1,1))
        temp=scaler.fit_transform(temp)
        temp=np.reshape(temp, (data.shape[1:]))
        data[i]=temp
    return data

def minmax_normalization_instance(data):
    for i in range(data.shape[0]):
        max=np.max(data[i])
        min=np.min(data[i])
        temp=data[i]
        temp=2.*(temp-min)/(max-min)-1.
        data[i]=temp
    return data

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
    labels=labels.values
    labels=labels.astype('int32')
    #labels = to_categorical(labels)
    return result_data, labels

def prepare_data_for_ResNet(data):
    new_data=np.zeros(shape=data.shape[:-1]+(3,))
    for i in range(data.shape[0]):
        max=np.max(data[i])
        min=np.min(data[i])
        temp=data[i].copy()
        temp=2.*(temp-min)/(max-min)-1.
        temp_array=np.concatenate((temp,temp.copy()), axis=-1)
        temp_array =np.concatenate((temp_array,temp.copy()), axis=-1)
        new_data[i]=temp_array
        if not (np.equal(new_data[i,:,:,0], new_data[i,:,:,1]).all() and np.equal(new_data[i,:,:,1], new_data[i,:,:,2]).all()):
            print('WTF')
    return new_data

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(4, 4), strides=(2, 2), activation='selu',
                     padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='selu', padding='same',kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='selu', padding='same',kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='selu', padding='same',kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(Dropout(0.2))
    model.add(AveragePooling2D(pool_size=(1, 2), strides=(1, 2), padding='same'))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(GRU(64, return_sequences=True)))
    model.add(Bidirectional(GRU(64)))
    model.add(Dense(3, activation='softmax'))
    return model

def create_model_1dconv(input_shape):
    model= Sequential()
    model.add(Conv1D(input_shape=input_shape, filters=32, kernel_size=11, activation='relu', padding='same'))
    model.add(Conv1D(input_shape=input_shape, filters=32, kernel_size=11, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=16,strides=16))
    model.add(Conv1D(input_shape=input_shape, filters=64, kernel_size=9, activation='relu', padding='same'))
    model.add(Conv1D(input_shape=input_shape, filters=64, kernel_size=9, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=8,strides=8))
    model.add(Conv1D(input_shape=input_shape, filters=128, kernel_size=7, activation='relu', padding='same'))
    model.add(Conv1D(input_shape=input_shape, filters=128, kernel_size=7, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=8, strides=8))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(Bidirectional(GRU(128)))
    model.add(Dense(3, activation='softmax'))
    return model

def create_model_ResNet(input_shape):
    model = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet',
                                               input_shape=input_shape, pooling=None)
    for i in range(156):
        model.layers[i].trainable=False
    last_layer=model.get_layer('conv5_block3_2_relu').output
    flat=TimeDistributed(Flatten())(last_layer)
    #dense1=TimeDistributed(Dense(128, activation='selu'))(flat)
    rnn1=Bidirectional(GRU(128, return_sequences=True))(flat)
    rnn2=Bidirectional(GRU(128))(rnn1)
    clf=Dense(3, activation='softmax')(rnn2)
    new_model=Model(inputs=model.input, outputs=clf)
    return new_model


def vizualization(data, labels, network, path_to_save='', prefix=''):
    embeddings=[]
    colors = ['red', 'green', 'blue']
    nb_classes=np.unique(labels).shape[0]
    embeddings=network.predict(data)
    embeddings=TSNE(n_components=2).fit_transform(embeddings)
    for i in range(nb_classes):
        plt.scatter(embeddings[labels[:,0]==i,0],embeddings[labels[:,0]==i,1],c=colors[i])
    #plt.show()
    if path_to_save!='':
        plt.savefig(path_to_save+prefix+'_TSNE.png')
    plt.clf()

    embeddings=network.predict(data)
    embeddings=PCA(n_components=2).fit_transform(embeddings)
    for i in range(nb_classes):
        plt.scatter(embeddings[labels[:,0]==i,0],embeddings[labels[:,0]==i,1],c=colors[i])
    #plt.show()
    if path_to_save != '':
        plt.savefig(path_to_save + prefix + '_PCA.png')
    plt.clf()

def vizualization_without_embeddings(data, labels, path_to_save='', prefix=''):
    embeddings=[]
    colors = ['red', 'green', 'blue']
    nb_classes=np.unique(labels).shape[0]
    embeddings=np.reshape(data, newshape=(data.shape[0], -1))
    embeddings=TSNE(n_components=2).fit_transform(embeddings)
    for i in range(nb_classes):
        plt.scatter(embeddings[labels[:,0]==i,0],embeddings[labels[:,0]==i,1],c=colors[i])
    #plt.show()
    if path_to_save!='':
        plt.savefig(path_to_save+prefix+'_TSNE.png')
    plt.clf()

    embeddings=np.reshape(data, newshape=(data.shape[0], -1))
    embeddings=PCA(n_components=2).fit_transform(embeddings)
    for i in range(nb_classes):
        plt.scatter(embeddings[labels[:,0]==i,0],embeddings[labels[:,0]==i,1],c=colors[i])
    #plt.show()
    if path_to_save != '':
        plt.savefig(path_to_save + prefix + '_PCA.png')
    plt.clf()
