## dataset
import sys

import pandas as pd
from keras import Sequential, regularizers

from keras.layers import Input, Flatten, Dense, concatenate, Dropout, Conv2D, AveragePooling2D, MaxPool2D, Lambda
from keras.optimizers import Adam


## required for semi-hard triplet loss:
from scipy.spatial.distance import pdist, cdist
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from keras import backend as K

## for visualizing
import matplotlib.pyplot as plt, numpy as np
from sklearn.decomposition import PCA

def create_model(shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(input_shape=(313, 128, 1), filters=64, kernel_size=(4, 4), strides=(2, 2), activation='selu',
                     padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='selu', padding='same',
                     kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='selu', padding='same',
                     kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='selu', padding='same',
                     kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(1, 2), strides=(1, 2), padding='same'))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128)))
    model.add(tf.keras.layers.Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    return model

def normalize_data_instance(data):
    for i in range(data.shape[0]):
        scaler=StandardScaler()
        temp= data[i].reshape((-1,1))
        temp=scaler.fit_transform(temp)
        temp=np.reshape(temp, (data.shape[1:]))
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
    return result_data, labels

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

@tf.function
def triplet_loss(y_true, y_pred):
    alpha=0.5
    type='euclidean'
    total_lenght = y_pred.shape.as_list()[-1]
    anchor, positive, negative = y_pred[:,:int(total_lenght/3)], y_pred[:,int(total_lenght/3):int(total_lenght/3*2)], y_pred[:,int(total_lenght/3*2):]
    #tf.print("anchor:\n",tf.shape(anchor), output_stream=sys.stdout)
    #tf.print("positive:\n", positive, output_stream=sys.stdout)
    #tf.print("negative:\n", negative, output_stream=sys.stdout)
    if type=='cosine':
        norm_anchor=tf.nn.l2_normalize(anchor, axis=-1)
        norm_positive = tf.nn.l2_normalize(positive, axis=-1)
        norm_negative = tf.nn.l2_normalize(negative, axis=-1)
        pos_dist=tf.reduce_sum(tf.multiply(norm_anchor,norm_positive), axis=-1)
        neg_dist=tf.reduce_sum(tf.multiply(norm_anchor,norm_negative), axis=-1)
        basic_loss = (1. - pos_dist) - (1. - neg_dist) + alpha
    elif type=='euclidean':
        pos_dist = tf.sqrt(tf.reduce_sum(tf.square(anchor - positive), axis=-1))
        neg_dist = tf.sqrt(tf.reduce_sum(tf.square(anchor - negative), axis=-1))
        basic_loss = pos_dist - neg_dist + alpha
    #tf.print("pos_dist:\n",pos_dist, output_stream=sys.stdout)
    #tf.print("neg_dist:\n", neg_dist, output_stream=sys.stdout)
    #tf.print("basic_loss:\n", basic_loss, output_stream=sys.stdout)
    loss = tf.reduce_sum(tf.maximum(basic_loss,0.0))
    #tf.print("loss:\n", loss, output_stream=sys.stdout)
    return loss

def create_batch_hard(data, labels, size_rand_sample_for_batch, size_hard_batch, size_rand_batch,
                      network, margin=0.5):
    classes=np.unique(labels)
    num_classes=classes.shape[0]
    anchor_class=np.random.randint(0,num_classes)
    negative_class=np.delete(classes, np.where(classes==anchor_class))

    anchor_data=data[labels.reshape((-1,))==anchor_class]
    negative_data=data[np.isin(labels.reshape((-1,)),negative_class)]
    indexes_anchor=np.random.choice(anchor_data.shape[0], size_rand_sample_for_batch)
    indexes_negative=np.random.choice(negative_data.shape[0], size_rand_sample_for_batch)
    anchor_data=anchor_data[indexes_anchor]
    negative_data=negative_data[indexes_negative]
    encoded_anchor=network.predict(anchor_data)
    encoded_negative=network.predict(negative_data)

    result_anchor = []
    result_pos = []
    result_neg = []

    # creating a hard batch
    pos_dists=cdist(encoded_anchor,encoded_anchor, metric='euclidean')
    pos_neg_dists=cdist(encoded_anchor, encoded_negative, metric='euclidean')

    sortet_pos_neg_dists_indexes=np.argsort(pos_neg_dists, axis=None)

    for i in range(size_hard_batch):
        idx_nearest_neg = np.unravel_index(sortet_pos_neg_dists_indexes[-i-1], pos_neg_dists.shape)
        idx_farest_pos=np.argmax(pos_dists[idx_nearest_neg[0]])
        #print('distance between anchor and negative:',pos_neg_dists[idx_nearest_neg[0],idx_nearest_neg[1]], '   distance between anchor and positive:', pos_dists[idx_nearest_neg[0],idx_farest_pos])
        #print('anchor index:',idx_nearest_neg[0], '  negative index:', idx_nearest_neg[1], '    positive index:', idx_farest_pos )
        result_anchor.append(anchor_data[idx_nearest_neg[0]])
        result_neg.append(negative_data[idx_nearest_neg[1]])
        result_pos.append(anchor_data[idx_farest_pos])

    # add some number of random batch
    i=0
    while i<size_rand_batch:
        rand_idx_anchor=np.random.randint(0,anchor_data.shape[0])
        while anchor_data[rand_idx_anchor] in np.array(result_anchor):
            rand_idx_anchor = np.random.randint(0, anchor_data.shape[0])
        result_anchor.append(anchor_data[rand_idx_anchor])

        rand_idx_pos = np.random.randint(0, anchor_data.shape[0])
        while anchor_data[rand_idx_pos] in np.array(result_pos) or rand_idx_pos==rand_idx_anchor:
            rand_idx_pos = np.random.randint(0, anchor_data.shape[0])
        result_pos.append(anchor_data[rand_idx_pos])

        rand_idx_neg = np.random.randint(0, negative_data.shape[0])
        while negative_data[rand_idx_neg] in np.array(result_pos):
            rand_idx_neg = np.random.randint(0, negative_data.shape[0])
        result_neg.append(negative_data[rand_idx_neg])
        i+=1


    return np.array(result_anchor), np.array(result_pos), np.array(result_neg)

def classify_from_embeddings_SVM(embeddings_train, train_labels, embeddings_test):
    clf = svm.SVC()
    clf.fit(embeddings_train, train_labels)
    return clf.predict(embeddings_test)

