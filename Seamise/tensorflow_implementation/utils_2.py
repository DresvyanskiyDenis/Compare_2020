## dataset
import pandas as pd
from keras import Sequential, regularizers
from keras.datasets import mnist

## for Model definition/training
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate, Dropout, Conv2D, AveragePooling2D, MaxPool2D, Lambda
from keras.optimizers import Adam

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

## required for semi-hard triplet loss:
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf
from keras import backend as K

## for visualizing
import matplotlib.pyplot as plt, numpy as np
from sklearn.decomposition import PCA

def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.

    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(array_ops.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                    array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.

    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.

    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums


def triplet_loss_adapted_from_tf(y_true, y_pred):
    del y_true
    margin =0.5
    labels = y_pred[:, :1]

    labels = tf.cast(labels, dtype='int32')

    embeddings = y_pred[:, 1:]

    ### Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:

    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    # lshape=array_ops.shape(labels)
    # assert lshape.shape == 1
    # labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    # global batch_size
    batch_size = array_ops.size(labels)  # was 'array_ops.size(labels)'

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, array_ops.reshape(
                array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    semi_hard_triplet_loss_distance = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')

    ### Code from Tensorflow function semi-hard triplet loss ENDS here.
    return semi_hard_triplet_loss_distance

def create_model():
    model = Sequential()
    #model.add(Flatten(input_shape=(313, 128, 1)))
    #model.add(Dense(1000, activation='selu', kernel_regularizer=regularizers.l2(0.0001)))
    #model.add(Dense(64, activation='selu', kernel_regularizer=regularizers.l1_l2(0.01, 0.01)))
    model.add(Conv2D(input_shape=(313, 128, 1), filters=32, kernel_size=(4, 4), strides=(2, 2), activation='selu',
                     padding='same', kernel_regularizer=regularizers.l1_l2(0.0001,0.0001)))
    model.add(Dropout(0.3))
    #model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
    #model.add(Dropout(0.3))
    #model.add(Conv2D(filters=64, kernel_size=(6, 6), strides=(3, 3), activation='selu', padding='same', kernel_regularizer=regularizers.l1_l2(0.001,0.001)))
    #model.add(Dropout(0.3))
    model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), activation='selu', padding='same', kernel_regularizer=regularizers.l1_l2(0.0001,0.0001)))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(4, 4), activation='selu', padding='same', kernel_regularizer=regularizers.l1_l2(0.0001, 0.0001)))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='selu', padding='same',
                     kernel_regularizer=regularizers.l1_l2(0.0001, 0.0001)))
    model.add(Dropout(0.3))
   # model.add(Dropout(0.3))
    #model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    #model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='selu', padding='same', kernel_regularizer=regularizers.l1_l2(0.001,0.001)))
    model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l1_l2(0.0001,0.0001)))
    model.add(Dropout(0.3))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    #model.add(Bidirectional(GRU(128, return_sequences=True)))
    #model.add(Bidirectional(GRU(128)))
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

def classify_from_embeddings_SVM(embeddings_train, train_labels, embeddings_test):
    clf = svm.SVC()
    clf.fit(embeddings_train, train_labels)
    return clf.predict(embeddings_test)

