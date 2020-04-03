import os
import time

from keras import Input, Model
from keras.layers import concatenate
import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score

from Seamise.keras_loss_implementation.utils import load_data, normalize_data_instance, create_model, vizualization, \
    triplet_loss, create_batch_hard, classify_from_embeddings_SVM

path_to_labels='C:\\Users\\Denis\\Desktop\\ComParE2020_Elderly\\ComParE2020_Elderly\\lab\\'
path_to_train_data='C:\\Users\\Denis\\Desktop\\ComParE2020_Elderly\\ComParE2020_Elderly\\Mel_spectrogram_2\\Mel_spectrogram_2\\'
path_to_save_model='model\\'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# data preprocessing
train_data, labels=load_data(path_to_labels=path_to_labels, path_to_data=path_to_train_data,
                                 prefix='train', kind_labels='A_cat_no')
train_data=normalize_data_instance(train_data)

validation_data, val_labels=load_data(path_to_labels=path_to_labels, path_to_data=path_to_train_data,
                                 prefix='devel', kind_labels='A_cat_no')
validation_data=normalize_data_instance(validation_data)


# parameters
shape1=313
shape2=128
input_shape=(shape1, shape2, 1)
network= create_model(input_shape)

A = tf.keras.layers.Input(shape=input_shape, name = 'anchor')
P = tf.keras.layers.Input(shape=input_shape, name = 'anchorPositive')
N = tf.keras.layers.Input(shape=input_shape, name = 'anchorNegative')

enc_A = network(A)
enc_P = network(P)
enc_N = network(N)
conc=tf.keras.layers.concatenate([enc_A, enc_P, enc_N], axis=-1, name='merged_layer')
tripletModel = tf.keras.Model(inputs=[A, P, N], outputs=conc)
tripletModel.summary()


tripletModel.compile(loss=triplet_loss, optimizer='Nadam')
print('-----------------------------------train---------------------------------')
random_batch_size=10
hard_batch_size=26
epochs=30
steps=100
start=time.time()
for i in range(epochs*steps):
    anchors, pos, neg=create_batch_hard(train_data, labels, 150, hard_batch_size, random_batch_size, network)
    test_batch=[anchors, pos, neg]
    test_y=np.empty((random_batch_size+hard_batch_size,1))
    loss=tripletModel.train_on_batch(x=test_batch, y=test_y)
    if i%10==0:
        print('iteration:', i, '    loss:',loss, '   time:', time.time()-start)
        train_embeddings = network.predict(train_data)
        val_embeddings = network.predict(validation_data)
        predictions_val_embeddings = classify_from_embeddings_SVM(train_embeddings, labels.reshape((-1)),
                                                                  val_embeddings)
        UAC = recall_score(y_true=val_labels.reshape((-1)), y_pred=predictions_val_embeddings, average='macro')
        print('dev UAC::', UAC)

        predictions_val_embeddings = classify_from_embeddings_SVM(train_embeddings, labels.reshape((-1)),
                                                                  train_embeddings)
        UAC = recall_score(y_true=labels.reshape((-1)), y_pred=predictions_val_embeddings, average='macro')
        print('train UAC:', UAC)
        start = time.time()

path_to_visualization='figures\\'
if not os.path.exists(path_to_visualization):
    os.mkdir(path_to_visualization)
prefix_to_visualization='train_1'
vizualization(train_data, labels, network, path_to_save=path_to_visualization, prefix=prefix_to_visualization)
prefix_to_visualization='dev_1'
vizualization(validation_data, val_labels, network, path_to_save=path_to_visualization, prefix=prefix_to_visualization)

train_embeddings=network.predict(train_data)
val_embeddings=network.predict(validation_data)

predictions_val_embeddings=classify_from_embeddings_SVM(train_embeddings, labels.reshape((-1)), val_embeddings)
UAC = recall_score(y_true=val_labels.reshape((-1)), y_pred=predictions_val_embeddings, average='macro')
print('dev UAC::',UAC)

predictions_val_embeddings=classify_from_embeddings_SVM(train_embeddings, labels.reshape((-1)), train_embeddings)
UAC = recall_score(y_true=labels.reshape((-1)), y_pred=predictions_val_embeddings, average='macro')
print('train UAC:',UAC)

if not os.path.exists(path_to_save_model):
    os.mkdir(path_to_save_model)
network.save(path_to_save_model+'model1.h5')
network.save_weights(path_to_save_model+'weights1.h5')