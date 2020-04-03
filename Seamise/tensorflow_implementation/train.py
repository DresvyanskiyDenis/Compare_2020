import os

from keras import Input, Model, optimizers
from keras.layers import concatenate
import numpy as np
from sklearn.metrics import recall_score

from Seamise.tensorflow_implementation.utils_2 import create_model, load_data, normalize_data_instance, \
    triplet_loss_adapted_from_tf, \
    vizualization, classify_from_embeddings_SVM

path_to_labels = 'C:\\Users\\Denis\\Desktop\\ComParE2020_Elderly\\ComParE2020_Elderly\\lab\\'
path_to_train_data = 'C:\\Users\\Denis\\Desktop\\ComParE2020_Elderly\\ComParE2020_Elderly\\Mel_spectrogram_2\\Mel_spectrogram_2\\'

# data preprocessing
train_data, labels=load_data(path_to_labels=path_to_labels, path_to_data=path_to_train_data,
                                 prefix='train', kind_labels='A_cat_no')
train_data=normalize_data_instance(train_data)

validation_data, val_labels=load_data(path_to_labels=path_to_labels, path_to_data=path_to_train_data,
                                 prefix='devel', kind_labels='A_cat_no')
validation_data=normalize_data_instance(validation_data)


network= create_model()

input_images = Input(shape=train_data[0].shape, name='input_image') # input layer for images
input_labels = Input(shape=(1,), name='input_label')    # input layer for labels
embeddings = network([input_images])               # output of network -> embeddings
labels_plus_embeddings = concatenate([input_labels, embeddings])  # concatenating the labels + embeddings
model = Model(inputs=[input_images, input_labels],
                      outputs=labels_plus_embeddings)

model.summary()

opt=optimizers.Nadam()
model.compile(loss=triplet_loss_adapted_from_tf, optimizer=opt)
batch_size=48
epochs=20
# Uses 'dummy' embeddings + dummy gt labels. Will be removed as soon as loaded, to free memory
dummy_gt_train = np.zeros((train_data.shape[0], batch_size + 1))
dummy_gt_val = np.zeros((validation_data.shape[0], batch_size + 1))

best=0
num_best=0
for epoch in range(epochs):
    H = model.fit( x=[train_data, labels], y=dummy_gt_train,
        batch_size=batch_size,
        epochs=1,
        validation_data=([validation_data, val_labels], dummy_gt_val),
                   verbose=2)

    path_to_visualization='figures\\'
    if not os.path.exists(path_to_visualization):
        os.mkdir(path_to_visualization)
    prefix_to_visualization='train_'+str(epoch)
    vizualization(train_data, labels, network, path_to_save=path_to_visualization, prefix=prefix_to_visualization)
    prefix_to_visualization='dev_'+str(epoch)
    vizualization(validation_data, val_labels, network, path_to_save=path_to_visualization, prefix=prefix_to_visualization)

    train_embeddings=network.predict(train_data)
    val_embeddings=network.predict(validation_data)

    predictions_val_embeddings=classify_from_embeddings_SVM(train_embeddings, labels.reshape((-1)), val_embeddings)
    UAC = recall_score(y_true=val_labels.reshape((-1)), y_pred=predictions_val_embeddings, average='macro')
    print('dev UAC::',UAC)
    if best<=UAC:
        best=UAC
        num_best=epoch

    predictions_val_embeddings=classify_from_embeddings_SVM(train_embeddings, labels.reshape((-1)), train_embeddings)
    UAC = recall_score(y_true=labels.reshape((-1)), y_pred=predictions_val_embeddings, average='macro')
    print('train UAC:',UAC)

    path_to_save_model='model\\'
    if not os.path.exists(path_to_save_model):
        os.mkdir(path_to_save_model)
    network.save(path_to_save_model+'model'+str(epoch)+'.h5')
    network.save_weights(path_to_save_model+'weights'+str(epoch)+'.h5')

print('num best:', num_best)