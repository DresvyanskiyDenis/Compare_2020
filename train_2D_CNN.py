import os

import keras
from keras import backend as K
from keras import Sequential, metrics
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, TimeDistributed, LSTM, AveragePooling2D, Dropout, \
    Bidirectional
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf

from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler

# data preprocessing
from sklearn.utils import compute_class_weight

from functions import create_model, normalize_data, load_data, normalize_data_instance, vizualization, \
    create_model_ResNet, prepare_data_for_ResNet, minmax_normalization_instance, create_model_1dconv

path_to_labels='C:\\Users\\Denis\\Desktop\\ComParE2020_Elderly\\ComParE2020_Elderly\\lab\\'
path_to_train_data='C:\\Users\\Denis\\Desktop\\ComParE2020_Elderly\\ComParE2020_Elderly\\Mel_spectrogram_2\\Mel_spectrogram_2\\'
path_to_models = 'models\\'
if not os.path.exists(path_to_models):
    os.mkdir(path_to_models)


# data preprocessing
train_data, labels = load_data(path_to_labels=path_to_labels, path_to_data=path_to_train_data,
                               prefix='train', kind_labels='A_cat_no')
#train_data = minmax_normalization_instance(train_data)
#train_data=prepare_data_for_ResNet(train_data)
#train_data=train_data.reshape((train_data.shape[0],-1,train_data.shape[-1]))
train_data=normalize_data_instance(train_data)

validation_data, val_labels = load_data(path_to_labels=path_to_labels, path_to_data=path_to_train_data,
                                        prefix='devel', kind_labels='A_cat_no')
#validation_data = minmax_normalization_instance(validation_data)
#validation_data=prepare_data_for_ResNet(validation_data)
#validation_data=validation_data.reshape((validation_data.shape[0],-1,validation_data.shape[-1]))
validation_data=normalize_data_instance(validation_data)


results = []
num_runs = 1
for i in range(num_runs):
    # model construction
    model = create_model(input_shape=(313,128,1))
    loss1=keras.losses.sparse_categorical_crossentropy
    model.compile(optimizer='Nadam', loss=loss1,
                  metrics=['accuracy'])
    print(model.summary())

    # training
    batch_size = 48
    epochs = 20
    class_weight_list = compute_class_weight('balanced', np.unique(labels), labels.reshape((-1)))
    class_weights = dict(zip(np.unique(labels), class_weight_list))
    #labels = to_categorical(labels, np.unique(labels).shape[0])
    #val_labels = to_categorical(val_labels, np.unique(val_labels).shape[0])
    best_result = 0

    for epoch in range(0, epochs):
        model.fit(x=train_data, y=labels, batch_size=batch_size, epochs=1, shuffle=True, validation_data=(train_data, labels), verbose=2, class_weight=class_weights)
        if epoch==15:
            a=1+2
        predictions = model.predict(train_data)
        predictions = np.argmax(predictions, axis=1)
        UAC = recall_score(y_true=labels.reshape((-1,)), y_pred=predictions, average='macro')
        print(
            'train-----------------------------------------------------------------------------------------------------------------',
            UAC)

        predictions = model.predict(validation_data)
        predictions = np.argmax(predictions, axis=1)
        UAC = recall_score(y_true=val_labels.reshape((-1,)), y_pred=predictions, average='macro')
        print(
            'val-----------------------------------------------------------------------------------------------------------------',
            UAC)
        if best_result < UAC:
            best_result = UAC
            model.save_weights(path_to_models + 'weights' + str(i) + '.h5')
            model.save(path_to_models + 'model' + str(i) + '.h5')
    results.append(best_result)
    # print best results
    '''model=create_model()
    model.load_weights(path_to_models+'weights'+str(i)+'.h5')
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[metrics.Recall(class_id=0),metrics.Recall(class_id=1),metrics.Recall(class_id=2)])
    predictions = model.predict(validation_data)
    predictions = np.argmax(predictions, axis=1)
    UAC = recall_score(y_true=np.argmax(val_labels, axis=1), y_pred=predictions, average='macro')
    print('best result:',UAC)'''

print('average result:', np.sum(np.array(results)) / num_runs)
print('best result:', np.max(np.array(results)))
print('num_model:', np.argmax(np.array(results)))

