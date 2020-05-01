import gc
import os
import time
# test
from utils import load_data, prepare_data, create_model, concatenate_prediction, pearson_coef, \
    correlation_coefficient_loss, new_concatenate_prediction
import numpy as np
import pandas as pd
from keras import backend as K

lengths=[16000, 32000, 64000, 96000, 128000, 160000, 192000, 224000, 256000, 512000]
batch_sizes=[200,100,60,40,30,24,19,17,15,7]

for num_run in range(10):
    window_length=lengths[num_run]
    step=int(window_length*2/5)
    # params
    batch_size=batch_sizes[num_run]
    epochs=110
    best=0
    # train data
    path_to_train_data='/content/drive/My Drive/ComParE2020_Breathing/wav/'
    path_to_train_labels='/content/drive/My Drive/ComParE2020_Breathing/lab/'
    train_data, train_labels, train_dict, frame_rate=load_data(path_to_train_data, path_to_train_labels, 'train')
    prepared_train_data, prepared_train_labels, prepared_train_labels_timesteps=prepare_data(train_data, train_labels, train_dict, frame_rate, window_length, step)
    # instance normalization
    #prepared_train_data, min_train, max_train=sample_minmax_normalization(prepared_train_data)
    # reshaping for training process
    prepared_train_data=prepared_train_data.reshape((prepared_train_data.shape+(1,)))
    prepared_train_data=prepared_train_data.reshape(((-1,)+prepared_train_data.shape[2:]))
    prepared_train_data=prepared_train_data.astype('float32')
    prepared_train_labels=prepared_train_labels.reshape(((-1,)+prepared_train_labels.shape[2:]))
    # model parameters
    input_shape=(prepared_train_data.shape[-2],prepared_train_data.shape[-1])
    output_shape=(prepared_train_labels.shape[-1])

    # validation data
    path_to_validation_data='/content/drive/My Drive/ComParE2020_Breathing/wav/'
    path_to_validation_labels='/content/drive/My Drive/ComParE2020_Breathing/lab/'
    val_data, val_labels, val_dict, frame_rate=load_data(path_to_validation_data, path_to_validation_labels, 'devel')
    prepared_val_data, prepared_val_labels,prepared_val_labels_timesteps=prepare_data(val_data, val_labels, val_dict, frame_rate, window_length, step)

    # reshaping for training process
    prepared_val_data=prepared_val_data.reshape((prepared_val_data.shape+(1,)))
    prepared_val_data=prepared_val_data.reshape(((-1,)+prepared_val_data.shape[2:]))
    prepared_val_data=prepared_val_data.astype('float32')
    prepared_val_labels=prepared_val_labels.reshape(((-1,)+prepared_val_labels.shape[2:]))


    model=create_model(input_shape=input_shape)
    model.compile(optimizer='Adam', loss=correlation_coefficient_loss, metrics=['mse'])


    path_to_stats='stats/'
    if not os.path.exists(path_to_stats): os.mkdir(path_to_stats)
    path_to_best_model='best_model/'
    if not os.path.exists(path_to_best_model): os.mkdir(path_to_best_model)
    train_loss=[]
    val_loss=[]

    for i in range(epochs):
        permutation=np.random.permutation(prepared_train_data.shape[0])
        prepared_train_data=prepared_train_data[permutation]
        prepared_train_labels=prepared_train_labels[permutation]
        history=model.fit(prepared_train_data, prepared_train_labels, batch_size=batch_size, epochs=1,
              shuffle=True, verbose=1, use_multiprocessing=True)
        train_loss.append([history.history['loss'][0],history.history['mse'][0]])
        #model.save(path_to_tmp_model+'model.h5')
        #model.save_weights(path_to_tmp_model+'model_weights.h5')
        if i%1==0:
            predicted_labels=model.predict(prepared_val_data, batch_size=batch_size)
            concatenated_predicted_labels=new_concatenate_prediction(true_values=val_labels, predicted_values=predicted_labels,
                                                                 timesteps_labels=prepared_val_labels_timesteps, class_dict=val_dict)
            prc_coef=pearson_coef(val_labels.iloc[:,2].values,concatenated_predicted_labels.iloc[:,2].values)
            print('iteration:',i,'     pearson_coef:',prc_coef)
            val_loss.append(prc_coef[0])
            pd.DataFrame(columns=['loss','mse'], data=train_loss).to_csv(
              path_to_stats+'train_loss_window_'+str(window_length)+'.csv', index=False)
            pd.DataFrame(columns=['prc_coef'], data=val_loss).to_csv(
              path_to_stats+'val_prc_coefs_window_'+str(window_length)+'.csv', index=False)
            if prc_coef[0]>best:
                best=prc_coef[0]
                model.save(path_to_best_model+'model_window_'+str(window_length)+'.h5')
                model.save_weights(path_to_best_model+'model_weights_window_'+str(window_length)+'.h5')

del model
K.clear_session()
gc.collect()
