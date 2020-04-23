import os

import pandas as pd
import numpy as np
import scipy
import tensorflow as tf
import gc

from Breathing_1.end_to_end.utils import create_model, load_data, prepare_data, correlation_coefficient_loss

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

def divide_data_on_parts(data, labels, timesteps, filenames_dict, parts=2):
    list_parts=[]
    length_part=int(data.shape[0]/parts)
    start_point=0
    for i in range(parts-1):
        tmp_data=data[start_point:(start_point+length_part)]
        tmp_labels=labels[start_point:(start_point+length_part)]
        tmp_timesteps = timesteps[start_point:(start_point + length_part)]
        tmp_filenames_dict={}
        idx=0
        for j in range(start_point,start_point+length_part):
            tmp_filenames_dict[idx]=list(filenames_dict.values())[j]
            idx+=1
        list_parts.append((tmp_data, tmp_labels, tmp_timesteps, tmp_filenames_dict))
        start_point+=length_part
    tmp_data = data[start_point:]
    tmp_labels = labels[start_point:]
    tmp_timesteps = timesteps[start_point:]
    tmp_filenames_dict = {}
    idx = 0
    for j in range(start_point, data.shape[0]):
        tmp_filenames_dict[idx] = list(filenames_dict.values())[j]
        idx += 1
    list_parts.append((tmp_data, tmp_labels, tmp_timesteps,tmp_filenames_dict))
    return list_parts


def form_train_and_val_datasets(train_parts, dev_parts, index_for_validation_part):
    total=[]
    for i in range(len(train_parts)):
        total.append(train_parts[i])
    for i in range(len(dev_parts)):
        total.append((dev_parts[i]))
    val_dataset=[total.pop(index_for_validation_part)]
    train_dataset=total
    return train_dataset, val_dataset

def extract_and_reshape_list_of_parts(list_of_parts):
    data=list_of_parts[0][0]
    labels=list_of_parts[0][1]
    timesteps=list_of_parts[0][2]
    dicts=[list_of_parts[0][3]]
    for i in range(1,len(list_of_parts)):
        data=np.append(data,list_of_parts[i][0], axis=0)
        labels = np.append(labels, list_of_parts[i][1], axis=0)
        timesteps = np.append(timesteps, list_of_parts[i][2], axis=0)
        dicts.append(list_of_parts[i][3])
    return data, labels, timesteps, dicts

def reshaping_data_for_model(data, labels):
    result_data=data.reshape((-1,data.shape[2])+(1,))
    result_labels=labels.reshape((-1,labels.shape[2]))
    return result_data, result_labels

def concatenate_prediction(predicted_values, labels_timesteps, filenames_dict, columns_for_real_labels=['filename', 'timeFrame', 'upper_belt']):
    predicted_values=predicted_values.reshape(labels_timesteps.shape)
    num_timesteps=np.unique(labels_timesteps[0]).shape[0]
    result_predicted_values = pd.DataFrame(data=np.zeros((num_timesteps*predicted_values.shape[0], len(columns_for_real_labels))), columns=columns_for_real_labels, dtype='float32')
    result_predicted_values_idx=0
    for instance_idx in range(predicted_values.shape[0]):
        timesteps=np.unique(labels_timesteps[instance_idx])
        for timestep in timesteps:
            result_predicted_values.iloc[result_predicted_values_idx, 0]=filenames_dict[instance_idx]
            result_predicted_values.iloc[result_predicted_values_idx, 1]=timestep
            result_predicted_values.iloc[result_predicted_values_idx, 2]=np.mean(predicted_values[instance_idx,labels_timesteps[instance_idx]==timestep])
            result_predicted_values_idx+=1
    return result_predicted_values

def choose_real_labs_only_with_filenames(labels, filenames):
    return labels[labels['filename'].isin(filenames)]



'''# params
length_sequence=256000
step_sequence=102400
batch_size=45
epochs=150
data_parts=2
path_to_save_best_model='best_models/'
if not os.path.exists(path_to_save_best_model):
    os.mkdir(path_to_save_best_model)
path_to_tmp_model='tmp_model/'
if not os.path.exists(path_to_tmp_model):
    os.mkdir(path_to_tmp_model)

# train data
path_to_train_data='/content/drive/My Drive/ComParE2020_Breathing/wav/'
path_to_train_labels='/content/drive/My Drive/ComParE2020_Breathing/lab/'
train_data, train_labels, train_dict, frame_rate=load_data(path_to_train_data, path_to_train_labels, 'train')
prepared_train_data, prepared_train_labels, prepared_train_labels_timesteps=prepare_data(train_data, train_labels, train_dict, frame_rate, length_sequence, step_sequence)
train_parts=divide_data_on_parts(prepared_train_data, prepared_train_labels, prepared_train_labels_timesteps, parts=data_parts, filenames_dict=train_dict)

# devel data
path_to_devel_data='/content/drive/My Drive/ComParE2020_Breathing/wav/'
path_to_devel_labels='/content/drive/My Drive/ComParE2020_Breathing/lab/'
devel_data, devel_labels, devel_dict, frame_rate=load_data(path_to_devel_data, path_to_devel_labels, 'devel')
prepared_devel_data, prepared_devel_labels,prepared_devel_labels_timesteps=prepare_data(devel_data, devel_labels, devel_dict, frame_rate, length_sequence, step_sequence)
devel_parts=divide_data_on_parts(prepared_devel_data, prepared_devel_labels, prepared_devel_labels_timesteps, parts=data_parts, filenames_dict=devel_dict)

for index_of_part in range(0, 3):
    best_result=0
    coefs=[]
    train_dataset, val_dataset=form_train_and_val_datasets(train_parts, devel_parts, index_for_validation_part=index_of_part)
    train_d, train_lbs, train_timesteps, _ = extract_and_reshape_list_of_parts(list_of_parts=train_dataset)
    val_d, val_lbs, val_timesteps, val_filenames_dict=extract_and_reshape_list_of_parts(list_of_parts=val_dataset)
    val_filenames_dict=val_filenames_dict[0]
    train_d, train_lbs=reshaping_data_for_model(train_d, train_lbs)
    val_d, _val_lbs=reshaping_data_for_model(val_d, val_lbs)
    if index_of_part<(len(train_parts)+len(devel_parts))/2:
        ground_truth_labels=choose_real_labs_only_with_filenames(train_labels,list(val_filenames_dict.values()))
    else:
        ground_truth_labels = choose_real_labs_only_with_filenames(devel_labels, list(val_filenames_dict.values()))
    model=create_model(input_shape=(train_d.shape[-2], train_d.shape[-1]))
    model.compile(optimizer='Adam', loss=correlation_coefficient_loss, metrics=['mse', 'mae'])
    for epoch in range(epochs):
        permutations=np.random.permutation(train_d.shape[0])
        train_d, train_lbs=train_d[permutations], train_lbs[permutations]
        model.fit(train_d, train_lbs, batch_size=batch_size, epochs=1,
                  shuffle=True, verbose=1, use_multiprocessing=True, 
                  validation_data=(val_d, _val_lbs), callbacks=[MyCustomCallback()])
        model.save_weights(path_to_tmp_model+'tmp_model_weights_idx_of_part_'+str(index_of_part)
                          +'_epoch_'+str(epoch)+'.h5')
        if epoch>2 and epoch%2==0:
            predicted_labels = model.predict(val_d, batch_size=batch_size)
            concatenated_predicted_labels=concatenate_prediction(predicted_labels, val_timesteps, val_filenames_dict)
            prc_coef=scipy.stats.pearsonr(ground_truth_labels.iloc[:,2].values,concatenated_predicted_labels.iloc[:,2].values)
            print('epoch:', epoch, '     pirson:', prc_coef)
            coefs.append(np.abs(prc_coef[0]))
            if prc_coef[0] > best_result:
                best_result = prc_coef[0]
                model.save_weights(path_to_save_best_model+'best_model_weights_idx_of_part_'+str(index_of_part)+'.h5')
    del model
    K.clear_session()'''