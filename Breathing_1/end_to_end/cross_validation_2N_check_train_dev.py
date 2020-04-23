import gc

import scipy

from Breathing_1.end_to_end.cross_validation_2N_train import divide_data_on_parts, extract_and_reshape_list_of_parts, \
    reshaping_data_for_model, choose_real_labs_only_with_filenames, concatenate_prediction
from Breathing_1.end_to_end.utils import create_model, load_data, prepare_data, correlation_coefficient_loss
import pandas as pd
import numpy as np
from keras import backend as K

data_parts=2
batch_size=30
# params
length_sequence=256000
step_sequence=102400
path_to_model_for_part_0='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2020\\models\\best_model_weights_idx_of_part_0.h5'
path_to_model_for_part_1='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2020\\models\\best_model_weights_idx_of_part_1.h5'
path_to_model_for_part_2='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2020\\models\\best_model_weights_idx_of_part_2.h5'
path_to_model_for_part_3='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2020\\models\\best_model_weights_idx_of_part_3.h5'
paths_to_models=[path_to_model_for_part_0, path_to_model_for_part_1, path_to_model_for_part_2, path_to_model_for_part_3]


# train data
path_to_train_data='C:/Users/Dresvyanskiy/Desktop/ComParE2020_Breathing/wav/'
path_to_train_labels='C:/Users/Dresvyanskiy/Desktop/ComParE2020_Breathing/lab/'
train_data, train_labels, train_dict, frame_rate=load_data(path_to_train_data, path_to_train_labels, 'train')
prepared_train_data, prepared_train_labels,prepared_train_labels_timesteps=prepare_data(train_data, train_labels, train_dict, frame_rate, length_sequence, step_sequence)
train_parts=divide_data_on_parts(prepared_train_data, prepared_train_labels, prepared_train_labels_timesteps, parts=data_parts, filenames_dict=train_dict)


# devel data
path_to_devel_data='C:/Users/Dresvyanskiy/Desktop/ComParE2020_Breathing/wav/'
path_to_devel_labels='C:/Users/Dresvyanskiy/Desktop/ComParE2020_Breathing/lab/'
devel_data, devel_labels, devel_dict, frame_rate=load_data(path_to_devel_data, path_to_devel_labels, 'devel')
prepared_devel_data, prepared_devel_labels,prepared_devel_labels_timesteps=prepare_data(devel_data, devel_labels, devel_dict, frame_rate, length_sequence, step_sequence)
devel_parts=divide_data_on_parts(prepared_devel_data, prepared_devel_labels, prepared_devel_labels_timesteps, parts=data_parts, filenames_dict=devel_dict)

ground_truth_labels=pd.DataFrame(columns=train_labels.columns)
total_predicted_labels=pd.DataFrame(columns=train_labels.columns)
total_parts=[train_parts[0], train_parts[1], devel_parts[0], devel_parts[1]]


for i in range(len(paths_to_models)):
    part=[total_parts[i]]
    part_d, part_lbs, part_timesteps, part_filenames_dict = extract_and_reshape_list_of_parts(list_of_parts=part)
    part_filenames_dict = part_filenames_dict[0]
    part_d, _ = reshaping_data_for_model(part_d, part_lbs)
    model = create_model(input_shape=(part_d.shape[-2], part_d.shape[-1]))
    model.load_weights(paths_to_models[i])
    model.compile(optimizer='Adam', loss=correlation_coefficient_loss, metrics=['mse', 'mae'])
    predicted_labels = model.predict(part_d, batch_size=batch_size)
    concatenated_predicted_labels = concatenate_prediction(predicted_labels, part_timesteps, part_filenames_dict)
    total_predicted_labels = pd.concat((total_predicted_labels, concatenated_predicted_labels), axis=0)
    if i<2:
        ground_truth_labels_part = choose_real_labs_only_with_filenames(train_labels, list(part_filenames_dict.values()))
    else:
        ground_truth_labels_part = choose_real_labs_only_with_filenames(devel_labels, list(part_filenames_dict.values()))
    r = scipy.stats.pearsonr(ground_truth_labels_part.iloc[:, 2].values, concatenated_predicted_labels.iloc[:, 2].values)
    ground_truth_labels = pd.concat((ground_truth_labels, ground_truth_labels_part), axis=0)
    print('r on part ', str(i)+':', r)
    del model
    K.clear_session()
    gc.collect()


r=scipy.stats.pearsonr(ground_truth_labels.iloc[:,2].values,total_predicted_labels.iloc[:,2].values)
print('correlation, total:',r)
