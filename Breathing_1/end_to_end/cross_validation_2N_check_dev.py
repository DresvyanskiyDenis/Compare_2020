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
path_to_model_for_part_1='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2020\\models\\best_model_weights_idx_of_part_2.h5'
path_to_model_for_part_2='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2020\\models\\best_model_weights_idx_of_part_3.h5'

# devel data
path_to_devel_data='C:/Users/Dresvyanskiy/Desktop/ComParE2020_Breathing/wav/'
path_to_devel_labels='C:/Users/Dresvyanskiy/Desktop/ComParE2020_Breathing/lab/'
devel_data, devel_labels, devel_dict, frame_rate=load_data(path_to_devel_data, path_to_devel_labels, 'devel')
prepared_devel_data, prepared_devel_labels,prepared_devel_labels_timesteps=prepare_data(devel_data, devel_labels, devel_dict, frame_rate, length_sequence, step_sequence)
devel_parts=divide_data_on_parts(prepared_devel_data, prepared_devel_labels, prepared_devel_labels_timesteps, parts=data_parts, filenames_dict=devel_dict)

total_predicted_labels=pd.DataFrame(columns=devel_labels.columns)

# part 1
part_1=[devel_parts[0]]
part_1_d, part_1_lbs, part_1_timesteps, part_1_filenames_dict = extract_and_reshape_list_of_parts(list_of_parts=part_1)
part_1_filenames_dict = part_1_filenames_dict[0]
part_1_d, _=reshaping_data_for_model(part_1_d, part_1_lbs)
model_for_part_1=create_model(input_shape=(part_1_d.shape[-2], part_1_d.shape[-1]))
model_for_part_1.load_weights(path_to_model_for_part_1)
model_for_part_1.compile(optimizer='Adam', loss=correlation_coefficient_loss, metrics=['mse', 'mae'])
predicted_labels = model_for_part_1.predict(part_1_d, batch_size=batch_size)
concatenated_predicted_labels=concatenate_prediction(predicted_labels, part_1_timesteps, part_1_filenames_dict)
total_predicted_labels=pd.concat((total_predicted_labels, concatenated_predicted_labels), axis=0)
# correlation for the first part?
ground_truth_labels = choose_real_labs_only_with_filenames(devel_labels, list(part_1_filenames_dict.values()))
r=scipy.stats.pearsonr(ground_truth_labels.iloc[:,2].values,concatenated_predicted_labels.iloc[:,2].values)
print('correlation, first part:',r)
# deleting all variables
del model_for_part_1
K.clear_session()
gc.collect()
# part 2
part_2=[devel_parts[1]]
part_2_d, part_2_lbs, part_2_timesteps, part_2_filenames_dict = extract_and_reshape_list_of_parts(list_of_parts=part_2)
part_2_filenames_dict = part_2_filenames_dict[0]
part_2_d, _=reshaping_data_for_model(part_2_d, part_2_lbs)
model_for_part_2=create_model(input_shape=(part_2_d.shape[-2], part_2_d.shape[-1]))
model_for_part_2.load_weights(path_to_model_for_part_2)
model_for_part_2.compile(optimizer='Adam', loss=correlation_coefficient_loss, metrics=['mse', 'mae'])
predicted_labels = model_for_part_2.predict(part_2_d, batch_size=batch_size)
concatenated_predicted_labels=concatenate_prediction(predicted_labels, part_2_timesteps, part_2_filenames_dict)
total_predicted_labels=pd.concat((total_predicted_labels, concatenated_predicted_labels), axis=0)

model_for_part_2=create_model(input_shape=(part_2_d.shape[-2], part_2_d.shape[-1]))
ground_truth_labels = choose_real_labs_only_with_filenames(devel_labels, list(part_2_filenames_dict.values()))
r=scipy.stats.pearsonr(ground_truth_labels.iloc[:,2].values,concatenated_predicted_labels.iloc[:,2].values)
print('correlation, second part:',r)
# deleting all variables
del model_for_part_1
K.clear_session()
gc.collect()



# correlation
total_predicted_labels=total_predicted_labels.sort_values(by=['filename', 'timeFrame'])

r=scipy.stats.pearsonr(devel_labels.iloc[:,2].values,total_predicted_labels.iloc[:,2].values)
print('correlation, total:',r)