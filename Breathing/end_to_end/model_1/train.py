import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from Breathing.end_to_end.utils import load_data, prepare_data, create_model, concatenate_prediction, pearson_coef, \
    instance_normalization, correlation_coefficient_loss, sample_minmax_normalization
import numpy as np
# train data
path_to_train_data='C:\\Users\\Denis\\Desktop\\Compare2020_Breathing\\wav\\'
path_to_train_labels='C:\\Users\\Denis\\Desktop\\Compare2020_Breathing\\lab\\'
train_data, train_labels, train_dict, frame_rate=load_data(path_to_train_data, path_to_train_labels, 'train')
prepared_train_data, prepared_train_labels, prepared_train_labels_timesteps=prepare_data(train_data, train_labels, train_dict, frame_rate, 384000, 153600)
# instance normalization
#prepared_train_data, min_train, max_train=sample_minmax_normalization(prepared_train_data)
# reshaping for training process
prepared_train_data=prepared_train_data.reshape((prepared_train_data.shape+(1,)))
prepared_train_data=prepared_train_data.reshape(((-1,)+prepared_train_data.shape[2:]))
prepared_train_labels=prepared_train_labels.reshape(((-1,)+prepared_train_labels.shape[2:]))
# model parameters
input_shape=(prepared_train_data.shape[-2],prepared_train_data.shape[-1])
output_shape=(prepared_train_labels.shape[-1])

# validation data
path_to_validation_data='C:\\Users\\Denis\\Desktop\\Compare2020_Breathing\\wav\\'
path_to_validation_labels='C:\\Users\\Denis\\Desktop\\Compare2020_Breathing\\lab\\'
val_data, val_labels, val_dict, frame_rate=load_data(path_to_validation_data, path_to_validation_labels, 'devel')
prepared_val_data, prepared_val_labels,prepared_val_labels_timesteps=prepare_data(val_data, val_labels, val_dict, frame_rate, 384000, 153600)
# instance normalization
#val_data=sample_minmax_normalization(val_data, min_train, max_train)
# reshaping for training process
prepared_val_data=prepared_val_data.reshape((prepared_val_data.shape+(1,)))
prepared_val_data=prepared_val_data.reshape(((-1,)+prepared_val_data.shape[2:]))
prepared_val_labels=prepared_val_labels.reshape(((-1,)+prepared_val_labels.shape[2:]))


model=create_model(input_shape=input_shape, output_shape=output_shape)
model.compile(optimizer='Adam', loss=correlation_coefficient_loss, metrics=['mse', 'mae'])
batch_size=40
epochs=200
coefs=[]
best=0
for i in range(epochs):
    model.fit(prepared_train_data, prepared_train_labels, batch_size=batch_size, epochs=1,
          shuffle=True, verbose=2, use_multiprocessing=True)
    if i%3==0:
        predicted_labels=model.predict(prepared_val_data, batch_size=batch_size)
        concatenated_predicted_labels=concatenate_prediction(true_values=val_labels, predicted_values=predicted_labels,
                                                             timesteps_labels=prepared_val_labels_timesteps, class_dict=val_dict)
        prc_coef=pearson_coef(val_labels.iloc[:,2].values,concatenated_predicted_labels.iloc[:,2].values)
        print('iteration:',i,'     pirson:',prc_coef)
        coefs.append(np.abs(prc_coef[0]))
        if prc_coef[0]>best:
            best=prc_coef[0]
            model.save('model.h5')
            model.save_weights('model_weights.h5')

print('best:', np.max(np.array(coefs)))