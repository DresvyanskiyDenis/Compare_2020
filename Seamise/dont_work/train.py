from pylab import *

from Seamise.dont_work.utils import create_model, build_model, load_data, preparing_data_for_triplet, get_batch_hard, compute_interclass_distance, vizualization
from functions import normalize_data_instance

path_to_labels='C:\\Users\\Dresvyanskiy\Desktop\\ComParE2020_Elderly\\lab\\'
path_to_train_data='C:\\Users\\Dresvyanskiy\Desktop\\ComParE2020_Elderly\\Mel_spectrogram_2\\'

# data preprocessing
train_data, labels=load_data(path_to_labels=path_to_labels, path_to_data=path_to_train_data,
                                 prefix='train', kind_labels='A_cat_no')
train_data=normalize_data_instance(train_data)
train_data=preparing_data_for_triplet(train_data,labels)


# Hyper parameters
batch_size = 16
n_iter = 2000 # No. of training iterations
network=create_model()
network_train=build_model(input_shape=(train_data[0].shape[1],train_data[0].shape[2],train_data[0].shape[3]),
                          network=network)
network_train.compile(loss=None, optimizer='Adam')
print('---------------------------train-----------------------')
#vizualization(train_data,np.unique(labels).shape[0], network)
for i in range(n_iter):
    triplets=get_batch_hard(train_data,np.unique(labels).shape[0],100,8,8,network)
    loss=network_train.train_on_batch(triplets, None)
    dists=compute_interclass_distance(train_data, np.unique(labels).shape[0], network)
    print('loss:', loss)
    print(dists)
vizualization(train_data,np.unique(labels).shape[0], network)
