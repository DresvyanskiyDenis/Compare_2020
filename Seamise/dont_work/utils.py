import numpy as np
import pandas as pd
from keras import Sequential, Model
from keras.engine.topology import Layer
from keras.layers import Input, Conv2D, Dropout, AveragePooling2D, TimeDistributed, Bidirectional, Dense, GRU, Flatten, \
    Lambda
from keras.utils import to_categorical
from keras import backend as K
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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

def preparing_data_for_triplet(data, labels):
    temp_labels=np.reshape(labels, newshape=(-1,))
    n_classes=np.unique(temp_labels).shape[0]
    result_data=[]
    for cl in range(n_classes):
        images_class_n=data[temp_labels==cl]
        result_data.append(images_class_n)
    return result_data


def get_batch_random(data,batch_size, nb_classes):
    """
    Create batch of APN triplets with a complete random strategy

    Arguments:
    batch_size -- integer

    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c)
    """
    X=data

    m, w, h, c = X[0].shape

    # initialize result
    triplets = [np.zeros((batch_size, w, h, c)) for i in range(3)]

    for i in range(batch_size):
        # Pick one random class for anchor
        anchor_class = np.random.randint(0, nb_classes)
        nb_sample_available_for_class_AP = X[anchor_class].shape[0]

        # Pick two different random pics for this class => A and P
        [idx_A, idx_P] = np.random.choice(nb_sample_available_for_class_AP, size=2, replace=False)

        # Pick another class for N, different from anchor_class
        negative_class = (anchor_class + np.random.randint(1, nb_classes)) % nb_classes
        nb_sample_available_for_class_N = X[negative_class].shape[0]

        # Pick a random pic for this negative class => N
        idx_N = np.random.randint(0, nb_sample_available_for_class_N)

        triplets[0][i, :, :, :] = X[anchor_class][idx_A, :, :, :]
        triplets[1][i, :, :, :] = X[anchor_class][idx_P, :, :, :]
        triplets[2][i, :, :, :] = X[negative_class][idx_N, :, :, :]

    return triplets


def get_batch_hard(data,nb_classes, draw_batch_size, hard_batchs_size, norm_batchs_size, network):
    """
    Create batch of APN "hard" triplets

    Arguments:
    draw_batch_size -- integer : number of initial randomly taken samples
    hard_batchs_size -- interger : select the number of hardest samples to keep
    norm_batchs_size -- interger : number of random samples to add

    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    """
    X=data

    m, w, h, c = X[0].shape

    # Step 1 : pick a random batch to study
    studybatch = get_batch_random(data, draw_batch_size, nb_classes)

    # Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
    studybatchloss = np.zeros((draw_batch_size))

    # Compute embeddings for anchors, positive and negatives
    A = network.predict(studybatch[0])
    P = network.predict(studybatch[1])
    N = network.predict(studybatch[2])

    # Compute d(A,P)-d(A,N)
    studybatchloss = np.sum(np.square(A - P), axis=1) - np.sum(np.square(A - N), axis=1)

    # Sort by distance (high distance first) and take the
    selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]

    # Draw other random samples from the batch
    selection2 = np.random.choice(np.delete(np.arange(draw_batch_size), selection), norm_batchs_size, replace=False)

    selection = np.append(selection, selection2)

    triplets = [studybatch[0][selection, :, :, :], studybatch[1][selection, :, :, :], studybatch[2][selection, :, :, :]]

    return triplets

def create_model():
    model = Sequential()
    model.add(Conv2D(input_shape=(313, 128, 1), filters=64, kernel_size=(4, 4), strides=(2, 2), activation='selu',
                     padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='selu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='selu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='selu', padding='same'))
    model.add(Dropout(0.2))
    model.add(AveragePooling2D(pool_size=(1, 2), strides=(1, 2), padding='same'))
    model.add(Flatten())
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    #model.add(Bidirectional(GRU(128, return_sequences=True)))
    #model.add(Bidirectional(GRU(128)))
    return model


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


def build_model(input_shape, network, margin=0.5):
    '''
    Define the Keras Model for training
        Input :
            input_shape : shape of input images
            network : Neural network to train outputing embeddings
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)

    '''
    # Define the tensors for the three input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input")

    # Generate the encodings (feature vectors) for the three images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)

    # TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=margin, name='triplet_loss_layer')([encoded_a, encoded_p, encoded_n])

    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss_layer)

    # return the model
    return network_train


def compute_interclass_distance(data, nb_classes, network):
    centroids=[]
    for cl in data:
        predictions=network.predict(cl)
        size=predictions.shape[0]
        predictions=np.sum(predictions, axis=0)/size
        centroids.append(predictions)
    distances=np.zeros(shape=(nb_classes, nb_classes))
    for i in range(nb_classes):
        for j in range(i, nb_classes):
            if i==j:
                distances[i,j]=0
            else:
                distances[i,j]=euclidean(centroids[i], centroids[j])
                distances[j,i]=distances[i,j]
    return distances

def vizualization(data, nb_classes, network):
    embeddings=[]
    colors=['red', 'green', 'blue']
    for i in range(nb_classes):
        embeddings.append(network.predict(data[i]))
    data_to_train_tsne=np.empty(shape=(0,embeddings[0].shape[1]))
    for i in range(nb_classes):
        data_to_train_tsne=np.concatenate((data_to_train_tsne,embeddings[i]))
    indexes=[data[0].shape[0]]
    for i in range(1,nb_classes):
        indexes.append(indexes[-1]+data[i].shape[0])
    new_data=TSNE(n_components=2).fit_transform(data_to_train_tsne)
    last_point=0
    for i in range(nb_classes):
        plt.scatter(new_data[last_point:int(indexes[i])][:,0], new_data[last_point:int(indexes[i])][:,1], c=colors[i])
        last_point=int(indexes[i])
    plt.show()