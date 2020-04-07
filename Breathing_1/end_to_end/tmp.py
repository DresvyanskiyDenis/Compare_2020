import numpy as np
from scipy.stats import pearsonr
from keras import backend as K
import tensorflow as tf
x1=np.array([23,5634,123,234,364,123,543])
y1=np.array([123,5,6745,2134,7432,457,11])
print(pearsonr(x1,y1))

x2=np.array([123,523,123,543,756,123,346])
y2=np.array([132,89,567,234,7,879,1])
print(pearsonr(x2,y2))

x3=np.array([234,1,2,4,12,543,3])
y3=np.array([123,4,54,234,5,767,98])
print(pearsonr(x3,y3))
x=np.array([x1,x2,x3])
y=np.array([y1,y2,y3])

x_tf=tf.Variable(x, dtype=tf.float32)
y_tf=tf.Variable(y, dtype=tf.float32)

def pearson_corr(y_true, y_pred):
    x=y_true
    y=y_pred
    mx=K.mean(x, axis=1, keepdims=True)
    my=K.mean(y, axis=1, keepdims=True)
    xm,ym=x-mx,y-my
    r_num=K.sum(tf.multiply(xm, ym), axis=1)
    sum_square_x=K.sum(K.square(xm), axis=1)
    sum_square_y = K.sum(K.square(ym), axis=1)
    sqrt_x = tf.sqrt(sum_square_x)
    sqrt_y = tf.sqrt(sum_square_y)
    r_den=tf.multiply(sqrt_x, sqrt_y)
    result=tf.divide(r_num, r_den)
    #tf.print('result:', result)
    result=K.mean(result)
    #tf.print('mean result:', result)
    return 1-result

pearson_corr(x_tf, y_tf)