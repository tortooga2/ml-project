from scipy.io import loadmat


import matplotlib.pyplot as plt

import random

import numpy as np

import sklearn.linear_model as ln
import sklearn.ensemble as es




##Notes on MNISTmini.mat
# 1. all 'images' are 1d arrays of 100 elements (10x10 image)
# 2. there are 4 arrays, train features (actual images), train ground (labels; is this a 1 or 7), test features, test ground
Mm_train_feats = "train_fea1"
Mm_train_gnd = "train_gnd1"
Mm_test_feats = "test_fea1"
Mm_test_gnd = "test_gnd1"


Mm_train = (Mm_train_feats, Mm_train_gnd)
Mm_test = (Mm_test_feats, Mm_test_gnd)



def read_mat(filename : str):
    data = loadmat(filename)

    return data


#Returns a list of indices where each number starts, since dataset has no zeros, zeroth entry in 'keys', is None. Technically a look up table.  
def MNIST_get_lookup(data, data_set):
    keys = [None]
    current_set_num = 0

    for idx, item in enumerate(data[data_set[1]]):
        if item[0] != current_set_num:
            current_set_num = item[0]
            keys.append(idx)
    return keys

def MNIST_get_feats_of_label(data, data_set, label : int):
    table = MNIST_get_lookup(data, data_set)
    start = table[label]
    if label + 1 < len(table):
        end = table[label + 1]
    else:
        end = len(data[data_set[1]])

    return (data[data_set[0]][start : end], data[data_set[1]][start : end])





data = loadmat("./datasets/MNISTmini.mat")


feat_8, gnd_8 = MNIST_get_feats_of_label(data, Mm_train, 8)
feat_9, gnd_9 = MNIST_get_feats_of_label(data, Mm_train, 9)

feat = np.concatenate( (feat_8, feat_9), axis=0)
gnd = np.concatenate( (gnd_8, gnd_9) ).ravel()

model_ln = ln.LogisticRegression(max_iter=100)
model_rf = es.RandomForestClassifier(n_estimators=2)

model_ln.fit(feat, gnd)
model_rf.fit(feat, gnd)

feat_test_8, gnd_test_8 = MNIST_get_feats_of_label(data, Mm_test, 8)
feat_test_9, gnd_test_9 = MNIST_get_feats_of_label(data, Mm_test, 9)

feat_test = np.concatenate( (feat_test_8, feat_test_9), axis=0)
gnd_test = np.concatenate( (gnd_test_8, gnd_test_9) ).ravel()


y_pred = model_ln.predict(feat_test)
y_rf_pred = model_rf.predict(feat_test)


tp_lin = np.mean(y_pred == gnd_test)
rf_lin = np.mean(y_rf_pred == gnd_test)
print(tp_lin, rf_lin)







