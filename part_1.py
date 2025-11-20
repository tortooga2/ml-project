import random
import math
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

import sklearn.linear_model as ln
import sklearn.ensemble as es
from xgboost import XGBClassifier



##Notes on MNISTmini.mat
# 1. all 'images' are 1d arrays of 100 elements (10x10 image)
# 2. there are 4 arrays, train features (actual images), train ground (labels; is this a 1 or 7), test features, test ground




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


Mm_train_feats = "train_fea1"
Mm_train_gnd = "train_gnd1"
Mm_test_feats = "test_fea1"
Mm_test_gnd = "test_gnd1"


Mm_train = (Mm_train_feats, Mm_train_gnd)
Mm_test = (Mm_test_feats, Mm_test_gnd)

## Part 1 ________________________________________________

# data = loadmat("./datasets/MNISTmini.mat")


# # assemble data
# feat_8, gnd_8 = MNIST_get_feats_of_label(data, Mm_train, 1)
# feat_9, gnd_9 = MNIST_get_feats_of_label(data, Mm_train, 7)

# feat = np.concatenate( (feat_8, feat_9), axis=0)
# gnd = np.concatenate( (gnd_8, gnd_9) ).ravel()


# feat_test_8, gnd_test_8 = MNIST_get_feats_of_label(data, Mm_test, 1)
# feat_test_9, gnd_test_9 = MNIST_get_feats_of_label(data, Mm_test, 7)

# feat_test = np.concatenate( (feat_test_8, feat_test_9), axis=0)
# gnd_test = np.concatenate( (gnd_test_8, gnd_test_9) ).ravel()


# #Test logistic regression model

# model_ln = ln.LogisticRegression(solver="liblinear")

# model_ln.fit(feat, gnd)

# y_pred = model_ln.predict(feat_test)

# tp_lin = np.mean(y_pred == gnd_test)


# #Test Random Forest Model.

# #Cross Validation
# #1. split data set into multtiple batches. since we have roughly 10k items lets do 10 batches of 1k

# #since feature and label are two seperate arrays, and we want thier relative position to match, we need to use a index table and do any shuffling or batching like that. 
# idx_array = np.arange(len(feat))


# batches = 20
# est_per_batch = 2
# batch_size = 1000
# results = []
# for i in range(batches):
#     batch_est = (i+1)*est_per_batch


#     np.random.shuffle(idx_array)
#     batch_idx = idx_array[: batch_size]

#     batch_feat = feat[batch_idx]
#     batch_gnd = gnd[batch_idx]

    
    
#     model_rf = es.RandomForestClassifier(n_estimators= batch_est)


#     model_rf.fit(batch_feat, batch_gnd)

#     y_rf_pred = model_rf.predict(feat_test)
#     acc = np.mean(y_rf_pred == gnd_test)
#     results.append((acc, batch_est))
#     print(f"with {batch_est}, we acheived an acc of: {acc}")

# #determine best estimator size:

# results.sort(key=lambda t: t[0])
# (tp_rf, estimators) = results.pop()
# print(f"{estimators} estimators was found to be the best with an accuracy of {tp_rf}")

# #Compute results / score

# model_rf = es.RandomForestClassifier(n_estimators= estimators)
# model_rf.fit(feat, gnd)
# y_rf_pred = model_rf.predict(feat_test)
# tp_rf = np.mean(y_rf_pred == gnd_test)



# print(f"Logistic Regression prediction accuracy: {tp_lin}\nRandom Forest prediction accuracy: {tp_rf}")



## Part 2 ____________________________________________

data = loadmat("./datasets/MNIST.mat")

feat = np.array(data["train_fea"])
gnd = np.array(data["train_gnd"]) - 1

feat_test = np.array(data["test_fea"])
gnd_test = np.array(data["test_gnd"])

model = XGBClassifier(n_estimators=10, max_depth=2)

model.fit(feat, gnd)

y_pred = model.predict(feat_test)
tp = np.mean(y_pred == gnd_test)
print(tp)


