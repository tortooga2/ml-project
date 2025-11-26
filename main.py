import random
import math
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

import sklearn.linear_model as ln
import sklearn.ensemble as es
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.glassbox import ExplainableBoostingRegressor




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




def cross_validate_model(model, train, param_grid, batch_size):
    m = model
    grid_search = GridSearchCV(estimator=m, param_grid=param_grid, cv=batch_size, verbose=3)
    grid_search.fit(train["feat"], train["gnd"])
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"[BEST] Params: {best_params} -> Average Accuracy: {best_score}")
    return best_params


Mm_train_feats = "train_fea1"
Mm_train_gnd = "train_gnd1"
Mm_test_feats = "test_fea1"
Mm_test_gnd = "test_gnd1"


Mm_train = (Mm_train_feats, Mm_train_gnd)
Mm_test = (Mm_test_feats, Mm_test_gnd)

## Part 1 ________________________________________________

data = loadmat("./datasets/MNISTmini.mat")


# # assemble data
feat_8, gnd_8 = MNIST_get_feats_of_label(data, Mm_train, 8)
feat_9, gnd_9 = MNIST_get_feats_of_label(data, Mm_train, 9)

feat = np.concatenate( (feat_8, feat_9), axis=0)
gnd = np.concatenate( (gnd_8, gnd_9) ).ravel()


feat_test_8, gnd_test_8 = MNIST_get_feats_of_label(data, Mm_test, 8)
feat_test_9, gnd_test_9 = MNIST_get_feats_of_label(data, Mm_test, 9)

feat_test = np.concatenate( (feat_test_8, feat_test_9), axis=0)
gnd_test = np.concatenate( (gnd_test_8, gnd_test_9) ).ravel()


# #Test logistic regression model

param_grid_lin = {
    "max_iter" : [100, 200, 500],
}

best_params_lin = cross_validate_model(
    ln.LogisticRegression(),
    {"feat" : feat, "gnd" : gnd},
    param_grid_lin,
    5
)


# #Test Random Forest Model.

param_grid_rf = {
    "n_estimators" : [10, 50, 100],
    "max_depth" : [1, 3, 5],
}

best_params_rf = cross_validate_model(
    es.RandomForestClassifier(),
    {"feat" : feat, "gnd" : gnd},
    param_grid_rf,
    5
)


## Part 2 __________________________________________


data = loadmat("./datasets/MNIST.mat")

feat = np.array(data["train_fea"])
gnd = np.array(data["train_gnd"]).ravel().astype(int) - 1

feat_test = np.array(data["test_fea"])
gnd_test = np.array(data["test_gnd"]).ravel().astype(int) - 1


param_grid_xgb =     {
        "n_estimators" : [10, 50], 
        "max_depth" : [1, 3],
        "learning_rate" : [1.0],
}

best_params_xgb = cross_validate_model(
    XGBClassifier(num_class=10, objective='multi:softmax'),
    {"feat" : feat, "gnd" : gnd}, 
    param_grid_xgb,
    5
)



