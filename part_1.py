import random
import math
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

import sklearn.linear_model as ln
import sklearn.ensemble as es
from xgboost import XGBClassifier
import xgboost


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

# #Cross Validation
# #1. split data set into multtiple batches. since we have roughly 10k items lets do 10 batches of 1k

# #since feature and label are two seperate arrays, and we want thier relative position to match, we need to use a index table and do any shuffling or batching like that. 
# 

def cross_validate_model(train, model, param_grid, batch_size):
    idx_array = np.arange(len(train['feat']))
    results = []

    
    match model:
        case "logit":

            np.random.shuffle(idx_array)
            for max_iter in param_grid["max_iter"]:

                n_batches = np.array_split(idx_array, batch_size) 

                batch_scores = []

                        
                for i in range(batch_size):
                            
                            
                    val_idx = n_batches[i]
                            

                    train_idx = np.concatenate([n_batches[j] for j in range(batch_size) if j != i])
                            

                    train_feat = train["feat"][train_idx]
                    train_gnd = train["gnd"][train_idx]
                            
                    val_feat = train["feat"][val_idx]
                    val_gnd = train["gnd"][val_idx]

                    m = ln.LogisticRegression(max_iter=max_iter)

                    m.fit(train_feat, train_gnd)
                    y_pred = m.predict(val_feat)

                    acc = np.mean(y_pred == val_gnd)
                    batch_scores.append(acc)

                acc = np.mean(batch_scores)
                results.append((acc, {"max_iter" : max_iter}))
                print(f"with max iteration set to: {max_iter}, we acheived an acc of: {acc}")
        case "rnd_forest":
            np.random.shuffle(idx_array)
            for estimators in param_grid["n_estimators"]:
                
                n_batches = np.array_split(idx_array, batch_size) 

                batch_scores = []

                        
                for i in range(batch_size):
                            
                            
                    val_idx = n_batches[i]
                            

                    train_idx = np.concatenate([n_batches[j] for j in range(batch_size) if j != i])
                            

                    train_feat = train["feat"][train_idx]
                    train_gnd = train["gnd"][train_idx]
                            
                    val_feat = train["feat"][val_idx]
                    val_gnd = train["gnd"][val_idx]

                    m = es.RandomForestClassifier(n_estimators=estimators)

                    m.fit(train_feat, train_gnd)
                    y_pred = m.predict(val_feat)

                    acc = np.mean(y_pred == val_gnd)
                    batch_scores.append(acc)
                        
                avg_acc = np.mean(batch_scores)
                results.append((avg_acc, {"estimators" : estimators}))
                print(f"with estimators set to: {estimators}, we acheived an acc of: {acc}")

        case "xgboost":
            np.random.shuffle(idx_array)
            for estimators in param_grid["n_estimators"]:
                for max_depth in param_grid["max_depth"]:
                    for learning_rate in param_grid["learning_rate"]:
                        n_batches = np.array_split(idx_array, batch_size) 

                        batch_scores = []

                        
                        for i in range(batch_size):
                            
                            
                            val_idx = n_batches[i]
                            

                            train_idx = np.concatenate([n_batches[j] for j in range(batch_size) if j != i])
                            

                            train_feat = train["feat"][train_idx]
                            train_gnd = train["gnd"][train_idx]
                            
                            val_feat = train["feat"][val_idx]
                            val_gnd = train["gnd"][val_idx]


                            m = XGBClassifier(
                                n_estimators=estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                objective="multi:softmax",
                                num_class=param_grid["num_class"]
                            )

                            m.fit(train_feat, train_gnd)
                            y_pred = m.predict(val_feat)

                            acc = np.mean(y_pred == val_gnd)
                            batch_scores.append(acc)
                        
                        acc = np.mean(batch_scores)
                        results.append((acc, {"estimators": estimators, "max_depth" : max_depth, "learning_rate" : learning_rate}))
                        print(f"Params: estimators: {estimators}, max_depth: {max_depth}, learning rate: {learning_rate} -> Average Accuracy: {acc}")


    match model:
        case "logit" : 
            results.sort(key=lambda t: t[0])
            (acc, params) = results.pop()
            print(f"{params["max_iter"]} iterations was found to be the best with an accuracy of {acc}")
            return params
        case "rnd_forest":
            results.sort(key=lambda t: t[0])
            (acc, params) = results.pop()
            print(f"{params["estimators"]} estimators was found to be the best with an accuracy of {acc}")
            return params
        case 'xgboost':
            results.sort(key=lambda t: t[0])
            (acc, params) = results.pop()
            print(f'[BEST] estimators : {params["estimators"]}; max_depth: {params["max_depth"]}; learning_rate: {params["learning_rate"]} -> acc of: {acc}')
            return params



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






# #Compute results / score

# model_rf = es.RandomForestClassifier(n_estimators= estimators)
# model_rf.fit(feat, gnd)
# y_rf_pred = model_rf.predict(feat_test)
# tp_rf = np.mean(y_rf_pred == gnd_test)



# print(f"Logistic Regression prediction accuracy: {tp_lin}\nRandom Forest prediction accuracy: {tp_rf}")



## Part 2 __________________________________________


data = loadmat("./datasets/MNIST.mat")

feat = np.array(data["train_fea"])
gnd = np.array(data["train_gnd"]).ravel().astype(int) - 1

feat_test = np.array(data["test_fea"])
gnd_test = np.array(data["test_gnd"]).ravel().astype(int) - 1


cross_validate_model(
        {"feat" : feat, "gnd" : gnd}, 
        {"feat" : feat_test, "gnd" : gnd_test}, 
        "xgboost",
        {
            "n_estimators" : [10, 30, 40, 50], 
            "max_depth" : [1, 2, 3],
            "learning_rate" : [0.1, 0.6, 1.0],
            "num_class" : 10
        },
        5
    )



