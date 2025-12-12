from fileinput import filename
import random
import math
from scipy.io import loadmat
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import sklearn.linear_model as ln
import sklearn.ensemble as es
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.glassbox import ExplainableBoostingRegressor

from cross_validate import cross_validate_model


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

# ## Part 1 ________________________________________________

# data = loadmat("./datasets/MNISTmini.mat")


# # # assemble data
# feat_8, gnd_8 = MNIST_get_feats_of_label(data, Mm_train, 8)
# feat_9, gnd_9 = MNIST_get_feats_of_label(data, Mm_train, 9)

# feat = np.concatenate( (feat_8, feat_9), axis=0)
# gnd = np.concatenate( (gnd_8, gnd_9) ).ravel()


# feat_test_8, gnd_test_8 = MNIST_get_feats_of_label(data, Mm_test, 8)
# feat_test_9, gnd_test_9 = MNIST_get_feats_of_label(data, Mm_test, 9)

# feat_test = np.concatenate( (feat_test_8, feat_test_9), axis=0)
# gnd_test = np.concatenate( (gnd_test_8, gnd_test_9) ).ravel()


# # #Test logistic regression model

# param_grid_lin = {
#     "max_iter" : [100, 200, 500],
# }

# best_params_lin = cross_validate_model(
#     ln.LogisticRegression(),
#     {"feat" : feat, "gnd" : gnd},
#     param_grid_lin,
#     5
# )


# # #Test Random Forest Model.

# param_grid_rf = {
#     "n_estimators" : [10, 50, 100],
#     "max_depth" : [1, 3, 5],
# }

# best_params_rf = cross_validate_model(
#     es.RandomForestClassifier(),
#     {"feat" : feat, "gnd" : gnd},
#     param_grid_rf,
#     5
# )


# ## Part 2 __________________________________________


# data = loadmat("./datasets/MNIST.mat")

# feat = np.array(data["train_fea"])
# gnd = np.array(data["train_gnd"]).ravel().astype(int) - 1

# feat_test = np.array(data["test_fea"])
# gnd_test = np.array(data["test_gnd"]).ravel().astype(int) - 1


# param_grid_xgb =     {
#         "n_estimators" : [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500], 
#         "max_depth" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         "learning_rate" : [0.05, 0.1, 0.5, 1.0],
# }

# df, best_params_xgb = cross_validate_model(
#     # Set tree_method='hist' for the GPU-accelerated algorithm
#     XGBClassifier(num_class=10, objective='multi:softmax', device='cuda', tree_method='hist'), 
#     {"feat" : feat, "gnd" : gnd}, 
#     param_grid_xgb,
#     5
# )

# df.to_csv("xgb_cross_validation_full_results.csv", index=False)

# print("Successfully exported cross-validation results to xgb_cross_validation_full_results.csv")

# #full train with best params
# best_model = XGBClassifier(
#     num_class=10, 
#     objective='multi:softmax',
#     device='cuda', # Ensure GPU is used for final training
#     tree_method='hist', # Ensure GPU is used for final training
#     n_estimators=best_params_xgb["n_estimators"],
#     max_depth=best_params_xgb["max_depth"],
#     learning_rate=best_params_xgb["learning_rate"],
# )
# best_model.fit(feat, gnd)


# # 1. Feature Importance Export
# importances = best_model.feature_importances_
# n_features = feat.shape[1] 

# # Create meaningful feature names (e.g., Pixel_0, Pixel_1, etc.)
# feature_names = [f"Pixel_{i}" for i in range(n_features)] 

# feature_importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# feature_importance_df.to_csv("xgb_feature_importances.csv", index=False)

# print("Saved feature importances to xgb_feature_importances.csv")

# # 2. Test Set Prediction and Error Analysis Export
# test_predictions = best_model.predict(feat_test)
# test_probabilities = best_model.predict_proba(feat_test)
# num_classes = best_model.get_params()['num_class'] # Should be 10

# test_results_df = pd.DataFrame({
#     'True_Label': gnd_test,
#     'Predicted_Label': test_predictions,
# })

# # Add columns for each class probability 
# prob_cols = [f'Prob_Class_{i}' for i in range(num_classes)]
# test_results_df = pd.concat(
#     [test_results_df, pd.DataFrame(test_probabilities, columns=prob_cols)],
#     axis=1
# )

# test_results_df.to_csv("xgb_test_set_predictions.csv", index=False)

# print("Saved test set predictions and probabilities to xgb_test_set_predictions.csv")

# test_accuracy = best_model.score(feat_test, gnd_test)
# print(f"Test Accuracy with best params: {test_accuracy}")


# # --- Plotting Block Fixes ---
# results_df = df.sort_values(by='mean_test_score', ascending=False)

# # FIX 1: Change 'param_C' to a parameter that exists in your grid
# # Let's plot the learning rate since it's common to analyze
# param_to_plot = 'param_learning_rate' 

# # FIX 2: Group the data since the scores are averaged over the OTHER parameters
# # This is complex, so let's use the simplest, most error-prone parameter: n_estimators
# param_to_plot = 'param_n_estimators'
# plot_df = results_df.groupby(param_to_plot).agg(
#     mean_score=('mean_test_score', 'mean'),
#     std_score=('std_test_score', 'mean')
# ).reset_index()

# scores = plot_df['mean_score']
# stds = plot_df['std_score']
# params = plot_df[param_to_plot]


# plt.figure(figsize=(10, 6))
# # Plot the mean scores
# plt.plot(params, scores, marker='o', linestyle='-', label='Mean Test Score')

# # Plot error bars (optional, using standard deviation)
# plt.errorbar(params, scores, yerr=stds, fmt='o', alpha=0.5, capsize=5)

# plt.xlabel(param_to_plot)
# plt.ylabel('Cross-Validated Params Score')
# plt.title(f'Score vs. {param_to_plot}')
# plt.grid(True)
# plt.legend()
# plt.show()

# # Final Model Export (Relocated to end for logical flow)
# with open("xgb_best_model.pkl", "wb") as f:
#     pickle.dump(best_model, f) 
# print("Saved the final trained model to xgb_best_model.pkl")



# EBM Classifier

# load dataset 












    