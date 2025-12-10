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
    return pd.DataFrame(grid_search.cv_results_), best_params


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

## Part 3 __________________________________________

#Load dataset


def load_csv(filename, labels, features, feature_conv = None):
    feat = []
    gnd = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames

        if headers:
            reader.fieldnames = [header.strip() for header in headers]
        
        data = list(reader)
        for row in data:
            
            labels_list = []
            for l in labels:
                try:
                    labels_list.append(row[l])
                except KeyError:
                    print(f"Label {l} not found in data.")
                    print(row)
                    continue
            if len(labels_list) == 1:
                label = labels_list[0]
            else:
                label = labels_list
            
            features_list = []
            for feature in features:
                value = row[feature]
                if feature_conv and feature in feature_conv["feat"]:
                    func_idx = feature_conv["feat"].index(feature)
                    func = feature_conv["func"][func_idx]
                    value = func(value)
                else:
                    try:
                        value = float(value)
                    except:
                        pass
                if isinstance(value, tuple) or isinstance(value, list):
                    for v in value:
                        features_list.append(v)
                else:
                    features_list.append(value)
            gnd.append(label)
            feat.append(features_list)
    feat = np.array(feat)
    gnd = np.array(gnd)
    return feat, gnd




# Define feature names
labal = ["Order_Demand"]
feature_names = [
    "Product_Category", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Petrol_price", "Date"
]

feature_types = [
    'nominal', # Product_Category
    'continuous', # Open (Likely binary, but can be treated as continuous/binned)
    'continuous', # Promo (Likely binary, but can be treated as continuous/binned)
    'nominal', # StateHoliday
    'continuous', # SchoolHoliday (Likely binary, but can be treated as continuous/binned)
    'continuous'  # Petrol_price
]

def convert_product_category(value : str):
    mapping = {
        'Category_005' : 0, 
        'Category_028' : 1, 
        'Category_003' : 2, 
        'Category_021' : 3, 
        'Category_013' : 4, 
        'Category_033' : 5, 
        'Category_007' : 6, 
        'Category_026' : 7, 
        'Category_010' : 8, 
        'Category_031' : 9, 
        'Category_019' : 10, 
        'Category_023' : 11, 
        'Category_008' : 12, 
        'Category_009' : 13, 
        'Category_030' : 14, 
        'Category_020' : 15, 
        'Category_006' : 16, 
        'Category_011' : 17, 
        'Category_016' : 18, 
        'Category_024' : 19, 
        'Category_022' : 20, 
        'Category_014' : 21, 
        'Category_012' : 22, 
        'Category_015' : 23, 
        'Category_032' : 24, 
        'Category_018' : 25, 
        'Category_017' : 26, 
        'Category_029' : 27, 
        'Category_001' : 28, 
        'Category_027' : 29,
    }
    return mapping[value]

def convert_state_holiday(value : str):
    mapping = {
        '0' : 0,
        'a' : 1,
        'b' : 2,
        'c' : 3,
    }
    return mapping[value]

def convert_date(value : str):
    import datetime
    import time
    dt = datetime.datetime.strptime(value, '%m/%d/%Y')
    # 2. Extract the cyclical components as integers
    day_of_week = dt.weekday()  # Monday is 0 and Sunday is 6
    month = dt.month
    day_of_month = dt.day
    # 3. Normalize the cyclical components to a range [0, 1]
    day_of_week_normalized = day_of_week / 6.0  # Normalize to [0, 1]
    month_normalized = (month - 1) / 11.0  # Normalize to [0, 1]
    day_of_month_normalized = (day_of_month - 1) / 30.0  # Normalize to [0, 1]
    return [day_of_week_normalized, month_normalized, day_of_month_normalized]

rg_feat, rg_gnd = load_csv("./datasets/Retail_Dataset2.csv", labal, feature_names, {"feat" : ["Product_Category", "StateHoliday", "Date"], "func" : [convert_product_category, convert_state_holiday, convert_date]})        


index_list = list(range(rg_feat.shape[0]))
random.shuffle(index_list)
split_idx = math.floor(0.9 * rg_feat.shape[0])
train_idx = index_list[:split_idx]
print(f"Train size: {len(train_idx)}")
test_idx = index_list[split_idx:]

rg_feat_train = rg_feat[train_idx]

scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(rg_feat_train)
rg_feat_train = X_train_normalized



rg_gnd_train = rg_gnd[train_idx]
rg_gnd_train = np.log1p(rg_gnd_train)



rg_feat_test = rg_feat[test_idx]
rg_gnd_test = rg_gnd[test_idx]

rg_feat_test = scaler.transform(rg_feat_test)

print(f"label: {rg_gnd_train[:10]}, features: {rg_feat_train[:10]}")






# # Explainable Boosting Classifier

param_grid_ebc =     {
    # "learning_rate" : [0.01],
    # "max_bins" : [128],
}

best_params_ebc = cross_validate_model(
    ExplainableBoostingRegressor(),
    {"feat" : rg_feat_train, "gnd" : rg_gnd_train}, 
    param_grid_ebc,
    5
)



# EBM Classifier

# load dataset 






# Load only the numerical sensor data (assuming columns 4 onward are channels)
sensor_data = np.loadtxt(
    './datasets/CoST.csv',
    delimiter=',',
    skiprows=1,           # Skip the header row
)
print(f"Loaded shape: {sensor_data.shape}")



feat = sensor_data[:, 4:]  # All rows, columns from index 4 onward
gnd_variant = sensor_data[:, 1].astype(int)  # All rows, second column
gnd_gesture = sensor_data[:, 2].astype(int)  # All rows, third column




frame_column = sensor_data[:, 3].astype(int)  # All rows, first column

max_frame = np.max(frame_column)


sequence_change = []
for i, frame in enumerate(frame_column):
    if frame == 1:
        sequence_change.append(i)
    


tensor_feat = np.split(feat, sequence_change[1:])
gnd_gesture = gnd_gesture[sequence_change]
gnd_variant = gnd_variant[sequence_change]

print(f"Number of sequences: {len(tensor_feat)}")

final_3d_array = np.zeros((len(tensor_feat), max_frame, 64), dtype=np.float32)

for i, seq in enumerate(tensor_feat):
    seq_len = seq.shape[0]
    final_3d_array[i, :seq_len, :] = seq




feat = final_3d_array
gnd = np.array(gnd_gesture - 1)



R, T_max, C_channels = feat.shape

feat_mean = np.mean(feat, axis=1)

feat_std = np.std(feat, axis=1)

feat_max = np.max(feat, axis=1)

feat_diff = np.diff(feat, axis=1)

feat_mean_abs_diff = np.mean(np.abs(feat_diff), axis=1)

feat_accel = np.diff(feat, axis=1, n=2)


feat_std_accel = np.std(feat_accel, axis=1)


mid_point = T_max // 2


feat_mean_half1 = np.mean(feat[:, :mid_point, :], axis=1)

feat_mean_half2 = np.mean(feat[:, mid_point:, :], axis=1)

feat_max_time = np.argmax(feat, axis=1)

signs = np.sign(feat)

# 2. Difference of the sign: non-zero where the sign changes
sign_changes = np.diff(signs, axis=1) != 0

# 3. Sum the sign changes and normalize by the number of frames
# Resulting shape: (R, C_channels)
feat_zcr = np.sum(sign_changes, axis=1) / (T_max - 1)

# You can normalize this feature by dividing by T_max to get a value between 0 and 1
feat_max_time_norm = feat_max_time / T_max

feat_energy = np.sum(np.square(feat), axis=1)



X_features = np.hstack([
    feat_mean,          # 64 features (Overall position)
    feat_std,           # 64 features (Overall volatility)
    feat_max,           # 64 features (Peak position)
    feat_mean_abs_diff, # 64 features (Average speed)
    feat_std_accel,     # 64 features (Jerkiness/consistency)
    feat_mean_half1,    # 64 features (Starting position/state)
    feat_mean_half2,    # 64 features (Ending position/state)
    feat_zcr,            # New 64 features
    feat_max_time_norm,  # New 64 features
    feat_energy          # New 64 features
])

# print(f"Final feature array shape: {X_features.shape}")

X_index_list = list(range(X_features.shape[0]))
random.shuffle(X_index_list)


min_max_scaler = RobustScaler()
X_features = min_max_scaler.fit_transform(X_features)

X_features = X_features[X_index_list]
gnd = gnd[X_index_list]







df, best_params_ebc = cross_validate_model(
    ExplainableBoostingClassifier(),
    {"feat" : X_features, "gnd" : gnd}, 
    param_grid_ebc,
    5
)






    