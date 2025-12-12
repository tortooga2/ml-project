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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.glassbox import ExplainableBoostingRegressor


def cross_validate_model(model, train, param_grid, batch_size):
    m = model
    grid_search = GridSearchCV(estimator=m, param_grid=param_grid, cv=batch_size, verbose=3)
    grid_search.fit(train["feat"], train["gnd"])
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"[BEST] Params: {best_params} -> Average Accuracy: {best_score}")
    df = pd.DataFrame(grid_search.cv_results_)
    
    df.to_csv("cross_validation_full_results.csv", index=False)
    print("Successfully exported cross-validation results to cross_validation_full_results.csv")
    
    
    return df, best_params