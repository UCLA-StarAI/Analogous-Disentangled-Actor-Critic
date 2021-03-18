import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Analysis.CollectData import collect_user_classified_data
import pickle
import xlwt

import xgboost as xgb

from sklearn.linear_model import RidgeCV, LassoCV

from utils.dbutil import insertAdaptationPlayer

import time
from datetime import datetime
import pytz

import matplotlib.pyplot as plt


def main():
    # chosen_AIs = ["MCTS200", "MCTS500" "RL"]
    chosen_AIs = ["MCTS10", "MCTS200"]
    chosen_levels = list(range(31, 40)) + \
                    list(range(41, 50)) + \
                    list(range(51, 54)) + list(range(55, 60)) + \
                    list(range(61, 68)) + list(range(69, 70)) + \
                    list(range(71, 78)) + \
                    list(range(81, 90)) + \
                    list(range(91, 100)) + \
                    list(range(101, 110)) + \
                    list(range(111, 120)) + \
                    list(range(121, 130)) + \
                    list(range(131, 140)) + \
                    list(range(141, 143)) + \
                    list(range(144, 145))

    y_feature = "Passrate"

    # features = ["Passrate", "AvegUsedStepPrecentile", "NormalizedStd", "Emplitude"]
    features = ["Passrate", "AvegUsedStepPrecentile"]

    level_idxs, data_x, data_y = collect_user_classified_data(
        chosen_AIs, chosen_levels, features, y_feature, "6_11"
    )
    level_idxs = np.array(level_idxs)
    data_y = np.reshape(data_y, (-1,))

    m = np.random.permutation(len(level_idxs))

    level_idxs = level_idxs[m]
    data_x = data_x[m, :]
    data_y = data_y[m]

    split_idx = int(len(level_idxs) * 0.6)

    train_x = data_x[:split_idx, :]
    train_y = data_y[:split_idx]

    test_x = data_x[split_idx:, :]
    test_y = data_y[split_idx:]

    best_model = None
    best_mae = 100.0

    for n_est in range(5, 100, 5):
        model = xgb.XGBRegressor(n_estimators = n_est)
        model.fit(train_x, train_y)

        y = model.predict(test_x)
        mae = np.abs(test_y - y).mean()
        if mae < best_mae:
            best_model = pickle.dumps(model)
            best_mae = mae

    best_model = pickle.loads(best_model)
    print(best_mae)

    chosen_levels = list(range(31, 40)) + \
                    list(range(41, 50)) + \
                    list(range(51, 60)) + \
                    list(range(61, 70)) + \
                    list(range(71, 80)) + \
                    list(range(81, 90)) + \
                    list(range(91, 100))

    level_idxs, data_x, data_y = collect_user_classified_data(
        chosen_AIs, chosen_levels, features, y_feature, "7_4"
    )
    data_y = np.reshape(data_y, (-1,))

    y = model.predict(data_x)
    mae = np.abs(data_y - y).mean()
    print(mae)

if __name__ == "__main__":
    main()

