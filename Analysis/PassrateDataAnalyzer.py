import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
import random
from copy import deepcopy
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate


def main(chosen_AIs, chosen_levels, features):
    data = pd.read_excel('Analysis/TapLogicData.xlsx')
    userPassrateData = pd.read_excel('Analysis/PassrateData.xlsx')

    level_num = len(chosen_levels)
    total_feature_num = len(chosen_AIs) * len(features)

    parsed_data = dict()
    for AI in chosen_AIs:
        for level in chosen_levels:
            parsed_data[(AI, level)] = []

    provided_step = dict()
    for level in chosen_levels:
        provided_step[level] = 0

    passrate_data = dict()
    for level in chosen_levels:
        passrate_data[level] = 0.0

    for row in range(data.iloc[:, 0].size):
        level = data.iloc[row, 0]
        AI = data.iloc[row, 2]

        if (AI, level) in parsed_data:
            parsed_data[(AI, level)].append(data.iloc[row, 1])

        if AI == "通关值":
            provided_step[level] = data.iloc[row, 1]

    for row in range(userPassrateData.iloc[:, 0].size):
        level = userPassrateData.iloc[row, 0]
        AI = userPassrateData.iloc[row, 3]

        if AI == "PLAYER":
            passrate_data[level] = userPassrateData.iloc[row, 1] / 100.0

    data_x = np.zeros([level_num, total_feature_num])
    data_y = np.zeros([level_num, 1])

    level_idx = []

    curr_feature_idx = 0
    for AI in chosen_AIs:
        for feature in features:
            for idx, level in enumerate(chosen_levels):
                if feature == "Passrate":
                    data_x[idx, curr_feature_idx] = (np.array(parsed_data[(AI, level)]) <=
                                                            provided_step[level]).astype(np.float32).mean()
                elif feature == "AvegUsedStepPrecentile":
                    data_x[idx, curr_feature_idx] = \
                        1.0 * np.array(parsed_data[(AI, level)]).mean() / provided_step[level]
                elif feature == "NormalizedStd":
                    data_x[idx, curr_feature_idx] = \
                        1.0 * np.array(parsed_data[(AI, level)]).std() / provided_step[level]
                elif feature == "Emplitude":
                    steps = np.array(parsed_data[(AI, level)])
                    percentiles = np.percentile(steps, [25, 75])
                    data_x[idx, curr_feature_idx] = \
                        (percentiles[1] - percentiles[0]) / provided_step[level]
                else:
                    raise NotImplementedError()

                if curr_feature_idx == 0:
                    data_y[idx, 0] = passrate_data[level]
                    level_idx.append(level)

            curr_feature_idx += 1

    data_y = np.reshape(data_y, (-1,))

    best_model, best_mae = GBR_grid_test(deepcopy(data_x), deepcopy(data_y), level_num)

    # best_model = LinearRegression()
    # best_model = RidgeCV()

    best_model.fit(data_x[:27, :], data_y[:27])
    y_pred = best_model.predict(data_x)
    print(level_idx)
    print(y_pred)
    print(data_y)
    print(data_x[:, 0])


def GBR_grid_test(data_x, data_y, level_num):
    grid_params = {
        "n_estimators": [2, 4, 8],
        "learning_rate": [0.02, 0.1, 0.01],
        "min_samples_split": [2],
        "min_samples_leaf": [3],
        "max_depth": [1, 2, 3, 5],
        "subsample": [0.8]
    }

    best_model = None
    best_mae = 1.0
    best_mae_train = 1.0

    for _ in range(1000):
        kwargs = dict()
        for key in grid_params:
            kwargs[key] = random.choice(grid_params[key])

        model, mae, mae_train = GBR_test(data_x, data_y, level_num, kwargs)

        if mae < best_mae:
            best_mae = mae
            best_mae_train = mae_train
            best_model = model

        print(best_mae, best_mae_train)

    print(best_model)
    print(best_mae)

    return best_model, best_mae


def GBR_test(data_x, data_y, level_num, kwargs):
    mae = []
    mae_train = []

    for _ in range(100):
        m = np.random.permutation(level_num)
        data_x = data_x[m, :]
        data_y = data_y[m]

        training_data_num = int(level_num * 0.8)

        training_data_x = data_x[:training_data_num, :]
        training_data_y = data_y[:training_data_num]

        test_data_x = data_x[training_data_num:, :]
        test_data_y = data_y[training_data_num:]

        gbt = GradientBoostingRegressor(
            max_features = 'sqrt',
            random_state = 10,
            **kwargs
        )

        '''gbt = GradientBoostingRegressor(
            n_estimators = 4,
            learning_rate = 0.1,
            min_samples_leaf = 3,
            max_depth = 8,
            max_features ='sqrt',
            subsample = 0.8,
            random_state = 10
        )'''

        gbt.fit(training_data_x, training_data_y)

        y_pred = gbt.predict(test_data_x)
        mae.append(np.abs((y_pred - test_data_y)).mean())

        y_pred = gbt.predict(training_data_x)
        mae_train.append(np.abs((y_pred - training_data_y)).mean())

    return gbt, np.array(mae).mean(), np.array(mae_train).mean()


if __name__ == "__main__":
    chosen_AIs = ["MCTS200", "MCTS500", "RL"]
    chosen_levels = list(range(81, 90)) + \
                    list(range(91, 100)) + \
                    list(range(101, 110)) + \
                    list(range(111, 120))

    features = ["Passrate", "AvegUsedStepPrecentile", "NormalizedStd", "Emplitude"]
    # features = ["Passrate"]

    main(chosen_AIs, chosen_levels, features)
