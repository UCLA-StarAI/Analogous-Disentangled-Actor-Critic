import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Analysis.CollectData import collect_train_data, collect_test_data
import pickle
import xlwt

import xgboost as xgb

from sklearn.linear_model import RidgeCV, LassoCV

from utils.dbutil import insertAdaptationPlayer

import time
from datetime import datetime
import pytz

import matplotlib.pyplot as plt


class SeparateRegionRegressor(nn.Module):
    def __init__(self, feature_size, separate_bins = 4):
        super(SeparateRegionRegressor, self).__init__()

        self.feature_size = feature_size
        self.separate_bins = separate_bins

        self.softmax_classifier = nn.Sequential(
            nn.Linear(feature_size, separate_bins),
            nn.Softmax(dim = 1)
        )

        self.linear_reg_model = nn.Linear(feature_size, separate_bins)

        self.optimizer = optim.Adam(self.parameters(), lr = 1e-4, weight_decay = 1e-5)

        self.mseLoss = nn.MSELoss()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return (self.softmax_classifier(x) * self.linear_reg_model(x)).sum(dim = 1, keepdim = True)

    def train_step(self, train_x, train_y):
        y = self.forward(train_x)

        # per_item_loss = self.mseLoss(y, train_y)

        per_item_loss = (y - train_y).pow(2)
        with torch.no_grad():
            loss_perc_num = np.percentile(per_item_loss.detach().cpu().numpy(), [90])[0]
            loss_perc_factor = loss_perc_num / per_item_loss.clamp(min = loss_perc_num)

        per_item_loss = (per_item_loss * loss_perc_factor)
        # per_item_loss = loss_perc_num.clamp(max = loss_perc_num).mean()

        per_item_loss = per_item_loss.clamp(min = 0.03 * 0.03)

        per_item_loss = per_item_loss.mean()

        self.optimizer.zero_grad()

        per_item_loss.backward()

        self.optimizer.step()

    def test_step(self, test_x, test_y):
        y = self.forward(test_x)

        per_item_loss = torch.abs(y - test_y).mean()

        # print(y.view(-1))
        # print(test_y.view(-1))
        return per_item_loss.detach().cpu().numpy()

    def plot_figure(self, level_idx, data_x, data_y, test_x, test_y):
        y = self.forward(data_x)

        y[y < 0.0] = 0.0

        y = y.detach().cpu().numpy()
        data_y = data_y.detach().cpu().numpy()

        yy = self.forward(test_x)
        yy[yy < 0.0] = 0.0
        yy = yy.detach().cpu().numpy()
        data_yy = test_y.detach().cpu().numpy()

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(level_idx, data_y)
        plt.plot(level_idx, y)

        plt.legend(["user", "AI_fit"])

        plt.subplot(2, 1, 2)
        plt.hist(np.abs(yy - data_yy), bins = 40)

        print((yy - data_yy).mean(), np.abs(yy - data_yy).mean())

        print(level_idx)
        print(np.reshape(data_y, (-1,)))
        print(np.reshape(y, (-1,)))

        print((np.abs(yy - data_yy) < 0.05).astype(np.float32).mean())
        print((np.abs(yy - data_yy) < 0.1).astype(np.float32).mean())

        plt.show()


def main():
    # chosen_AIs = ["MCTS200", "MCTS500" "RL"]
    chosen_AIs = ["MCTS10"]
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
    '''chosen_levels = list(range(31, 40)) + \
                    list(range(41, 50)) + \
                    list(range(51, 60)) + \
                    list(range(61, 70)) + \
                    list(range(71, 80)) + \
                    list(range(81, 90)) + \
                    list(range(91, 100))'''


    y_feature = "Passrate"
    # y_feature = "UseExtraStepRate"
    # y_feature = "UsePropRate"
    # y_feature = "UserStayRate"
    # y_feature = "Prop_4"

    # features = ["Passrate", "AvegUsedStepPrecentile", "NormalizedStd", "Emplitude"]
    features = ["Passrate", "AvegUsedStepPrecentile"]

    level_idxs, data_x, data_y = collect_train_data(chosen_AIs, chosen_levels, features, y_feature)
    # data_x = torch.tensor(data_x, dtype = torch.float32)
    # data_y = torch.tensor(data_y, dtype = torch.float32)
    data_y = np.reshape(data_y, (-1,))

    # exit()

    # m = np.random.permutation(len(chosen_AIs))

    train_x = data_x[:55, :]
    train_y = data_y[:55]

    test_x = data_x[55:, :]
    test_y = data_y[55:]

    # print(train_x)
    # print(train_y)

    '''model = SeparateRegionRegressor(len(features) * len(chosen_AIs), separate_bins = 1)

    best_model = None
    best_test_error = 100.0

    for _ in range(20000):
        model.train_step(train_x, train_y)

        error = model.test_step(test_x, test_y)
        print(error)

        if error < best_test_error:
            best_model = pickle.dumps(model)
            best_test_error = error

    model = pickle.loads(best_model)'''

    # model = RidgeCV()

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

    y = best_model.predict(data_x)
    error = np.abs(data_y - y)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(level_idxs, y)
    plt.plot(level_idxs, data_y)

    plt.subplot(2, 1, 2)
    plt.plot(level_idxs, np.reshape(error, (-1,)))

    plt.show()

    chosen_levels = list(range(31, 40)) + \
                    list(range(41, 50)) + \
                    list(range(51, 60)) + \
                    list(range(61, 70)) + \
                    list(range(71, 80)) + \
                    list(range(81, 90)) + \
                    list(range(91, 100))

    test_level_idxs, test_data_x = collect_test_data(chosen_AIs, chosen_levels, features)

    test_data_y = best_model.predict(test_data_x)

    print(test_data_y)

    plt.figure()
    plt.plot(test_level_idxs, test_data_y)
    plt.show()

    workbook = xlwt.Workbook(encoding = 'utf-8')
    worksheet = workbook.add_sheet('My Worksheet')

    worksheet.write(0, 0, label = '关卡')
    worksheet.write(0, 1, label = '预测通关率')

    for i in range(test_data_y.size):
        worksheet.write(i + 1, 0, label = test_level_idxs[i])
        worksheet.write(i + 1, 1, label = int(test_data_y[i] * 100))

    workbook.save('./output.xls')

    # print(best_test_error)
    # model.plot_figure(level_idxs, data_x, data_y, test_x, test_y)

    '''chosen_levels = list(range(31, 40)) + \
                    list(range(41, 50)) + \
                    list(range(51, 60)) + \
                    list(range(61, 70)) + \
                    list(range(71, 80)) + \
                    list(range(81, 90)) + \
                    list(range(91, 100))

    test_level_idxs, test_data_x = collect_test_data(chosen_AIs, chosen_levels, features)

    test_data_y = model(torch.tensor(test_data_x, dtype = torch.float32)).detach().cpu().numpy()

    save_dict = dict()
    date_time = datetime.fromtimestamp(int(time.time()),
                                       pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    save_dict['datetime'] = date_time
    save_dict['train_count'] = 1
    save_dict['level_version'] = "7.4"
    save_dict['code_version'] = "1.0.0"
    save_dict['type'] = "AI"

    workbook = xlwt.Workbook(encoding = 'utf-8')
    worksheet = workbook.add_sheet('My Worksheet')

    worksheet.write(0, 0, label = '关卡')
    worksheet.write(0, 1, label = '预测通关率')

    for i in range(test_data_y.size):
        save_dict['pass_rate'] = int(test_data_y[i, 0] * 100)
        save_dict['level'] = test_level_idxs[i]

        print(save_dict['level'], save_dict['pass_rate'])

        # insertAdaptationPlayer(save_dict)

        worksheet.write(i + 1, 0, label = save_dict['level'])
        worksheet.write(i + 1, 1, label = save_dict['pass_rate'])

    workbook.save('./output.xls')

    plt.figure()
    plt.plot(np.array(test_level_idxs), np.reshape(test_data_y, (-1, )))
    plt.show()'''


if __name__ == "__main__":
    main()

