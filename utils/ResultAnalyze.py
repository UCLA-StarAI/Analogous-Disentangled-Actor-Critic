import numpy as np
import xlrd
import xlwt
import scipy.io as sio
import os
from shutil import copyfile


def analyze_result():
    workbook = xlwt.Workbook(encoding = 'utf-8')
    worksheet = workbook.add_sheet('强化学习与蒙特卡洛对比')

    worksheet.write(0, 0, label = '关卡')
    worksheet.write(0, 1, label = '强化学习均值')
    worksheet.write(0, 2, label = '强化学习标准差')
    worksheet.write(0, 3, label = '蒙特卡洛均值')
    worksheet.write(0, 4, label = '蒙特卡洛最小值')
    worksheet.write(0, 5, label = '蒙特卡洛最大值')

    for level_idx in range(1, 151):
        RL_path = os.path.join('save/HappyElimination/A2CSD_PolicyMimic', str(level_idx),
                               'action_mode_0/eval_rewards.mat')

        if not os.path.exists(RL_path):
            continue

        data = sio.loadmat(RL_path)
        RL_data = np.reshape(data["reward"][:, -5:], (-1,)).astype(np.float32)
        RL_mean = np.mean(RL_data)
        RL_std = np.std(RL_data)

        MCTS_path = os.path.join('save/MCTSevalResult', str(level_idx) + ".txt")

        if not os.path.exists(MCTS_path):
            continue

        with open(MCTS_path, 'r') as f:
            data = f.readline().split(' ')[:-1]

        MCTS_data = np.array([int(item) for item in data], dtype = np.float32)
        MCTS_mean = np.mean(MCTS_data)
        MCTS_min = np.min(MCTS_data)
        MCTS_max = np.max(MCTS_data)

        worksheet.write(level_idx, 0, label = level_idx)
        worksheet.write(level_idx, 1, label = float(RL_mean))
        worksheet.write(level_idx, 2, label = float(RL_std))
        worksheet.write(level_idx, 3, label = float(MCTS_mean))
        worksheet.write(level_idx, 4, label = float(MCTS_min))
        worksheet.write(level_idx, 5, label = float(MCTS_max))

    workbook.save('Excel_test.xls')


def analyze_passrate():
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('强化学习与蒙特卡洛对比')

    worksheet.write(0, 0, label = '关卡')
    worksheet.write(0, 1, label = '强化学习通关率')
    worksheet.write(0, 2, label = '蒙特卡洛通关率')
    worksheet.write(0, 3, label = '蒙特卡洛次数')

    level_data = xlrd.open_workbook('给定步数.xls').sheets()[0]

    for level_idx in range(1, 151):
        RL_path = os.path.join('save/HappyElimination/A2CSD_PolicyMimic', str(level_idx),
                               'action_mode_0/eval_rewards.mat')

        if not os.path.exists(RL_path):
            continue

        data = sio.loadmat(RL_path)
        RL_data = np.reshape(data["reward"][:, -5:], (-1,)).astype(np.float32)
        given_steps = level_data.cell(level_idx, 1).value
        RL_passrate = (RL_data <= given_steps).astype(np.float32).mean()

        MCTS_path = os.path.join('save/MCTSevalResult', str(level_idx) + ".txt")

        if not os.path.exists(MCTS_path):
            continue

        with open(MCTS_path, 'r') as f:
            data = f.readline().split(' ')[:-1]

        MCTS_data = np.array([int(item) for item in data], dtype = np.float32)
        MCTS_passrate = (MCTS_data <= given_steps).astype(np.float32).mean()
        MCTS_num = MCTS_data.size

        worksheet.write(level_idx, 0, label = level_idx)
        worksheet.write(level_idx, 1, label = float(RL_passrate))
        worksheet.write(level_idx, 2, label = float(MCTS_passrate))
        worksheet.write(level_idx, 3, label = float(MCTS_num))

    workbook.save('Passrate.xls')


def analyze_result2():
    workbook = xlwt.Workbook(encoding = 'utf-8')
    worksheet = workbook.add_sheet('Sheet 1')

    worksheet.write(0, 0, label = '关卡')
    worksheet.write(0, 1, label = '6.11 RL mean')
    worksheet.write(0, 2, label = '7.4 RL mean')
    worksheet.write(0, 3, label = '6.11 RL 25%')
    worksheet.write(0, 4, label = '7.4 RL 25%')
    worksheet.write(0, 5, label = '6.11 RL 50%')
    worksheet.write(0, 6, label = '7.4 RL 50%')
    worksheet.write(0, 7, label = '6.11 RL 75%')
    worksheet.write(0, 8, label = '7.4 RL 75%')
    worksheet.write(0, 9, label = '6.11 MCTS10 mean')
    worksheet.write(0, 10, label = '7.4 MCTS10 mean')

    for level_idx in range(71, 79):
        RL_path = os.path.join('save/EvalResult/RL', str(level_idx) + '.txt')
        MCTS_path = os.path.join('save/EvalResult/MCTS10', str(level_idx) + '.txt')

        if not os.path.exists(RL_path) or not os.path.exists(MCTS_path):
            continue

        with open(RL_path, 'r') as f:
            data = f.readline().split(' ')[:-1]

        RL_data = np.array([int(item) for item in data], dtype = np.float32)
        RL_mean = np.mean(RL_data)
        RL_25 = np.percentile(RL_data, 25)
        RL_25 = np.min(RL_data)

        '''worksheet.write(level_idx, 0, label = level_idx)
        worksheet.write(level_idx, 1, label = float(RL_mean))
        worksheet.write(level_idx, 2, label = float(RL_std))
        worksheet.write(level_idx, 3, label = float(MCTS_mean))
        worksheet.write(level_idx, 4, label = float(MCTS_min))
        worksheet.write(level_idx, 5, label = float(MCTS_max))'''

    workbook.save('Excel_test.xls')


def move_training_curves():
    if not os.path.exists("save/TrainingCurves"):
        os.mkdir("save/TrainingCurves")

    for level_idx in range(1, 151):
        folder_path = os.path.join("save/HappyElimination/A2CSD_PolicyMimic/", str(level_idx),
                                   "action_mode_0", "eval_rewards.png")

        if not os.path.exists(folder_path):
            continue

        copyfile(folder_path, "save/TrainingCurves/" + str(level_idx) + ".png")


if __name__ == "__main__":
    analyze_result()
    # analyze_passrate()
    # move_training_curves()
