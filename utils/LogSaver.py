import numpy as np
import matplotlib.pyplot as plt
from scipy.io import (savemat, loadmat)
import os

from utils.MovingAvegCalculator import MovingAvegCalculator


class LogSaver():
    def __init__(self, save_folder_path, args):
        self.save_folder_path = save_folder_path
        if False and self.args.env_name == "HappyElimination":
            self.save_file_name = os.path.join(save_folder_path, "eval_rewards_" + str(args.level_idx) + ".mat")
            self.save_eval_file_name = os.path.join(save_folder_path, "eval_results_" + str(args.level_idx) + ".mat")
            self.save_fig_name = os.path.join(save_folder_path, "eval_rewards_" + str(args.level_idx) + ".png")
        else:
            self.save_file_name = os.path.join(save_folder_path, "eval_rewards.mat")
            self.save_eval_file_name = os.path.join(save_folder_path, "eval_results.mat")
            self.save_fig_name = os.path.join(save_folder_path, "eval_rewards.png")

        if os.path.exists(self.save_file_name) and (not args.do_not_load):
            self.results = loadmat(self.save_file_name)["reward"]
        else:
            self.results = np.array([]).reshape((args.evaluate_num_episodes, 0))

        if self.results.shape[0] == 0:
            self.results = np.array([]).reshape((args.evaluate_num_episodes, 0))

        self.interval = args.evaluate_interval

        self.args = args

        self.movingAvegCalculator = None
        self.platform_alarm = False
        self.last_mean = float("inf")
        self.last_std = 0

        self.steps = 0

    def record(self, result):
        result = np.array(result).reshape(-1, 1)
        self.results = np.hstack([self.results, result])

        self.steps += 1

        if self.movingAvegCalculator is not None:
            mean, std = self.movingAvegCalculator.add_number(np.mean(result))

            if self.steps % 10 == 0:
                if mean > 50 and abs(mean - self.last_mean) < std:
                    self.platform_alarm = True
                else:
                    self.platform_alarm = False

                self.last_mean = mean

    def save_results(self):
        y = np.mean(self.results, axis = 0)
        error = np.std(self.results, axis = 0)

        x = range(0, self.results.shape[1] * self.interval, self.interval)
        fig, ax = plt.subplots(1, 1, figsize = (6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr = error, fmt = '-o')
        plt.savefig(self.save_fig_name)
        savemat(self.save_file_name, {'reward': self.results})

        plt.close(fig)

    def save_eval_result(self, results, scores, step_counts = None):
        if not os.path.exists(os.path.join("logs", self.args.agent)):
            os.mkdir(os.path.join("logs", self.args.agent))

        if not os.path.exists(os.path.join("logs", self.args.agent, "mats")):
            os.mkdir(os.path.join("logs", self.args.agent, "mats"))
        savemat(os.path.join("logs", self.args.agent, "mats", str(self.args.level_idx) + ".mat"),
                {'results': results, "scores": scores})

        if step_counts is None:
            return

        data = np.reshape(np.array(step_counts), (-1,))

        plt.hist(data, bins = 20, range = (0, 50))
        if self.args.env_name == "HappyElimination":
            plt.title(self.args.level_idx)
            if not os.path.exists(os.path.join("logs", self.args.agent, "bar_50")):
                os.mkdir(os.path.join("logs", self.args.agent, "bar_50"))
            plt.savefig(os.path.join("logs", self.args.agent, "bar_50", "bar_50_level_" + str(self.args.level_idx) + ".png"))
        plt.close()

        plt.hist(data, bins = 20, range = (0, 100))
        if self.args.env_name == "HappyElimination":
            plt.title(self.args.level_idx)
            if not os.path.exists(os.path.join("logs", self.args.agent, "bar_100")):
                os.mkdir(os.path.join("logs", self.args.agent, "bar_100"))
            plt.savefig(os.path.join("logs", self.args.agent, "bar_100", "bar_100_level_" + str(self.args.level_idx) + ".png"))
        plt.close()

        plt.hist(data, bins = 20, range = (0, 200))
        if self.args.env_name == "HappyElimination":
            plt.title(self.args.level_idx)
            if not os.path.exists(os.path.join("logs", self.args.agent, "bar_200")):
                os.mkdir(os.path.join("logs", self.args.agent, "bar_200"))
            plt.savefig(os.path.join("logs", self.args.agent, "bar_200", "bar_200_level_" + str(self.args.level_idx) + ".png"))
        plt.close()

    def enable_platform_sensor(self):
        self.movingAvegCalculator = MovingAvegCalculator(window_length = 20)

    def check_platform(self):
        return self.platform_alarm



