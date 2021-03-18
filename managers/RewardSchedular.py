import math

from utils.ColoredPrintFunctions import *


class MovingAvegCalculator():
    def __init__(self, window_length):
        self.num_added = 0
        self.window_length = window_length
        self.window = [0.0 for _ in range(window_length)]

        self.aveg = 0.0
        self.var = 0.0

    def add_number(self, num):
        idx = self.num_added % self.window_length
        old_num = self.window[idx]
        self.window[idx] = num
        self.num_added += 1

        old_aveg = self.aveg
        if self.num_added <= self.window_length:
            delta = num - old_aveg
            self.aveg += delta / self.num_added
            self.var += delta * (num - self.aveg)
        else:
            delta = num - old_num
            self.aveg += delta / self.window_length
            self.var += delta * ((num - self.aveg) + (old_num - old_aveg))

        if self.num_added <= self.window_length:
            if self.num_added == 1:
                variance = 0.1
            else:
                variance = self.var / (self.num_added - 1)
        else:
            variance = self.var / self.window_length

        try:
            std = math.sqrt(variance)
            if math.isnan(std):
                std = 0.1
        except:
            std = 0.1

        return self.aveg, std


class RewardSchedular():
    def __init__(self, factors = [0.6, 1.1], tol_num = 4, window_length = 100, gamma = 0.99):
        self.curr_factor = 1.0

        assert factors[0] < 1.0 and factors[1] > 1.0
        self.factors = factors

        self.correct_reward_averager = MovingAvegCalculator(window_length = window_length)
        self.support_reward_averager = MovingAvegCalculator(window_length = window_length)

        self.gamma = gamma

        self.episode_correct_reward = 0.0
        self.episode_support_reward = 0.0
        self.episode_gamma = 1.0

        self.reward1_baseline = -1000.0
        self.reward1_tol = 0.0

        self.reward2_baseline = -1000.0
        self.reward2_tol = 0.0

        self.tol_num = tol_num
        self.curr_tol_num1 = 0
        self.curr_tol_num2 = 0

        self.count = 0
        self.window_length = window_length

        self.cooldown = window_length

    def step(self, correct_reward, support_reward, done):
        self.episode_correct_reward += correct_reward * self.episode_gamma
        self.episode_support_reward += support_reward * self.episode_gamma

        if done:
            reward1, std1 = self.correct_reward_averager.add_number(self.episode_correct_reward)
            reward2, std2 = self.support_reward_averager.add_number(self.episode_support_reward)

            self.count += 1

            if self.cooldown > 0:
                self.cooldown -= 1

            elif self.count > self.window_length and self.episode_correct_reward < self.reward1_baseline - self.reward1_tol:
                if self.episode_support_reward > self.reward2_baseline + 0.0 * self.reward2_tol:
                    self.curr_tol_num1 += 1
                    self.curr_tol_num2 = 0
                    if self.curr_tol_num1 > self.tol_num:
                        self.curr_tol_num1 = 0

                        self.curr_factor *= self.factors[0]

                        self.reward1_baseline = reward1
                        self.reward1_tol = std1
                        self.reward2_baseline = reward2
                        self.reward2_tol = std2

                        self.cooldown = self.window_length

                        prRed("> Reward factor changed to: {}".format(self.curr_factor))

                elif self.episode_support_reward < self.reward2_baseline - self.reward2_tol:
                    self.curr_tol_num2 += 1
                    self.curr_tol_num1 = 0
                    if self.curr_tol_num2 > self.tol_num:
                        self.curr_tol_num2 = 0

                        self.curr_factor *= self.factors[1]

                        self.reward1_baseline = reward1
                        self.reward1_tol = std1
                        self.reward2_baseline = reward2
                        self.reward2_tol = std2

                        self.cooldown = self.window_length

                        prRed("> Reward factor changed to: {}".format(self.curr_factor))

            elif self.count > self.window_length and self.episode_correct_reward > self.reward1_baseline + self.reward1_tol:
                self.reward1_baseline = reward1
                self.reward1_tol = std1
                self.reward2_baseline = reward2
                self.reward2_tol = std2

                self.curr_tol_num1 = 0
                self.curr_tol_num2 = 0

            elif self.count > self.window_length:
                self.curr_tol_num1 = 0
                self.curr_tol_num2 = 0

            self.episode_correct_reward = 0.0
            self.episode_support_reward = 0.0
            self.episode_gamma = 1.0
        else:
            self.episode_gamma *= self.gamma

        return self.curr_factor
