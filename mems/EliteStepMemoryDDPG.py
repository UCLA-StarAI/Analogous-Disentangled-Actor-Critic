import torch
import numpy as np
import warnings
import random
from copy import deepcopy
import math

from mems.StepMemoryWithSeqRewardOnly import StepMemoryWithSeqRewardOnly
from mems.StepMemoryWithSeqRewardOnlyHE import StepMemoryWithSeqRewardOnlyHE


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


class EliteStepMemoryDDPG():
    def __init__(self, max_episode_num, device = None, mem_type = 0):
        self.max_episode_num = max_episode_num
        self.device = device
        self.mem_type = mem_type

        # Reference score
        self.averager = MovingAvegCalculator(2000)

        # Current score
        self.current_score = 0.0

        # Episode buffer
        self.episode_buffer = []

        # Main buffer
        self.main_buffer = [None for _ in range(max_episode_num)]
        self.episodic_score = [None for _ in range(max_episode_num)]
        self.episodic_probability = [None for _ in range(max_episode_num)]
        self.curr_episode_num = 0

    def append(self, state, action, reward, done):
        self.episode_buffer.append((state, action, reward, done))
        self.current_score += reward

        if done:
            # Update reference score
            aveg, std = self.averager.add_number(self.current_score)

            if self.current_score > aveg + std:
                # Record current episode
                if self.curr_episode_num < self.max_episode_num:
                    # append directly
                    mem_idx = self.curr_episode_num
                else:
                    mem_idx = np.argmin(self.episodic_score)
                self.curr_episode_num += 1

                self.main_buffer[mem_idx] = self.episode_to_step_memory(episode_buffer = self.episode_buffer)
                self.episodic_score[mem_idx] = self.current_score

            self.episode_buffer.clear()
            self.current_score = 0.0

            # Update episodic probability
            sum_episodic_probability = 0.0
            for i in range(min(self.curr_episode_num, self.max_episode_num)):
                self.episodic_probability[i] = (self.episodic_score[i] - aveg) / std
                self.episodic_probability[i] = max(min(self.episodic_probability[i], 5), -5)
                self.episodic_probability[i] = math.exp(self.episodic_probability[i])
                sum_episodic_probability += self.episodic_probability[i]
            for i in range(min(self.curr_episode_num, self.max_episode_num)):
                self.episodic_probability[i] /= sum_episodic_probability

    def episode_to_step_memory(self, episode_buffer):
        if self.mem_type == 0:
            mem = StepMemoryWithSeqRewardOnly(max_buffer_size=len(self.episode_buffer), device=self.device)
        elif self.mem_type == 1:
            mem = StepMemoryWithSeqRewardOnlyHE(max_buffer_size=len(self.episode_buffer), device=self.device)

        for item in episode_buffer:
            mem.append(item[0], item[1], item[2], item[3])
        mem.append(item[0], item[1], item[2], item[3])

        return mem

    def sample(self, batch_size, to_tensor = True):
        if self.curr_episode_num == 0:
            return None, None, None, None, None, None

        # Sampling
        idx = self.multinomial(self.episodic_probability)
        return self.main_buffer[idx].sample(batch_size, to_tensor = to_tensor)

    def multinomial(self, p):
        num = np.random.random()
        for i in range(len(p)):
            if p[i] is None:
                return i - 1
            if num < p[i]:
                return i
            else:
                num -= p[i]
        return len(p) - 1
