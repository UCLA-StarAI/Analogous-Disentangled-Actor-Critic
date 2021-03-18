from copy import deepcopy
import math
import torch
import numpy as np

from mems.Memory import Memory


class CriticImproveStepMemory(Memory):
    def __init__(self, max_buffer_size, device = None, use_priorized_heap = False, collect_interval = 16, gamma = 0.99):
        items_name = ["state", "action", "reward", "next_state", "done",
                      "later_state", "later_gamma", "accu_reward",
                      "accu_action_prob", "ready_for_sample"]
        max_sampling_count = 1000

        super(CriticImproveStepMemory, self).__init__(items_name,
                                         max_buffer_size = max_buffer_size,
                                         max_sampling_count = max_sampling_count,
                                         device = device,
                                         use_priorized_heap = use_priorized_heap)

        # Temporary items
        self.buffer_t = dict()
        self.buffer_t["state"] = None
        self.buffer_t["action"] = None
        self.buffer_t["reward"] = None
        self.buffer_t["next_state"] = None
        self.buffer_t["done"] = None
        self.buffer_t["later_state"] = None
        self.buffer_t["later_gamma"] = None
        self.buffer_t["accu_reward"] = None
        self.buffer_t["ready_for_sample"] = False

        self.gamma = gamma

        self.collect_interval = collect_interval
        self.buffer_count = 0

    def append(self, state, action, reward, done, action_prob, prior_heap_key = 0.0):
        if self.buffer_t["state"] is not None:
            self.buffer_t["next_state"] = deepcopy(state)

            # Append to memory
            super(CriticImproveStepMemory, self).append(self.buffer_t, prior_heap_key = prior_heap_key)

        self.buffer_t["state"] = deepcopy(state)
        self.buffer_t["action"] = deepcopy(action)
        self.buffer_t["reward"] = deepcopy(reward)
        self.buffer_t["done"] = deepcopy(done)
        self.buffer_t["accu_action_prob"] = deepcopy(action_prob)

        if self.buffer_count != 0 and self.buffer_count % self.collect_interval == 0:
            idx = self.buffer_size - 2
            accu_reward = 0.0
            k = 0
            curr_later_state = deepcopy(self.buffers["next_state"][idx])
            accu_action_prob = []
            accu_action_prob2 = 1.0
            for j in range(self.collect_interval):
                # Fill accu_reward
                if not self.buffers["done"][idx]:
                    accu_reward = self.gamma * accu_reward + self.buffers["reward"][idx]
                    k += 1
                    accu_action_prob2 = 1.0
                else:
                    accu_reward = self.buffers["reward"][idx]
                    k = 1
                self.buffers["accu_reward"][idx] = accu_reward
                self.buffers["later_state"][idx] = deepcopy(curr_later_state)
                self.buffers["later_gamma"][idx] = math.pow(self.gamma, k) if k != 0 else 0
                self.buffers["ready_for_sample"][idx] = True

                if self.buffers["done"][idx]:
                    accu_action_prob = [self.buffers["accu_action_prob"][idx]]
                else:
                    for k in range(len(accu_action_prob)):
                        accu_action_prob[k] *= self.buffers["accu_action_prob"][idx]
                        accu_action_prob.append(self.buffers["accu_action_prob"][idx])

                total_weight = 1e-8
                total_coeff = 0.0
                for j in range(len(accu_action_prob)):
                    curr_weight = math.pow(self.gamma, len(accu_action_prob) - 1 - j)
                    total_coeff += curr_weight * accu_action_prob[j]
                    total_weight += curr_weight

                accu_action_prob2 *= self.buffers["accu_action_prob"][idx]

                mode = 1
                if mode == 0:
                    self.buffers["accu_action_prob"][idx] = total_coeff / total_weight
                else:
                    self.buffers["accu_action_prob"][idx] = accu_action_prob2

        self.buffer_count += 1

    def sample(self, batch_size, to_tensor = True, sample_recent_threshold = 0, sample_recent_prob = 0.0):
        recent_batch_size = int(batch_size * sample_recent_prob)

        recent_batch, recent_batch_idxs = super(CriticImproveStepMemory, self).sample(recent_batch_size, to_tensor,
                                                                   return_idxs = True,
                                                                   sample_recent = sample_recent_threshold)
        other_batch, other_batch_idxs = super(CriticImproveStepMemory, self).sample(batch_size - recent_batch_size,
                                                                   to_tensor,
                                                                   return_idxs = True)
        batch = dict()
        for key in recent_batch:
            batch[key] = torch.cat((recent_batch[key], other_batch[key]), dim = 0)
        batch_idxs = np.concatenate((recent_batch_idxs, other_batch_idxs), axis = 0)
        batch_idxs = torch.tensor(batch_idxs, dtype = torch.int64)

        if batch is None:
            return None, None, None, None, None
        else:
            return (batch["state"], batch["action"], batch["reward"],
                    batch["next_state"], batch["done"], batch["accu_reward"],
                    batch["later_state"], batch["later_gamma"], batch["accu_action_prob"], batch_idxs)

    def sampling_condition(self, idx):
        return self.buffers["ready_for_sample"][idx]
