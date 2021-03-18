import numpy as np
from copy import deepcopy

from mems.Memory import Memory


class StepMemoryWithSeqRewardOnly(Memory):
    def __init__(self, max_buffer_size, gamma = 0.99, device = None):
        items_name = ["state", "action", "reward", "seq_reward",
                      "next_state", "done", "seq_reward_indicator"]
        max_sampling_count = 1000

        super(StepMemoryWithSeqRewardOnly, self).__init__(items_name,
                                                      max_buffer_size = max_buffer_size,
                                                      max_sampling_count = max_sampling_count,
                                                      device = device)

        # Temporary items
        self.buffer_t = dict()
        self.buffer_t["state"] = None
        self.buffer_t["action"] = None
        self.buffer_t["reward"] = None
        self.buffer_t["seq_reward"] = 0.0
        self.buffer_t["next_state"] = None
        self.buffer_t["done"] = None
        self.buffer_t["seq_reward_indicator"] = False

        # Discount factor
        self.gamma = gamma

    def append(self, state, action, reward, done):
        if self.buffer_t["state"] is not None:
            self.buffer_t["next_state"] = deepcopy(state)

            if self.buffer_t["done"]:
                if isinstance(self.buffer_t["next_state"], list):
                    for item in self.buffer_t["next_state"]:
                        item *= 0.0

            # Append to memory
            super(StepMemoryWithSeqRewardOnly, self).append(self.buffer_t)

            # Generate sequential reward
            if self.buffer_t["done"]:
                self.generate_seq_reward_for_episode()

        self.buffer_t["state"] = deepcopy(state)
        self.buffer_t["action"] = deepcopy(action)
        self.buffer_t["reward"] = deepcopy(reward)
        self.buffer_t["done"] = deepcopy(done)

    def sample(self, batch_size, to_tensor = True):
        batch = super(StepMemoryWithSeqRewardOnly, self).sample(
            batch_size,
            to_tensor
        )

        return (batch["state"], batch["action"], batch["reward"],
                batch["seq_reward"], batch["next_state"], batch["done"])

    def sampling_condition(self, idx):
        return self.buffers["seq_reward_indicator"][idx]

    def generate_seq_reward_for_episode(self):
        curr_idx = self.buffer_size - 1
        curr_accu_reward = 0.0
        while curr_idx >= 0 and (curr_idx >= 1 and (not self.buffers["done"][curr_idx - 1])):
            curr_accu_reward = self.buffers["reward"][curr_idx] + self.gamma * curr_accu_reward
            self.buffers["seq_reward"][curr_idx] = curr_accu_reward
            self.buffers["seq_reward_indicator"][curr_idx] = True

            curr_idx -= 1
