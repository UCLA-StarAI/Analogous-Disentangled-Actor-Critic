import numpy as np
from copy import deepcopy

from mems.Memory import Memory


class StepMemoryWithSeqRewardOnlyDoubleReward(Memory):
    def __init__(self, max_buffer_size, gamma = 0.99, device = None):
        items_name = ["state", "action", "reward1", "reward2", "seq_reward1",
                      "seq_reward2", "next_state", "done", "seq_reward_indicator"]
        max_sampling_count = 1000

        super(StepMemoryWithSeqRewardOnlyDoubleReward, self).__init__(items_name,
                                                      max_buffer_size = max_buffer_size,
                                                      max_sampling_count = max_sampling_count,
                                                      device = device)

        # Temporary items
        self.buffer_t = dict()
        self.buffer_t["state"] = None
        self.buffer_t["action"] = None
        self.buffer_t["reward1"] = None
        self.buffer_t["reward2"] = None
        self.buffer_t["seq_reward1"] = 0.0
        self.buffer_t["seq_reward2"] = 0.0
        self.buffer_t["next_state"] = None
        self.buffer_t["done"] = None
        self.buffer_t["seq_reward_indicator"] = False

        # Discount factor
        self.gamma = gamma

    def append(self, state, action, reward1, reward2, done):
        if self.buffer_t["state"] is not None:
            self.buffer_t["next_state"] = deepcopy(state)

            if self.buffer_t["done"]:
                if isinstance(self.buffer_t["next_state"], list):
                    for item in self.buffer_t["next_state"]:
                        item *= 0.0

            # Append to memory
            super(StepMemoryWithSeqRewardOnlyDoubleReward, self).append(self.buffer_t)

            # Generate sequential reward
            if self.buffer_t["done"]:
                self.generate_seq_reward_for_episode()

        self.buffer_t["state"] = deepcopy(state)
        self.buffer_t["action"] = deepcopy(action)
        self.buffer_t["reward1"] = deepcopy(reward1)
        self.buffer_t["reward2"] = deepcopy(reward2)
        self.buffer_t["done"] = deepcopy(done)

    def sample(self, batch_size, to_tensor = True):
        batch = super(StepMemoryWithSeqRewardOnlyDoubleReward, self).sample(
            batch_size,
            to_tensor
        )

        return (batch["state"], batch["action"], batch["reward1"], batch["reward2"],
                batch["seq_reward1"], batch["seq_reward2"], batch["next_state"], batch["done"])

    def sampling_condition(self, idx):
        return self.buffers["seq_reward_indicator"][idx]

    def generate_seq_reward_for_episode(self):
        curr_idx = self.buffer_size - 1
        curr_accu_reward1 = 0.0
        curr_accu_reward2 = 0.0
        while curr_idx >= 0 and (curr_idx >= 1 and (not self.buffers["done"][curr_idx - 1])):
            curr_accu_reward1 = self.buffers["reward1"][curr_idx] + self.gamma * curr_accu_reward1
            curr_accu_reward2 = self.buffers["reward2"][curr_idx] + self.gamma * curr_accu_reward2
            self.buffers["seq_reward1"][curr_idx] = curr_accu_reward1
            self.buffers["seq_reward2"][curr_idx] = curr_accu_reward2
            self.buffers["seq_reward_indicator"][curr_idx] = True

            curr_idx -= 1
