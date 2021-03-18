import numpy as np
from copy import deepcopy

from mems.Memory import Memory


class StepMemoryWithSeqRewardOnlyHE(Memory):
    def __init__(self, max_buffer_size, gamma = 0.99, device = None):
        items_name = ["state_img", "state_vec", "action", "reward", "seq_reward",
                      "next_state_img", "next_state_vec", "done", "seq_reward_indicator"]
        max_sampling_count = 1000

        super(StepMemoryWithSeqRewardOnlyHE, self).__init__(items_name,
                                                      max_buffer_size = max_buffer_size,
                                                      max_sampling_count = max_sampling_count,
                                                      device = device)

        # Temporary items
        self.buffer_t = dict()
        self.buffer_t["state_img"] = None
        self.buffer_t["state_vec"] = None
        self.buffer_t["action"] = None
        self.buffer_t["reward"] = None
        self.buffer_t["seq_reward"] = 0.0
        self.buffer_t["next_state_img"] = None
        self.buffer_t["next_state_vec"] = None
        self.buffer_t["done"] = None
        self.buffer_t["seq_reward_indicator"] = False

        # Discount factor
        self.gamma = gamma

    def append(self, state, action, reward, done):
        if self.buffer_t["state_img"] is not None:
            self.buffer_t["next_state_img"] = deepcopy(state[0])
            self.buffer_t["next_state_vec"] = deepcopy(state[1])

            if self.buffer_t["done"]:
                if isinstance(self.buffer_t["next_state_img"], list):
                    for item in self.buffer_t["next_state_img"]:
                        item *= 0.0
                    for item in self.buffer_t["next_state_vec"]:
                        item *= 0.0

            # Append to memory
            super(StepMemoryWithSeqRewardOnlyHE, self).append(self.buffer_t)

            # Generate sequential reward
            if self.buffer_t["done"]:
                self.generate_seq_reward_for_episode()

        self.buffer_t["state_img"] = deepcopy(state[0])
        self.buffer_t["state_vec"] = deepcopy(state[1])
        self.buffer_t["action"] = deepcopy(action)
        self.buffer_t["reward"] = deepcopy(reward)
        self.buffer_t["done"] = deepcopy(done)

    def sample(self, batch_size, to_tensor = True):
        batch = super(StepMemoryWithSeqRewardOnlyHE, self).sample(
            batch_size,
            to_tensor
        )

        if batch is None:
            return None, None, None, None, None, None

        return ([batch["state_img"], batch["state_vec"]], batch["action"], batch["reward"],
                batch["seq_reward"], [batch["next_state_img"], batch["next_state_vec"]], batch["done"])

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
