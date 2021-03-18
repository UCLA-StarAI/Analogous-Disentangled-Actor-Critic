import numpy as np
from copy import deepcopy

from mems.Memory import Memory


class StepMemoryWithSeqReward(Memory):
    def __init__(self, max_buffer_size, gamma = 0.99, device = None):
        items_name = ["state", "action", "action_prob", "reward", "seq_reward",
                      "next_state", "done", "seq_reward_indicator"]
        max_sampling_count = 1000

        super(StepMemoryWithSeqReward, self).__init__(items_name,
                                                      max_buffer_size = max_buffer_size,
                                                      max_sampling_count = max_sampling_count,
                                                      device = device)

        # Temporary items
        self.buffer_t = dict()
        self.buffer_t["state"] = None
        self.buffer_t["action"] = None
        self.buffer_t["action_prob"] = None
        self.buffer_t["reward"] = None
        self.buffer_t["seq_reward"] = 0.0
        self.buffer_t["next_state"] = None
        self.buffer_t["done"] = None
        self.buffer_t["seq_reward_indicator"] = False

        # Discount factor
        self.gamma = gamma

    def append(self, state, action, action_prob, reward, done):
        if self.buffer_t["state"] is not None:
            self.buffer_t["next_state"] = deepcopy(state)

            if self.buffer_t["done"]:
                if isinstance(self.buffer_t["next_state"], list):
                    for item in self.buffer_t["next_state"]:
                        item *= 0.0

            # Append to memory
            super(StepMemoryWithSeqReward, self).append(self.buffer_t)

            # Generate sequential reward
            if self.buffer_t["done"]:
                self.generate_seq_reward_for_episode()

        self.buffer_t["state"] = deepcopy(state)
        self.buffer_t["action"] = deepcopy(action)
        self.buffer_t["action_prob"] = deepcopy(action_prob)
        self.buffer_t["reward"] = deepcopy(reward)
        self.buffer_t["done"] = deepcopy(done)

    def sample(self, batch_size, to_tensor = True, recent_sample_prob = 0.5, recent_sample_threshold = 1000):
        recent_sample_num = int(batch_size * recent_sample_prob)
        non_recent_sample_num = batch_size - recent_sample_num

        batch1, batch_idxs1 = super(StepMemoryWithSeqReward, self).sample(
            recent_sample_num,
            to_tensor,
            return_idxs = True,
            sample_recent = recent_sample_threshold
        )

        batch2, batch_idxs2 = super(StepMemoryWithSeqReward, self).sample(
            non_recent_sample_num,
            to_tensor,
            return_idxs = True
        )

        batch = dict()
        for key in batch1.keys():
            item1 = batch1[key]
            item2 = batch2[key]
            if isinstance(item1, list):
                item = []
                for mini_item1, mini_item2 in zip(item1, item2):
                    item.append(np.concatenate((mini_item1, mini_item2), axis = 0))
            else:
                item = np.concatenate((item1, item2), axis = 0)

            batch[key] = item

        batch_idxs = np.concatenate((batch_idxs1, batch_idxs2), axis = 0)

        return (batch["state"], batch["action"], batch["action_prob"], batch["reward"],
                batch["seq_reward"], batch["next_state"], batch["done"], batch_idxs)

    def sampling_condition(self, idx):
        return self.buffers["seq_reward_indicator"][idx]

    def generate_seq_reward_for_episode(self):
        curr_idx = self.buffer_size - 1
        curr_accu_reward = 0.0
        curr_action_prob = 1.0
        while curr_idx >= 0 and (curr_idx >= 1 and (not self.buffers["done"][curr_idx - 1])):
            curr_accu_reward = self.buffers["reward"][curr_idx] + self.gamma * curr_accu_reward
            self.buffers["seq_reward"][curr_idx] = curr_accu_reward
            self.buffers["seq_reward_indicator"][curr_idx] = True

            curr_action_prob *= self.buffers["action_prob"][curr_idx]
            self.buffers["action_prob"][curr_idx] = curr_action_prob

            curr_idx -= 1
