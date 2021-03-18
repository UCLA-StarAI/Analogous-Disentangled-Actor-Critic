from copy import deepcopy

from mems.Memory import Memory


class StepMemoryDemo(Memory):
    def __init__(self, max_buffer_size, device = None, use_priorized_heap = False, gamma = 0.99):
        items_name = ["state", "action", "reward", "seq_reward",
                      "next_state", "done", "pos", "seq_reward_indicator"]
        max_sampling_count = 1000

        super(StepMemoryDemo, self).__init__(items_name,
                                         max_buffer_size = max_buffer_size,
                                         max_sampling_count = max_sampling_count,
                                         device = device,
                                         use_priorized_heap = use_priorized_heap)

        # Temporary items
        self.buffer_t = dict()
        self.buffer_t["state"] = None
        self.buffer_t["action"] = None
        self.buffer_t["reward"] = None
        self.buffer_t["seq_reward"] = 0.0
        self.buffer_t["next_state"] = None
        self.buffer_t["done"] = None
        self.buffer_t["pos"] = None
        self.buffer_t["seq_reward_indicator"] = False

        # T count
        self.t_count = 0

        self.gamma = gamma

    def append(self, state, action, reward, done, prior_heap_key = 0.0):
        self.t_count += 1

        if self.buffer_t["state"] is not None:
            self.buffer_t["next_state"] = deepcopy(state)

            # Append to memory
            super(StepMemoryDemo, self).append(self.buffer_t, prior_heap_key = prior_heap_key)

            # Generate sequential reward
            if self.buffer_t["done"]:
                self.generate_seq_reward_for_episode()

        self.buffer_t["state"] = deepcopy(state)
        self.buffer_t["action"] = deepcopy(action)
        self.buffer_t["reward"] = deepcopy(reward)
        self.buffer_t["done"] = deepcopy(done)
        self.buffer_t["pos"] = self.t_count

        if done:
            self.t_count = 0

    def sample(self, batch_size, to_tensor = True):
        batch = super(StepMemoryDemo, self).sample(batch_size, to_tensor)

        if batch is None:
            return None, None, None, None, None, None
        else:
            return (batch["state"], batch["action"], batch["reward"],
                    batch["seq_reward"], batch["next_state"], batch["done"], batch["pos"])

    def sampling_condition(self, idx):
        return True

    def generate_seq_reward_for_episode(self):
        curr_idx = self.buffer_size - 1
        curr_accu_reward = 0.0
        while curr_idx >= 0 and (curr_idx >= 1 and (not self.buffers["done"][curr_idx - 1])):
            curr_accu_reward = self.buffers["reward"][curr_idx] + self.gamma * curr_accu_reward
            self.buffers["seq_reward"][curr_idx] = curr_accu_reward
            self.buffers["seq_reward_indicator"][curr_idx] = True

            curr_idx -= 1
