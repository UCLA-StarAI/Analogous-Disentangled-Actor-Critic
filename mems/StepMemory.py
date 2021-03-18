import math
from copy import deepcopy

from mems.Memory import Memory

from collections import deque


class StepMemory(Memory):
    def __init__(self, max_buffer_size, device = None, use_priorized_heap = False, unroll_length = 1, gamma = 0.99):
        items_name = ["state", "action", "reward", "next_state", "done"]
        max_sampling_count = 1000

        super(StepMemory, self).__init__(items_name,
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

        self.gamma = gamma

        self.unroll_length = unroll_length
        if unroll_length > 1:
            self.temporal_storage_buffer = deque()
            self.temporal_cumalative_reward = 0.0

        self.no_done_sample = False

    def append(self, state, action, reward, done, prior_heap_key = 0.0):
        if self.buffer_t["state"] is not None:
            self.buffer_t["next_state"] = deepcopy(state)

            if self.unroll_length <= 1:
                # Append to memory
                super(StepMemory, self).append(self.buffer_t, prior_heap_key = prior_heap_key)
            else:
                self.temporal_storage_buffer.append(
                    [deepcopy(self.buffer_t["state"]),
                     deepcopy(self.buffer_t["action"]),
                     deepcopy(self.buffer_t["reward"]),
                     deepcopy(self.buffer_t["done"])]
                )

                self.temporal_cumalative_reward += self.buffer_t["reward"] * math.pow(self.gamma, self.unroll_length - 1)

                if self.buffer_t["done"]:
                    accu_reward = 0.0

                    final_state = self.buffer_t["next_state"]

                    while len(self.temporal_storage_buffer) > 0:
                        temp_buffer = self.temporal_storage_buffer.pop()
                        temp_state = temp_buffer[0]
                        temp_action = temp_buffer[1]
                        temp_reward = temp_buffer[2]

                        accu_reward = temp_reward + self.gamma * accu_reward

                        buffer = {
                            "state": temp_state,
                            "action": temp_action,
                            "reward": accu_reward,
                            "done": True,
                            "next_state": deepcopy(final_state)
                        }

                        super(StepMemory, self).append(buffer, prior_heap_key)

                    self.temporal_cumalative_reward = 0.0

                elif len(self.temporal_storage_buffer) >= self.unroll_length:
                    temp_buffer = self.temporal_storage_buffer.popleft()
                    temp_state = temp_buffer[0]
                    temp_action = temp_buffer[1]
                    temp_reward = temp_buffer[2]

                    buffer = {
                        "state": temp_state,
                        "action": temp_action,
                        "reward": self.temporal_cumalative_reward,
                        "done": temp_reward,
                        "next_state": deepcopy(self.buffer_t["next_state"])
                    }

                    self.temporal_cumalative_reward -= temp_reward
                    self.temporal_cumalative_reward /= self.gamma

                    super(StepMemory, self).append(buffer, prior_heap_key)

                else:
                    self.temporal_cumalative_reward /= self.gamma

        self.buffer_t["state"] = deepcopy(state)
        self.buffer_t["action"] = deepcopy(action)
        self.buffer_t["reward"] = deepcopy(reward)
        self.buffer_t["done"] = deepcopy(done)

    def sample(self, batch_size, to_tensor = True, no_done_sample = False):
        self.no_done_sample = no_done_sample

        batch = super(StepMemory, self).sample(batch_size, to_tensor)

        if batch is None:
            return None, None, None, None, None
        else:
            return (batch["state"], batch["action"], batch["reward"],
                    batch["next_state"], batch["done"])

    def sampling_condition(self, idx):
        return (not self.no_done_sample) or (not self.buffers["done"][idx])


