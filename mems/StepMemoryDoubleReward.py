import math
from copy import deepcopy

from mems.Memory import Memory

from collections import deque


class StepMemoryDoubleReward(Memory):
    def __init__(self, max_buffer_size, device = None, use_priorized_heap = False, unroll_length = 1, gamma = 0.99):
        items_name = ["state", "action", "reward1", "reward2", "next_state", "done"]
        max_sampling_count = 1000

        super(StepMemoryDoubleReward, self).__init__(items_name,
                                                     max_buffer_size = max_buffer_size,
                                                     max_sampling_count = max_sampling_count,
                                                     device = device,
                                                     use_priorized_heap = use_priorized_heap)

        # Temporary items
        self.buffer_t = dict()
        self.buffer_t["state"] = None
        self.buffer_t["action"] = None
        self.buffer_t["reward1"] = None
        self.buffer_t["reward2"] = None
        self.buffer_t["next_state"] = None
        self.buffer_t["done"] = None

        self.gamma = gamma

        self.unroll_length = unroll_length
        if unroll_length > 1:
            self.temporal_storage_buffer = deque()
            self.temporal_cumalative_reward1 = 0.0
            self.temporal_cumalative_reward2 = 0.0

        self.no_done_sample = False

    def append(self, state, action, reward1, reward2, done, prior_heap_key = 0.0):
        if self.buffer_t["state"] is not None:
            self.buffer_t["next_state"] = deepcopy(state)

            if self.unroll_length <= 1:
                # Append to memory
                super(StepMemoryDoubleReward, self).append(self.buffer_t, prior_heap_key = prior_heap_key)
            else:
                self.temporal_storage_buffer.append(
                    [deepcopy(self.buffer_t["state"]),
                     deepcopy(self.buffer_t["action"]),
                     deepcopy(self.buffer_t["reward1"]),
                     deepcopy(self.buffer_t["reward2"]),
                     deepcopy(self.buffer_t["done"])]
                )

                self.temporal_cumalative_reward1 += self.buffer_t["reward1"] * \
                                                    math.pow(self.gamma, self.unroll_length - 1)
                self.temporal_cumalative_reward2 += self.buffer_t["reward2"] * \
                                                    math.pow(self.gamma, self.unroll_length - 1)

                if self.buffer_t["done"]:
                    accu_reward1 = 0.0
                    accu_reward2 = 0.0

                    final_state = self.buffer_t["next_state"]

                    while len(self.temporal_storage_buffer) > 0:
                        temp_buffer = self.temporal_storage_buffer.pop()
                        temp_state = temp_buffer[0]
                        temp_action = temp_buffer[1]
                        temp_reward1 = temp_buffer[2]
                        temp_reward2 = temp_buffer[3]

                        accu_reward1 = temp_reward1 + self.gamma * accu_reward1
                        accu_reward2 = temp_reward2 + self.gamma * accu_reward2

                        buffer = {
                            "state": temp_state,
                            "action": temp_action,
                            "reward1": accu_reward1,
                            "reward2": accu_reward2,
                            "done": True,
                            "next_state": deepcopy(final_state)
                        }

                        super(StepMemoryDoubleReward, self).append(buffer, prior_heap_key)

                    self.temporal_cumalative_reward1 = 0.0
                    self.temporal_cumalative_reward2 = 0.0

                elif len(self.temporal_storage_buffer) >= self.unroll_length:
                    temp_buffer = self.temporal_storage_buffer.popleft()
                    temp_state = temp_buffer[0]
                    temp_action = temp_buffer[1]
                    temp_reward1 = temp_buffer[2]
                    temp_reward2 = temp_buffer[3]

                    buffer = {
                        "state": temp_state,
                        "action": temp_action,
                        "reward1": self.temporal_cumalative_reward1,
                        "reward2": self.temporal_cumalative_reward2,
                        "done": False,
                        "next_state": deepcopy(self.buffer_t["next_state"])
                    }

                    self.temporal_cumalative_reward1 -= temp_reward1
                    self.temporal_cumalative_reward1 /= self.gamma
                    self.temporal_cumalative_reward2 -= temp_reward2
                    self.temporal_cumalative_reward2 /= self.gamma

                    super(StepMemoryDoubleReward, self).append(buffer, prior_heap_key)

                else:
                    self.temporal_cumalative_reward1 /= self.gamma
                    self.temporal_cumalative_reward2 /= self.gamma

        self.buffer_t["state"] = deepcopy(state)
        self.buffer_t["action"] = deepcopy(action)
        self.buffer_t["reward1"] = deepcopy(reward1)
        self.buffer_t["reward2"] = deepcopy(reward2)
        self.buffer_t["done"] = deepcopy(done)

    def sample(self, batch_size, to_tensor = True, no_done_sample = False):
        self.no_done_sample = no_done_sample

        batch = super(StepMemoryDoubleReward, self).sample(batch_size, to_tensor)

        if batch is None:
            return None, None, None, None, None, None
        else:
            return (batch["state"], batch["action"], batch["reward1"], batch["reward2"],
                    batch["next_state"], batch["done"])

    def sampling_condition(self, idx):
        return (not self.no_done_sample) or (not self.buffers["done"][idx])


